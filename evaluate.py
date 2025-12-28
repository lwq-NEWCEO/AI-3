import torch
import torch.nn as nn
import argparse
import os
import sys
import json
import time
import math
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from bert_score import score as bert_score_calc
from torchmetrics.text.rouge import ROUGEScore

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.data_utils import DataManager
from models.transformer_model import Seq2SeqTransformer


# ==========================================
# 🛠️ 辅助函数：计算参数量
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


# ==========================================
# 📉 核心功能：计算 PPL (Teacher Forcing)
# ==========================================
def calculate_loss_ppl(model, iterator, criterion, device):
    """
    计算验证集/测试集的 Loss 和 PPL。
    【修改点】新增返回 batch_losses 列表，用于绘制 Loss 曲线。
    """
    model.eval()
    epoch_loss = 0
    total_tokens = 0  # 统计非 pad 的 token 数

    # 【修改 1】初始化列表用于存储每个 batch 的 loss
    batch_losses = []

    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Calculating PPL & Loss"):
            src, trg = src.long().to(device), trg.long().to(device)

            # 准备输入输出
            # trg_input: [sos, w1, w2, ...] (去掉最后一个)
            # trg_label: [w1, w2, ..., eos] (去掉第一个)
            trg_input = trg[:, :-1]
            trg_label = trg[:, 1:]

            # 创建 Mask
            src_mask = model.make_src_mask(src)
            trg_mask = model.make_trg_mask(trg_input)

            # --- Forward Pass (Teacher Forcing) ---
            # Encoder
            enc_src = model.src_embedding(src) * model.scale
            enc_src = model.pos_encoding(enc_src)
            for layer in model.encoder_layers:
                enc_src = layer(enc_src, src_mask)
            if model.pre_norm:
                enc_src = model.final_norm_enc(enc_src)

            # Decoder
            trg_emb = model.trg_embedding(trg_input) * model.scale
            trg_emb = model.pos_encoding(trg_emb)
            for layer in model.decoder_layers:
                trg_emb, _ = layer(trg_emb, enc_src, trg_mask, src_mask)
            if model.pre_norm:
                trg_emb = model.final_norm_dec(trg_emb)

            output = model.fc_out(trg_emb)  # [batch, len, vocab]

            # 计算 Loss
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_label = trg_label.contiguous().view(-1)

            loss = criterion(output, trg_label)

            # 【修改 2】记录当前 batch 的 loss (标量)
            batch_losses.append(loss.item())

            # 加权累积
            non_pad_count = (trg_label != criterion.ignore_index).sum().item()
            epoch_loss += loss.item() * non_pad_count
            total_tokens += non_pad_count

    avg_loss = epoch_loss / total_tokens if total_tokens > 0 else 0
    ppl = math.exp(avg_loss) if avg_loss < 50 else 1e9

    # 【修改 3】返回 PPL 和 batch_losses 列表
    return ppl, batch_losses


# ==========================================
# 🧠 推理核心：贪婪解码
# ==========================================
def translate_sentence(sentence_tensor, model, device, max_len=50, sos_idx=2, eos_idx=3):
    model.eval()
    src = sentence_tensor.unsqueeze(0).to(device).long()
    src_mask = model.make_src_mask(src)

    with torch.no_grad():
        enc_src = model.src_embedding(src) * model.scale
        enc_src = model.pos_encoding(enc_src)
        for layer in model.encoder_layers:
            enc_src = layer(enc_src, src_mask)
        if model.pre_norm:
            enc_src = model.final_norm_enc(enc_src)

        trg_indexes = [sos_idx]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            trg = model.trg_embedding(trg_tensor) * model.scale
            trg = model.pos_encoding(trg)
            for layer in model.decoder_layers:
                trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            if model.pre_norm:
                trg = model.final_norm_dec(trg)
            output = model.fc_out(trg)
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)
            if pred_token == eos_idx:
                break
    return trg_indexes[1:]


# ==========================================
# 📊 评估主程序 (保存全量指标)
# ==========================================
def calculate_metrics(model, iterator, src_vocab, trg_vocab, device, pad_idx, save_path=None):
    model.eval()
    trgs = []
    pred_trgs = []

    print("正在进行翻译评估 (Greedy Decoding)...")

    # --- 1. Latency & Inference ---
    start_time = time.time()
    total_samples = 0

    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Inference"):
            src, trg = src.long(), trg.long()
            batch_size = src.shape[0]
            total_samples += batch_size

            for i in range(batch_size):
                src_sentence = src[i]
                trg_sentence = trg[i]
                valid_src = src_sentence[src_sentence != pad_idx]

                pred_indices = translate_sentence(
                    valid_src, model, device,
                    sos_idx=trg_vocab.stoi['<sos>'],
                    eos_idx=trg_vocab.stoi['<eos>']
                )

                target_indices = [idx.item() for idx in trg_sentence if
                                  idx not in [pad_idx, trg_vocab.stoi['<sos>'], trg_vocab.stoi['<eos>']]]
                pred_indices = [idx for idx in pred_indices if idx != trg_vocab.stoi['<eos>']]

                pred_tokens = [trg_vocab.itos[idx] for idx in pred_indices]
                target_tokens = [trg_vocab.itos[idx] for idx in target_indices]

                pred_trgs.append(pred_tokens)
                trgs.append([target_tokens])

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000 / total_samples
    print(f"⚡ Latency: {latency_ms:.2f} ms/sample")

    # 初始化指标字典
    metrics = {
        'latency': latency_ms,
        'params': count_parameters(model)
    }

    # --- 2. Calculate PPL & Loss Curve ---
    print("\n计算 Perplexity (PPL) 和 Batch Losses...")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    # 【修改 4】接收返回的 batch_losses
    ppl, batch_losses = calculate_loss_ppl(model, iterator, criterion, device)

    metrics['ppl'] = ppl
    # 【修改 5】将 batch_losses 保存到字典中
    metrics['batch_losses'] = batch_losses

    print(f"📊 PPL: {ppl:.2f}")

    # --- 3. BLEU ---
    print("计算 BLEU Scores...")
    try:
        metrics['bleu1'] = bleu_score(pred_trgs, trgs, max_n=1, weights=[1]) * 100
        metrics['bleu2'] = bleu_score(pred_trgs, trgs, max_n=2, weights=[0.5, 0.5]) * 100
        metrics['bleu4'] = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25]) * 100
        print(f"BLEU-4: {metrics['bleu4']:.2f}")
    except:
        metrics.update({'bleu1': 0, 'bleu2': 0, 'bleu4': 0})

    # 准备字符串格式
    pred_strs = [" ".join(p) for p in pred_trgs]
    trg_strs = [" ".join(t[0]) for t in trgs]

    # --- 4. ROUGE ---
    print("计算 ROUGE-L...")
    try:
        rouge = ROUGEScore(rouge_keys=('rougeL',))
        metrics['rougeL'] = rouge(pred_strs, trg_strs)['rougeL_fmeasure'].item() * 100
        print(f"ROUGE-L: {metrics['rougeL']:.2f}")
    except Exception as e:
        print(f"ROUGE error: {e}")
        metrics['rougeL'] = 0.0

    # --- 5. BERTScore ---
    print("计算 BERTScore...")
    try:
        P, R, F1 = bert_score_calc(pred_strs, trg_strs, lang="en",
                                   model_type="microsoft/deberta-large-mnli", batch_size=16, verbose=False)
        metrics['bert_score'] = F1.mean().item() * 100
        print(f"BERTScore: {metrics['bert_score']:.2f}")
    except Exception as e:
        print(f"BERTScore error: {e}")
        metrics['bert_score'] = 0.0

    # --- 保存 ---
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(metrics, f)
        print(f"✅ 详细评估指标（含 {len(batch_losses)} 条 Loss 数据）已保存至 {save_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--pre_norm', action='store_true', help="Use Pre-Norm architecture")
    parser.add_argument('--use_test', action='store_true')
    parser.add_argument('--n_head', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--d_ff', type=int, default=512, help="FeedForward Dimension")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 强制检查测试集模式
    if args.use_test:
        print("📢 正在使用 Test Set 进行评估...")
        # 尝试传入 path，如果 DataManager 不支持会自动忽略（依赖之前的讨论，这里保持基础调用）
        try:
            dm = DataManager(path='data/test', batch_size=128)
        except TypeError:
            dm = DataManager(batch_size=128)
    else:
        dm = DataManager(batch_size=128)

    _, valid_loader, test_loader = dm.get_loaders()
    eval_iterator = test_loader if args.use_test else valid_loader

    print(f"⏳ Loading model with config: d_ff={args.d_ff}, pre_norm={args.pre_norm}")

    model = Seq2SeqTransformer(
        src_vocab=len(dm.vocab_src), trg_vocab=len(dm.vocab_trg),
        d_model=256,
        n_head=args.n_head,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=200, dropout=0.1, device=device,
        src_pad_idx=dm.PAD_IDX, trg_pad_idx=dm.PAD_IDX,
        pre_norm=args.pre_norm
    ).to(device)

    model_path = f'output/{args.exp_name}/model.pt'
    norm_type = 'Pre-Norm' if args.pre_norm else 'Post-Norm'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded successfully from {model_path} ({norm_type})")
    else:
        print(f"❌ Model not found: {model_path}")
        exit()

    save_file = f'output/{args.exp_name}/metrics.json'
    calculate_metrics(model, eval_iterator, dm.vocab_src, dm.vocab_trg, device, dm.PAD_IDX, save_file)
