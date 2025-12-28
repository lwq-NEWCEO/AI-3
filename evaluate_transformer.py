import torch
import torch.nn as nn
import argparse
import os
import sys
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from bert_score import score as bert_score_calc

# 【关键】设置 HuggingFace 镜像，解决连接超时问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.data_utils import DataManager
from models.transformer_model import Seq2SeqTransformer


# ==========================================
# 🧠 推理核心：贪婪解码 (Greedy Decoding)
# ==========================================
def translate_sentence(sentence_tensor, model, device, max_len=50, sos_idx=2, eos_idx=3):
    model.eval()

    # src: [1, src_len]
    src = sentence_tensor.unsqueeze(0).to(device).long()

    # 【修正 1】新版模型内部处理 pad_idx，不需要传入 1
    src_mask = model.make_src_mask(src)

    with torch.no_grad():
        # 1. Encoder Forward
        enc_src = model.src_embedding(src) * model.scale
        enc_src = model.pos_encoding(enc_src)
        for layer in model.encoder_layers:
            enc_src = layer(enc_src, src_mask)
        if model.pre_norm:
            enc_src = model.final_norm_enc(enc_src)

        # 2. Decoder Loop
        trg_indexes = [sos_idx]

        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            # 【修正 2】同上，不需要传入 pad_idx
            trg_mask = model.make_trg_mask(trg_tensor)

            # Decoder Forward
            trg = model.trg_embedding(trg_tensor) * model.scale
            trg = model.pos_encoding(trg)

            for layer in model.decoder_layers:
                trg, attention = layer(trg, enc_src, trg_mask, src_mask)

            if model.pre_norm:
                trg = model.final_norm_dec(trg)

            output = model.fc_out(trg)

            # 取最后一个 token 的预测结果
            pred_token = output.argmax(2)[:, -1].item()

            trg_indexes.append(pred_token)

            if pred_token == eos_idx:
                break

    return trg_indexes[1:]  # 去掉 <sos>


# ==========================================
# 📊 评估主程序
# ==========================================
def calculate_metrics(model, iterator, src_vocab, trg_vocab, device, pad_idx):
    model.eval()

    trgs = []
    pred_trgs = []

    print("正在对数据进行翻译...")
    with torch.no_grad():
        for src, trg in tqdm(iterator):
            src = src.long()
            trg = trg.long()

            for i in range(src.shape[0]):
                src_sentence = src[i]
                trg_sentence = trg[i]

                # 【修正 3】使用传入的 pad_idx 过滤，而不是硬编码 1
                valid_src = src_sentence[src_sentence != pad_idx]

                # 翻译
                pred_indices = translate_sentence(
                    valid_src, model, device,
                    sos_idx=trg_vocab.stoi['<sos>'],
                    eos_idx=trg_vocab.stoi['<eos>']
                )

                # 转换回单词 (过滤 pad, sos, eos)
                target_indices = [idx.item() for idx in trg_sentence if
                                  idx not in [pad_idx, trg_vocab.stoi['<sos>'], trg_vocab.stoi['<eos>']]]
                pred_indices = [idx for idx in pred_indices if idx != trg_vocab.stoi['<eos>']]

                pred_tokens = [trg_vocab.itos[idx] for idx in pred_indices]
                target_tokens = [trg_vocab.itos[idx] for idx in target_indices]

                pred_trgs.append(pred_tokens)
                trgs.append([target_tokens])

    # 1. Calculate BLEU
    print("\n计算 BLEU Scores...")
    try:
        bleu1 = bleu_score(pred_trgs, trgs, max_n=1, weights=[1])
        bleu2 = bleu_score(pred_trgs, trgs, max_n=2, weights=[0.5, 0.5])
        bleu4 = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])

        print(f'BLEU-1: {bleu1 * 100:.2f}')
        print(f'BLEU-2: {bleu2 * 100:.2f}')
        print(f'BLEU-4: {bleu4 * 100:.2f}')
    except ZeroDivisionError:
        print("BLEU 计算错误: 模型可能输出了空句子或全部是UNK")
        bleu4 = 0

    # 2. Calculate BERTScore
    print("计算 BERTScore (使用 hf-mirror)...")
    pred_sentences = [" ".join(p) for p in pred_trgs]
    ref_sentences = [" ".join(t[0]) for t in trgs]

    try:
        # 增加 batch_size 参数防止显存溢出
        P, R, F1 = bert_score_calc(pred_sentences, ref_sentences, lang="en", verbose=True,
                                   model_type="microsoft/deberta-large-mnli", batch_size=32)
        bert_val = F1.mean().item() * 100
        print(f'BERTScore F1: {bert_val:.2f}')
    except Exception as e:
        print(f"BERTScore 计算出错: {e}")
        bert_val = 0

    return bleu4, bert_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help='Experiments name to load model')
    parser.add_argument('--n_layers', type=int, default=3, help='Must match training config')
    parser.add_argument('--use_test', action='store_true', help='Use Test set instead of Validation set')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data & Vocab
    dm = DataManager(batch_size=128)
    train_loader, valid_loader, test_loader = dm.get_loaders()

    # 获取 Pad Index
    PAD_IDX = dm.PAD_IDX

    if args.use_test:
        print("注意：正在使用【测试集】进行评估！")
        eval_iterator = test_loader
    else:
        print("正在使用【验证集】进行评估...")
        eval_iterator = valid_loader

    # 配置必须与 Training 一致
    INPUT_DIM = len(dm.vocab_src)
    OUTPUT_DIM = len(dm.vocab_trg)
    HID_DIM = 256
    ENC_LAYERS = args.n_layers
    DEC_LAYERS = args.n_layers
    ENC_HEADS = 8
    ENC_PF_DIM = 512
    ENC_DROPOUT = 0.1

    # 【修正 4】模型构建必须包含 pad_idx 并且强制 pre_norm=True (因为修复版训练代码就是这样设定的)
    model = Seq2SeqTransformer(
        src_vocab=INPUT_DIM,
        trg_vocab=OUTPUT_DIM,
        d_model=HID_DIM,
        n_head=ENC_HEADS,
        n_layers=ENC_LAYERS,
        d_ff=ENC_PF_DIM,
        max_len=200,
        dropout=ENC_DROPOUT,
        device=device,
        src_pad_idx=PAD_IDX,  # 新增
        trg_pad_idx=PAD_IDX,  # 新增
        pre_norm=True  # 必须为 True，因为 saved model 包含 final_norm 层
    ).to(device)

    # 3. Load Weights
    model_path = f'output/{args.exp_name}/model.pt'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {model_path}")
        print("请检查 exp_name 是否正确，或者训练是否已生成 model.pt")
        exit()

    # 4. Run Evaluation
    # 传入 pad_idx
    calculate_metrics(model, eval_iterator, dm.vocab_src, dm.vocab_trg, device, PAD_IDX)
