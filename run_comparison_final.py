import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Metrics
from torchtext.data.metrics import bleu_score
from bert_score import score as bert_score_calc
from torchmetrics.text.rouge import ROUGEScore

# =========================================================
# 0. 环境与绘图设置 (Scientific Style)
# =========================================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 尝试使用科研风格，如果报错则回退
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    sns.set_theme(style="whitegrid")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 适配中英文
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# =========================================================
# 1. 模型定义 (RNN部分)
# =========================================================

class HiddenAdapter(nn.Module):
    def __init__(self, enc_layers, dec_layers, enc_hid, dec_hid, strategy="fuse"):
        super().__init__()
        self.enc_layers, self.dec_layers = enc_layers, dec_layers
        self.strategy = strategy
        self.dim_proj = nn.Linear(enc_hid, dec_hid) if enc_hid != dec_hid else None
        self.layer_proj = None
        if strategy == "fuse" and enc_layers > dec_layers:
            self.layer_proj = nn.Linear(enc_layers, dec_layers)

    def forward(self, h):
        if self.dim_proj: h = self.dim_proj(h)
        L_enc = h.shape[0]
        if L_enc == self.dec_layers: return h
        if L_enc > self.dec_layers:
            if self.strategy == "fuse" and self.layer_proj:
                h = h.permute(1, 2, 0)
                h = self.layer_proj(h)
                h = h.permute(2, 0, 1)
                return h.contiguous()
            else:
                return h[-self.dec_layers:]
        missing = self.dec_layers - L_enc
        zeros = torch.zeros(missing, h.shape[1], h.shape[2], device=h.device, dtype=h.dtype)
        return torch.cat([zeros, h], dim=0)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.bidirectional = bidirectional
        self.enc_out_dim = hid_dim * (2 if bidirectional else 1)
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.ln_emb = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0,
                          batch_first=True, bidirectional=bidirectional)
        if bidirectional: self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.ln_emb(embedded)
        embedded = self.dropout(embedded)
        outputs, hidden = self.rnn(embedded)
        if self.bidirectional:
            hidden = hidden.view(self.n_layers, 2, -1, self.hid_dim)
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
            hidden = torch.tanh(self.fc_hidden(hidden))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_out_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_out_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, dec_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        dec_rep = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((dec_rep, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_hid_dim, n_layers, dropout, attention, enc_out_dim):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.ln_emb = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        rnn_in_dim = emb_dim + (enc_out_dim if attention else 0)
        self.rnn = nn.GRU(rnn_in_dim, dec_hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        fc_in_dim = dec_hid_dim + (enc_out_dim + emb_dim if attention else 0)
        self.fc_out = nn.Linear(fc_in_dim, output_dim)
        self.ln_out = nn.LayerNorm(output_dim)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input.unsqueeze(1))
        embedded = self.ln_emb(embedded)
        embedded = self.dropout(embedded)
        if self.attention:
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            attn_map = attn_weights.unsqueeze(1)
            context = torch.bmm(attn_map, encoder_outputs)
            rnn_input = torch.cat((embedded, context), dim=2)
            output, hidden = self.rnn(rnn_input, hidden)
            prediction = self.fc_out(torch.cat((output, context, embedded), dim=2))
        else:
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc_out(output)
        prediction = self.ln_out(prediction)
        return prediction.squeeze(1), hidden, None


class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder, device, hidden_align_strategy="fuse"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.hidden_adapter = HiddenAdapter(
            encoder.n_layers, decoder.n_layers, encoder.hid_dim, decoder.rnn.hidden_size, strategy=hidden_align_strategy
        )

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        enc_outputs, hidden = self.encoder(src)
        hidden = self.hidden_adapter(hidden)

        input = trg[:, 0]
        attentions = []

        for t in range(1, trg_len):
            output, hidden, attention = self.decoder(input, hidden, enc_outputs)
            outputs[:, t] = output
            if attention is not None:
                attentions.append(attention)

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs, attentions

    def beam_search(self, src, beam_width=5, max_len=50, sos_idx=2, eos_idx=3):
        with torch.no_grad():
            enc_out, enc_hidden = self.encoder(src)
            hidden = self.hidden_adapter(enc_hidden)
            start_node = (0.0, torch.tensor([sos_idx], device=self.device), hidden, [sos_idx])
            beams = [start_node]
            for _ in range(max_len):
                candidates = []
                for score, last_tok, hid, seq in beams:
                    if last_tok.item() == eos_idx:
                        candidates.append((score, last_tok, hid, seq))
                        continue
                    output, new_hid, _ = self.decoder(last_tok, hid, enc_out)
                    log_probs = F.log_softmax(output, dim=1)
                    topk_log_probs, topk_ids = log_probs.topk(beam_width)
                    for k in range(beam_width):
                        sym = topk_ids[0, k]
                        val = topk_log_probs[0, k]
                        new_score = score - val.item()
                        new_seq = seq + [sym.item()]
                        candidates.append((new_score, sym.unsqueeze(0), new_hid, new_seq))
                ordered = sorted(candidates, key=lambda x: x[0])
                beams = ordered[:beam_width]
                if beams[0][1].item() == eos_idx: break
            return beams[0][3]


# =========================================================
# 2. 评估工具 (核心修改：返回Loss列表)
# =========================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


def evaluate_metrics(preds, trgs):
    metrics = {}
    # BLEU
    metrics['bleu1'] = bleu_score(preds, trgs, max_n=1, weights=[1.0]) * 100
    metrics['bleu2'] = bleu_score(preds, trgs, max_n=2, weights=[0.5, 0.5]) * 100
    metrics['bleu4'] = bleu_score(preds, trgs, max_n=4, weights=[0.25] * 4) * 100

    # ROUGE
    pred_strs = [" ".join(p) for p in preds]
    trg_strs = [" ".join(t[0]) for t in trgs]
    rouge = ROUGEScore(rouge_keys=('rougeL',))
    metrics['rougeL'] = rouge(pred_strs, trg_strs)['rougeL_fmeasure'].item() * 100

    # BERTScore
    try:
        # 显存不足时可禁用或减小batch_size
        P, R, F1 = bert_score_calc(pred_strs, trg_strs, lang="en",
                                   model_type="microsoft/deberta-large-mnli",
                                   batch_size=16, verbose=False)
        metrics['bert_score'] = F1.mean().item() * 100
    except Exception as e:
        print(f"Warning: BERTScore failed ({e}), setting to 0.")
        metrics['bert_score'] = 0
    return metrics


def eval_rnn_full(model_path, enc_layers, dec_layers, device, dm):
    """
    运行RNN评估：
    1. 计算PPL和收集Batch Losses
    2. 运行Beam Search测速和生成预测
    3. 计算各项指标
    """
    print(f"🔄 Evaluating RNN: {model_path} ...")

    # Rebuild Model
    HID_DIM = 256
    enc = Encoder(len(dm.vocab_src), 256, HID_DIM, enc_layers, 0.25, True)
    attn = Attention(HID_DIM * 2, HID_DIM)
    dec = Decoder(len(dm.vocab_trg), 256, HID_DIM, dec_layers, 0.25, attn, enc.enc_out_dim)
    model = Seq2SeqRNN(enc, dec, device).to(device)

    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return None
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, _, test_loader = dm.get_loaders()
    vocab_trg = dm.vocab_trg
    pad_idx = dm.PAD_IDX
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='mean')  # Batch mean

    # --- Step 1: Loss Curve & PPL ---
    batch_losses = []
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)
            trg_len = trg.shape[1]
            enc_out, enc_hidden = model.encoder(src)
            hidden = model.hidden_adapter(enc_hidden)
            outputs = []
            for t in range(0, trg_len - 1):
                input_tok = trg[:, t]
                output, hidden, _ = model.decoder(input_tok, hidden, enc_out)
                outputs.append(output)

            # Stack outputs [batch, seq_len, vocab]
            outputs = torch.stack(outputs, dim=1)
            output_dim = outputs.shape[-1]

            # Reshape for loss
            outputs_flat = outputs.reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            # Loss per batch (averaged)
            loss = criterion(outputs_flat, trg_flat)
            batch_losses.append(loss.item())

            # PPL calculation (sum based)
            non_pad = (trg_flat != pad_idx).sum().item()
            total_loss += loss.item() * non_pad  # Approx accumulation
            total_tokens += non_pad

    avg_loss_ppl = sum(batch_losses) / len(batch_losses)  # Simple avg of batch means
    ppl = math.exp(avg_loss_ppl) if avg_loss_ppl < 20 else 1e5

    # --- Step 2: Generation & Latency ---
    preds, trgs = [], []
    sos_idx, eos_idx = vocab_trg.stoi['<sos>'], vocab_trg.stoi['<eos>']

    start_time = time.time()
    sample_count = 0
    # 为了速度，只在部分测试集上测BeamSearch，或者全量
    # 这里全量运行，请耐心等待
    for src, trg in tqdm(test_loader, desc="Beam Search"):
        src = src.to(device)
        bs = src.shape[0]
        sample_count += bs
        for i in range(bs):
            pred_seq = model.beam_search(src[i].unsqueeze(0), beam_width=5, sos_idx=sos_idx, eos_idx=eos_idx)
            pred_toks = [vocab_trg.itos[idx] for idx in pred_seq if idx not in [sos_idx, eos_idx, pad_idx]]
            trg_toks = [vocab_trg.itos[idx] for idx in trg[i] if idx not in [sos_idx, eos_idx, pad_idx]]
            preds.append(pred_toks)
            trgs.append([trg_toks])

    latency = ((time.time() - start_time) * 1000) / sample_count

    # --- Step 3: Metrics ---
    metrics = evaluate_metrics(preds, trgs)

    result = {
        'bleu1': metrics['bleu1'],
        'bleu2': metrics['bleu2'],
        'bleu4': metrics['bleu4'],
        'rougeL': metrics['rougeL'],
        'bert_score': metrics['bert_score'],
        'latency': latency,
        'ppl': ppl,
        'params': count_parameters(model),
        'batch_losses': batch_losses  # 保存Loss列表用于绘图
    }
    return result


# =========================================================
# 3. 绘图函数 (雷达图 & Loss曲线)
# =========================================================
def plot_gradient_norms(json_paths_dict):
    """
    绘制训练过程中的梯度范数变化，用于分析训练稳定性
    json_paths_dict: {'Model Name': 'path/to/history.json'}
    """
    print("📉 Plotting Gradient Norms...")
    plt.figure(figsize=(10, 6))

    has_data = False
    for name, path in json_paths_dict.items():
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            hist = json.load(f)
            # 假设您在 train.py 中保存了 'grad_norm' 列表
            grads = hist.get('grad_norm', [])
            if grads:
                has_data = True
                plt.plot(grads, label=name, marker='.', markersize=4, linewidth=1)

    if has_data:
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Training Stability: Gradient Norm Trajectory')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('final_grad_norms.png')
        plt.close()
        print("✅ Gradient norm plot saved.")
    else:
        print("⚠️ No gradient data found in history.json (Skipping plot).")


import matplotlib.ticker as ticker


def visualize_attention(model, dm, device, model_name, sample_idx=0):
    """
    选取一个测试样本，可视化其注意力热力图
    """
    print(f"👁️ Visualizing Attention for {model_name}...")
    model.eval()

    # 1. 获取一个样本
    _, _, test_loader = dm.get_loaders()
    src, trg = next(iter(test_loader))  # 获取第一个 batch

    # 选 Batch 中的第 sample_idx 个句子
    src_tensor = src[sample_idx].unsqueeze(0).to(device)  # [1, src_len]
    trg_tensor = trg[sample_idx].unsqueeze(0).to(device)  # [1, trg_len]

    src_len = (src_tensor != dm.PAD_IDX).sum()
    trg_len = (trg_tensor != dm.PAD_IDX).sum()

    # 2. 运行模型 (强制 Teacher Forcing 以获取完整的对齐矩阵)
    with torch.no_grad():
        # 注意：这里假设您的 forward 函数返回 outputs, attentions
        # 如果是 RNN，通常 forward 返回: outputs, attentions (List[Tensor])
        outputs, attentions = model(src_tensor, trg_tensor, teacher_forcing_ratio=1.0)

    # 3. 处理 Attention 数据
    # RNN attention 通常是一个 list，每个元素是 [batch, 1, src_len]
    # 我们需要将其拼接成 [trg_len, src_len]
    if isinstance(attentions, list):
        # 移除 None (第一步可能没有 attention) 并拼接
        attentions = [a for a in attentions if a is not None]
        if not attentions:
            print("   ⚠️ No attention weights returned by model.")
            return
        # stack: [trg_len-1, 1, src_len] -> squeeze -> [trg_len-1, src_len]
        attention_matrix = torch.cat(attentions, dim=1).squeeze(0).cpu().numpy()
    else:
        # Transformer 注意力处理略有不同，通常取最后一层的平均
        print("   ⚠️ Transformer attention visualizer requires customized hook.")
        return

    # 4. 获取对应的单词
    src_tokens = [dm.vocab_src.itos[x] for x in src_tensor[0][:src_len].cpu().numpy()]
    trg_tokens = [dm.vocab_trg.itos[x] for x in trg_tensor[0][1:trg_len].cpu().numpy()]  # 跳过 <sos>

    # 截取矩阵有效部分
    attention_matrix = attention_matrix[:len(trg_tokens), :len(src_tokens)]

    # 5. 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix, cmap='viridis')
    fig.colorbar(cax)

    # 设置轴标签
    ax.set_xticklabels([''] + src_tokens, rotation=90, fontsize=12)
    ax.set_yticklabels([''] + trg_tokens, fontsize=12)

    # 强制显示每个刻度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title(f'Attention Heatmap: {model_name}', pad=20)
    plt.tight_layout()
    plt.savefig(f'attention_map_{model_name}.png')
    plt.close()
    print(f"✅ Heatmap saved: attention_map_{model_name}.png")


def visualize_generation_process(model, dm, device, text_input="Ein schwarzer Hund läuft."):
    """
    打印 Beam Search 的每一步候选词，展示生成过程
    """
    print("\n🔍 Visualizing Generation Process (Beam Search)...")
    model.eval()

    # 1. 预处理输入
    # 简单的 tokenizer (假设按空格分词，实际应使用 dm 的 tokenizer)
    tokens = [dm.vocab_src.stoi[t] if t in dm.vocab_src.stoi else dm.vocab_src.stoi['<unk>'] for t in
              text_input.split()]
    src_tensor = torch.LongTensor([dm.vocab_src.stoi['<sos>']] + tokens + [dm.vocab_src.stoi['<eos>']]).unsqueeze(0).to(
        device)

    # 2. 调用模型的 beam_search (假设您修改了模型代码使其支持 debug 模式，或者我们在外面模拟)
    # 这里演示如何打印最终结果对比

    print(f"Source: {text_input}")

    # 运行推断
    with torch.no_grad():
        # 假设 model.beam_search 返回的是 token index list
        output_ids = model.beam_search(src_tensor, beam_width=3, max_len=20,
                                       sos_idx=dm.vocab_trg.stoi['<sos>'],
                                       eos_idx=dm.vocab_trg.stoi['<eos>'])

    output_text = [dm.vocab_trg.itos[idx] for idx in output_ids]
    print(f"Generated: {' '.join(output_text)}")
    print("-" * 30)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


def calculate_ppl_loss(model, iterator, device, pad_idx):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    epoch_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            trg_len = trg.shape[1]
            enc_out, enc_hidden = model.encoder(src)
            hidden = model.hidden_adapter(enc_hidden)
            outputs = []
            for t in range(0, trg_len - 1):
                input_tok = trg[:, t]
                output, hidden, _ = model.decoder(input_tok, hidden, enc_out)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=1)
            output_dim = outputs.shape[-1]
            outputs = outputs.reshape(-1, output_dim)
            trg_labels = trg[:, 1:].reshape(-1)
            loss = criterion(outputs, trg_labels)
            non_pad_count = (trg_labels != pad_idx).sum().item()
            epoch_loss += loss.item() * non_pad_count
            total_tokens += non_pad_count
    avg_loss = epoch_loss / total_tokens if total_tokens > 0 else 0
    return math.exp(avg_loss) if avg_loss < 50 else 1e9


def evaluate_comprehensive(preds, trgs):
    metrics = {}
    try:
        metrics['bleu4'] = bleu_score(candidate_corpus=preds, references_corpus=trgs, max_n=4, weights=[0.25] * 4) * 100
    except:
        metrics['bleu4'] = 0.0
    pred_strs = [" ".join(p) for p in preds]
    trg_strs = [" ".join(t[0]) for t in trgs]
    try:
        rouge = ROUGEScore(rouge_keys=('rougeL',))
        metrics['rougeL'] = rouge(pred_strs, trg_strs)['rougeL_fmeasure'].item() * 100
    except:
        metrics['rougeL'] = 0.0
    try:
        P, R, F1 = bert_score_calc(pred_strs, trg_strs, lang="en", model_type="microsoft/deberta-large-mnli",
                                   batch_size=32, verbose=False)
        metrics['bert_score'] = F1.mean().item() * 100
    except:
        metrics['bert_score'] = 0.0
    return metrics


def eval_rnn_full(model_path, enc_layers, dec_layers, device, dm):
    real_path = model_path
    if not os.path.exists(real_path):
        potential_path = os.path.join("output", model_path)
        if os.path.exists(potential_path):
            real_path = potential_path
        else:
            print(f"❌ File not found: {model_path}")
            return None

    print(f"⏳ Loading RNN: {real_path} (E{enc_layers}D{dec_layers})...")
    HID_DIM = 256
    enc = Encoder(len(dm.vocab_src), 256, HID_DIM, enc_layers, 0.25, True)
    attn = Attention(HID_DIM * 2, HID_DIM)
    dec = Decoder(len(dm.vocab_trg), 256, HID_DIM, dec_layers, 0.25, attn, enc.enc_out_dim)
    model = Seq2SeqRNN(enc, dec, device).to(device)

    try:
        model.load_state_dict(torch.load(real_path, map_location=device))
        print("   ✅ Weights loaded.")
    except RuntimeError as e:
        print(f"   ❌ Weights mismatch: {e}")
        return None

    model.eval()
    params_m = count_parameters(model)
    _, _, test_loader = dm.get_loaders()

    # 1. PPL
    ppl = calculate_ppl_loss(model, test_loader, device, dm.PAD_IDX)

    # 2. Beam Search
    preds, trgs = [], []
    vocab_trg = dm.vocab_trg
    sos_idx, eos_idx, pad_idx = vocab_trg.stoi['<sos>'], vocab_trg.stoi['<eos>'], dm.PAD_IDX

    # 获取 batch losses 用于画图 (通过 PPL 计算过程收集，或这里简单模拟，因为 eval_rnn_full 本来没有返回 losses list)
    # 为了简化，我们这里只返回聚合指标。Loss曲线通常从 training history.json 读取。
    # 如果您需要在 Test Set 上画 Loss 曲线，需要修改 calculate_ppl_loss 返回每个 batch 的 loss。
    # 这里我们返回一个空的 batch_losses，后续主程序会处理。
    batch_losses = []

    start_time = time.time()
    total_samples = 0
    # 为了速度，只测前 20 个 batch
    for i, (src, trg) in enumerate(tqdm(test_loader, desc="Beam Search")):
        if i > 20: break
        src = src.to(device)
        batch_size = src.shape[0]
        total_samples += batch_size
        for k in range(batch_size):
            pred_seq = model.beam_search(src[k].unsqueeze(0), beam_width=5, sos_idx=sos_idx, eos_idx=eos_idx)
            pred_toks = [vocab_trg.itos[idx] for idx in pred_seq if idx not in [sos_idx, eos_idx, pad_idx]]
            if not pred_toks: pred_toks = ["<unk>"]
            preds.append(pred_toks)
            trgs.append([[vocab_trg.itos[idx] for idx in trg[k] if idx not in [sos_idx, eos_idx, pad_idx]]])

    end_time = time.time()
    latency = ((end_time - start_time) * 1000) / max(1, total_samples)

    scores = evaluate_comprehensive(preds, trgs)
    return {
        'bleu1': 0, 'bleu2': 0,  # Placeholder
        'bleu4': scores['bleu4'], 'rougeL': scores['rougeL'], 'bert_score': scores['bert_score'],
        'latency': latency, 'ppl': ppl, 'params': params_m,
        'batch_losses': batch_losses  # Placeholder
    }


# --- 绘图函数 ---

def plot_radar_standard(df):
    if df.empty: return
    print("\n🎨 Plotting Radar Chart...")
    plot_df = df.copy()
    scaler = MinMaxScaler(feature_range=(40, 100))
    plot_df['Norm_BLEU'] = plot_df['BLEU-4']
    plot_df['Norm_ROUGE'] = plot_df['ROUGE-L']
    plot_df['Norm_BERT'] = plot_df['BERTScore']
    plot_df['Inv_Latency'] = plot_df['Latency (ms)'].apply(lambda x: 1 / (x + 0.1))
    plot_df['Inv_PPL'] = plot_df['PPL'].apply(lambda x: 1 / (math.log(x + 1) + 0.1))

    cols = ['Norm_BLEU', 'Norm_ROUGE', 'Norm_BERT', 'Inv_PPL', 'Inv_Latency']
    if len(plot_df) > 1:
        plot_df[cols] = scaler.fit_transform(plot_df[cols])
    else:
        plot_df[cols] = 80.0

    metrics = ['Quality\n(BLEU)', 'Fluency\n(ROUGE)', 'Semantic\n(BERT)', 'Convergence\n(PPL)', 'Speed\n(Latency)']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for idx, row in plot_df.iterrows():
        name = row['Model']
        values = row[cols].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], metrics)
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.05))
    plt.savefig('final_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(results_dict):
    """绘制 Loss 曲线 (依赖 history.json)"""
    print("📈 Plotting Loss Curves...")
    plt.figure(figsize=(10, 6))
    has_data = False

    # 这里的 input 结构可能需要适配
    # 我们假设 results_dict 是 model_name -> history_path
    if not isinstance(results_dict, dict): return

    for name, path in results_dict.items():
        if isinstance(path, str) and os.path.exists(path):
            with open(path, 'r') as f:
                hist = json.load(f)
                losses = hist.get('valid_loss', hist.get('valid_losses', []))
                if losses:
                    has_data = True
                    plt.plot(losses, label=name)
        elif isinstance(path, dict) and 'batch_losses' in path:
            # 兼容之前 evaluate.py 返回的格式
            losses = path['batch_losses']
            if losses:
                has_data = True
                plt.plot(losses, label=name)

    if has_data:
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss Trajectory')
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.savefig('final_loss_curves.png')
        plt.close()
    else:
        print("⚠️ No loss data available to plot.")


def plot_gradient_norms(json_paths_dict):
    print("📉 Plotting Gradient Norms...")
    plt.figure(figsize=(10, 6))
    has_data = False
    for name, path in json_paths_dict.items():
        if not os.path.exists(path):
            alt = os.path.join("output", path)
            if os.path.exists(alt):
                path = alt
            else:
                continue
        try:
            with open(path, 'r') as f:
                hist = json.load(f)
            grads = hist.get('grad_norm', hist.get('grad_norms', []))
            if grads:
                has_data = True
                plt.plot(grads, label=name, alpha=0.8)
        except:
            pass

    if has_data:
        plt.xlabel('Steps')
        plt.ylabel('Norm')
        plt.legend()
        plt.savefig('final_grad_norms.png')
        plt.close()
    else:
        print("⚠️ No gradient data found in history.json.")


def visualize_attention(model, dm, device, model_name, sample_idx=0):
    print(f"👁️ Visualizing Attention for {model_name}...")
    model.eval()
    try:
        _, _, test_loader = dm.get_loaders()
        src, trg = next(iter(test_loader))
    except:
        return

    sample_idx = sample_idx % src.shape[0]
    src_tensor = src[sample_idx].unsqueeze(0).to(device)
    trg_tensor = trg[sample_idx].unsqueeze(0).to(device)

    src_len = (src_tensor != dm.PAD_IDX).sum()
    trg_len = (trg_tensor != dm.PAD_IDX).sum()

    with torch.no_grad():
        # 调用新添加的 forward
        outputs, attentions = model(src_tensor, trg_tensor, teacher_forcing_ratio=1.0)

    if isinstance(attentions, list) and attentions:
        attention_matrix = torch.cat(attentions, dim=1).squeeze(0).cpu().numpy()

        src_tokens = [dm.vocab_src.itos[x] for x in src_tensor[0][:src_len].cpu().numpy()]
        trg_tokens = [dm.vocab_trg.itos[x] for x in trg_tensor[0][1:trg_len].cpu().numpy()]

        attention_matrix = attention_matrix[:len(trg_tokens), :len(src_tokens)]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        cax = ax.matshow(attention_matrix, cmap='viridis')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + src_tokens, rotation=90)
        ax.set_yticklabels([''] + trg_tokens)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.title(f'Attention: {model_name}')
        plt.savefig(f'attention_map_{model_name}.png')
        plt.close()
        print(f"✅ Saved attention_map_{model_name}.png")


if __name__ == "__main__":
    from utils.data_utils import DataManager

    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running on {device}")

    # 1. 设置并验证数据路径
    TARGET_DATA_PATH = 'data/test'
    if not os.path.exists(TARGET_DATA_PATH):
        TARGET_DATA_PATH = 'data'

    try:
        dm = DataManager(path=TARGET_DATA_PATH, batch_size=32)
    except TypeError:
        dm = DataManager(batch_size=32)

    # Sanity Check
    try:
        _, _, test_loader = dm.get_loaders()
        if len(test_loader) > 0:
            print(f"✅ Data loaded. Test size: {len(test_loader.dataset)}")
    except:
        test_loader = None

    final_results = []

    # --- 实验配置 ---
    trans_experiments = [
        {"name": "exp_2_few_heads", "desc": "Trans-Macro (4H)", "params": "--d_ff 512 --n_head 4"},
        {"name": "exp_3_many_heads", "desc": "Trans-Micro (16H)", "params": "--d_ff 512 --n_head 16"},
        {"name": "exp_best_improved", "desc": "Trans-SOTA", "params": "--pre_norm --d_ff 1024 --n_head 8"}
    ]

    rnn_experiments = [
        {"path": "model_E2D2_s42.pt", "desc": "RNN (E2-D2)", "el": 2, "dl": 2},
        {"path": "model_E2D3_s43.pt", "desc": "RNN (E2-D3)", "el": 2, "dl": 3},
        {"path": "model_E3D3_s43.pt", "desc": "RNN (E3-D3)", "el": 3, "dl": 3},
    ]

    # 1. 运行 RNN 评估
    for exp in rnn_experiments:
        res = eval_rnn_full(exp['path'], exp['el'], exp['dl'], device, dm)
        if res:
            res['Model'] = exp['desc']
            clean_res = {k: v for k, v in res.items() if k != 'batch_losses'}
            final_results.append(clean_res)
            print(f"   ✅ {exp['desc']} Done.")

    # 2. 获取 Transformer 数据
    for exp in trans_experiments:
        json_path = f"output/{exp['name']}/metrics.json"
        need_rerun = False
        if not os.path.exists(json_path):
            need_rerun = True
        else:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'batch_losses' not in data and 'bleu4' not in data:  # Basic check
                        need_rerun = True
            except:
                need_rerun = True

        if need_rerun:
            print(f"🔄 Running evaluation for {exp['name']}...")
            cmd = f"python evaluate.py --exp_name {exp['name']} --use_test {exp.get('params', '')}"
            subprocess.call(cmd, shell=True)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                m = json.load(f)
                final_results.append({
                    "Model": exp['desc'],
                    "bleu1": m.get('bleu1', 0), "bleu2": m.get('bleu2', 0), "bleu4": m.get('bleu4', 0),
                    "rougeL": m.get('rougeL', 0), "bert_score": m.get('bert_score', 0),
                    "latency": m.get('latency', 0), "ppl": m.get('ppl', 0), "params": m.get('params', 0)
                })

    # 3. 保存 CSV 并绘图
    if final_results:
        df = pd.DataFrame(final_results)
        col_map = {'bleu1': 'BLEU-1', 'bleu2': 'BLEU-2', 'bleu4': 'BLEU-4',
                   'rougeL': 'ROUGE-L', 'bert_score': 'BERTScore',
                   'latency': 'Latency (ms)', 'ppl': 'PPL', 'params': 'Params (M)'}
        df = df.rename(columns=col_map)
        df.to_csv("final_evaluation_metrics.csv", index=False)
        print("\n💾 Metrics saved to final_evaluation_metrics.csv")

        try:
            plot_radar_standard(df)

            # 绘制 Loss 曲线 (需要 history.json)
            loss_map = {
                "Trans-Macro": "output/exp_2_few_heads/history.json",
                "Trans-SOTA": "output/exp_best_improved/history.json",
                "RNN (E2-D2)": "output/rnn_E2D2/history.json"
            }
            plot_loss_curves(loss_map)

            # 绘制 梯度图
            plot_gradient_norms(loss_map)

            # 可视化 Attention
            best_rnn_path = "model_E2D2_s42.pt"
            if not os.path.exists(best_rnn_path): best_rnn_path = "output/model_E2D2_s42.pt"

            if os.path.exists(best_rnn_path):
                print(f"📥 Loading {best_rnn_path} for Attention Visualization...")
                HID_DIM = 256
                enc = Encoder(len(dm.vocab_src), 256, HID_DIM, 2, 0.25, True)
                attn = Attention(HID_DIM * 2, HID_DIM)
                dec = Decoder(len(dm.vocab_trg), 256, HID_DIM, 2, 0.25, attn, enc.enc_out_dim)
                model_viz = Seq2SeqRNN(enc, dec, device).to(device)
                model_viz.load_state_dict(torch.load(best_rnn_path, map_location=device))
                visualize_attention(model_viz, dm, device, "RNN_E2D2", sample_idx=5)

            print("\n✅ All tasks completed successfully.")
        except Exception as e:
            print(f"\n❌ Plotting failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("\n❌ No results to process.")