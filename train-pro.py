# -*- coding: utf-8 -*-
"""
Train-Pro: Optimized Seq2Seq with GRU + Attention + LayerNorm + Beam Search
功能：自动运行多组层数实验，对比 Greedy 与 Beam Search 结果，并分析稳定性 (Std)。
评估指标：BLEU-1/2/4, ROUGE-L, BERTScore (DeBERTa-large)
"""
import os

# =========================================================
# 【修复核心】设置 Hugging Face 国内镜像，解决下载超时问题
# =========================================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import math
import json
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 引入 TorchMetrics ---
from torchmetrics.text import BLEUScore, ROUGEScore, BERTScore

# --- 假设 DataManager 已经准备好 ---
from utils.data_utils import DataManager

# 绘图后端设置
plt.switch_backend('agg')


# =========================================================
# 1) 基础工具
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# 2) 模型定义 (LayerNorm + Fuse Adapter)
# =========================================================
class HiddenAdapter(nn.Module):
    def __init__(self, enc_layers: int, dec_layers: int, enc_hid: int, dec_hid: int, strategy: str = "fuse"):
        super().__init__()
        self.enc_layers, self.dec_layers = enc_layers, dec_layers
        self.strategy = strategy
        self.dim_proj = nn.Linear(enc_hid, dec_hid) if enc_hid != dec_hid else None
        self.layer_proj = None
        if strategy == "fuse" and enc_layers > dec_layers:
            self.layer_proj = nn.Linear(enc_layers, dec_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
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
        seq_len = encoder_outputs.shape[1]
        dec_rep = dec_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((dec_rep, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_hid_dim, n_layers, dropout, attention, enc_out_dim):
        super().__init__()
        self.output_dim = output_dim
        self.dec_hid_dim = dec_hid_dim
        self.n_layers = n_layers
        self.attention = attention
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
        attn_map = None
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
        return prediction.squeeze(1), hidden, attn_map


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
        B, T = trg.shape
        V = self.decoder.output_dim
        outputs = torch.zeros(B, T, V, device=self.device)
        enc_out, enc_hidden = self.encoder(src)
        hidden = self.hidden_adapter(enc_hidden)
        input_token = trg[:, 0]
        for t in range(1, T):
            output, hidden, _ = self.decoder(input_token, hidden, enc_out)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs

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
# 3) 训练与评估 (多指标 + 容错)
# =========================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, x, target):
        assert x.size(1) == self.classes
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.classes - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0: true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.detach())


def train_epoch(model, loader, optimizer, criterion, clip, pad_idx):
    model.train()
    total_loss = 0.0
    for src, trg in tqdm(loader, desc="Train", leave=False):
        src, trg = src.to(model.device), trg.to(model.device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(F.log_softmax(output, dim=-1), trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        non_pad = trg.ne(pad_idx).sum().item()
        total_loss += loss.item() / (non_pad if non_pad > 0 else 1)
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, pad_idx, metrics_dict, id_to_token, eos_token,
             use_beam_search=False, beam_width=5, use_bert_score=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_trgs = [], []

    try:
        eos_idx = list(id_to_token).index(eos_token)
        sos_idx = 2
    except:
        eos_idx = 3
        sos_idx = 2

    for src, trg in tqdm(loader, desc=f"Eval(Beam={use_beam_search})", leave=False):
        src, trg = src.to(model.device), trg.to(model.device)

        # Loss
        output = model(src, trg, teacher_forcing_ratio=0.0)
        out_flat = output[:, 1:].reshape(-1, output.shape[-1])
        trg_flat = trg[:, 1:].reshape(-1)
        loss = criterion(F.log_softmax(out_flat, dim=-1), trg_flat)
        non_pad = trg_flat.ne(pad_idx).sum().item()
        total_loss += loss.item() / (non_pad if non_pad > 0 else 1)

        batch_size = src.shape[0]

        # 生成
        if use_beam_search:
            for i in range(batch_size):
                single_src = src[i].unsqueeze(0)
                pred_seq = model.beam_search(single_src, beam_width=beam_width,
                                             sos_idx=sos_idx, eos_idx=eos_idx)
                pred_toks = []
                for idx in pred_seq:
                    if idx == eos_idx: break
                    if idx not in [sos_idx, pad_idx]:
                        pred_toks.append(id_to_token[idx])
                all_preds.append(" ".join(pred_toks))

                trg_toks = []
                for idx in trg[i]:
                    if idx.item() == eos_idx: break
                    if idx.item() not in [sos_idx, pad_idx]:
                        trg_toks.append(id_to_token[idx.item()])
                all_trgs.append(" ".join(trg_toks))
        else:
            pred_indices = output.argmax(dim=-1)
            for i in range(batch_size):
                p_toks = []
                for idx in pred_indices[i, 1:]:
                    if idx.item() == eos_idx: break
                    if idx.item() != pad_idx:
                        p_toks.append(id_to_token[idx.item()])
                all_preds.append(" ".join(p_toks))

                t_toks = []
                for idx in trg[i, 1:]:
                    if idx.item() == eos_idx: break
                    if idx.item() != pad_idx:
                        t_toks.append(id_to_token[idx.item()])
                all_trgs.append(" ".join(t_toks))

    # 指标计算
    all_trgs_nested = [[t] for t in all_trgs]

    results = {}

    # 1. BLEU
    results['bleu1'] = metrics_dict['bleu1'](all_preds, all_trgs_nested).item()
    results['bleu2'] = metrics_dict['bleu2'](all_preds, all_trgs_nested).item()
    results['bleu4'] = metrics_dict['bleu4'](all_preds, all_trgs_nested).item()

    # 2. ROUGE-L
    rouge_out = metrics_dict['rouge'](all_preds, all_trgs_nested)
    results['rougeL'] = rouge_out['rougeL_fmeasure'].item()

    # 3. BERTScore (带防崩溃保护)
    if use_bert_score and 'bert' in metrics_dict:
        try:
            bert_out = metrics_dict['bert'](all_preds, all_trgs)
            results['bert_f1'] = bert_out['f1'].mean().item()
        except Exception as e:
            print(f"\n[Warning] BERTScore calculation failed: {e}")
            print("Trying to continue with score 0.0")
            results['bert_f1'] = 0.0
    else:
        results['bert_f1'] = 0.0

    return total_loss, results


# =========================================================
# 4) 实验运行控制器
# =========================================================
def run_experiment(train_loader, valid_loader, test_loader, vocab_info,
                   enc_layers, dec_layers, seed, run_name):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    HID_DIM = 256
    LR = 5e-4
    EPOCHS = 30
    PATIENCE = 5

    enc = Encoder(vocab_info['src_vocab'], 256, HID_DIM, enc_layers, 0.25, True)
    attn = Attention(HID_DIM * 2, HID_DIM)
    dec = Decoder(vocab_info['trg_vocab'], 256, HID_DIM, dec_layers, 0.25, attn, enc.enc_out_dim)
    model = Seq2SeqRNN(enc, dec, device, hidden_align_strategy="fuse").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = LabelSmoothingLoss(vocab_info['trg_vocab'], vocab_info['pad_idx'], smoothing=0.1)

    print("Initializing metrics...")
    metrics = {
        'bleu1': BLEUScore(n_gram=1).to(device),
        'bleu2': BLEUScore(n_gram=2).to(device),
        'bleu4': BLEUScore(n_gram=4).to(device),
        'rouge': ROUGEScore(rouge_keys=('rougeL',)).to(device),
    }
    # 尝试加载 BERTScore，如果失败不影响训练，但在最终评估时会再次尝试或跳过
    try:
        metrics['bert'] = BERTScore(model_name_or_path="microsoft/deberta-large-mnli").to(device)
        print("BERTScore loaded.")
    except Exception as e:
        print(f"Warning: Failed to load BERTScore during init ({e}).")

    best_bleu4 = 0.0
    no_improve = 0
    save_path = f"model_{run_name}.pt"

    print(f"\n>>> Start Run: {run_name} (Seed={seed}) | Enc={enc_layers} Dec={dec_layers}")

    # 1. 训练
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 1.0, vocab_info['pad_idx'])

        val_loss, res = evaluate(model, valid_loader, criterion, vocab_info['pad_idx'],
                                 metrics, vocab_info['id_to_token'], vocab_info['eos_token'],
                                 use_beam_search=False, use_bert_score=False)

        current_bleu4 = res['bleu4']
        scheduler.step(current_bleu4)

        if current_bleu4 > best_bleu4:
            best_bleu4 = current_bleu4
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        print(f"  Ep {epoch:02d} | Train:{train_loss:.3f} | Val Loss:{val_loss:.3f} | "
              f"B-4:{res['bleu4'] * 100:.2f} (Best:{best_bleu4 * 100:.2f})")

        if no_improve >= PATIENCE:
            print("  Early Stopping.")
            break

    # 2. 最终测试 (Load Best + Beam Search + Full Metrics)
    print("  Loading Best Model for Final Beam Search & Full Evaluation...")
    model.load_state_dict(torch.load(save_path))

    _, final_res = evaluate(model, valid_loader, criterion, vocab_info['pad_idx'],
                            metrics, vocab_info['id_to_token'], vocab_info['eos_token'],
                            use_beam_search=True, beam_width=5, use_bert_score=True)

    print(f"  >>> Final Results (Beam=5): B-4={final_res['bleu4'] * 100:.2f} | BERT={final_res['bert_f1'] * 100:.2f}")

    return final_res


def main():
    print("Loading Data...")
    dm = DataManager(batch_size=64)
    train_loader, valid_loader, test_loader = dm.get_loaders()

    vocab_info = {
        'src_vocab': len(dm.vocab_src),
        'trg_vocab': len(dm.vocab_trg),
        'pad_idx': dm.PAD_IDX,
        'id_to_token': dm.vocab_trg.itos,
        'eos_token': '<eos>'
    }

    layer_pairs = [
        (2, 2),  # 基准
        (2, 3),  # 增强解码
        (3, 3)  # 深层验证
    ]
    seeds = [42, 43]

    summary = []

    for enc_L, dec_L in layer_pairs:
        group_metrics = {k: [] for k in ['bleu1', 'bleu2', 'bleu4', 'rougeL', 'bert_f1']}

        for seed in seeds:
            tag = f"E{enc_L}D{dec_L}_s{seed}"
            res = run_experiment(train_loader, valid_loader, test_loader, vocab_info,
                                 enc_L, dec_L, seed, tag)

            for k in group_metrics:
                group_metrics[k].append(res[k])

        group_stat = {"cfg": f"E{enc_L}-D{dec_L}"}
        for k, v in group_metrics.items():
            group_stat[f"{k}_mean"] = np.mean(v) * 100
            group_stat[f"{k}_std"] = np.std(v) * 100

        summary.append(group_stat)

    print("\n\n=========================================================================================")
    print("FINAL RESULTS SUMMARY (Mean ± Std over seeds)")
    print("Metrics: BLEU-1/2/4, ROUGE-L, BERTScore (deberta-large-mnli)")
    print("=========================================================================================")
    header = f"{'Config':<10} | {'BLEU-1':<14} | {'BLEU-2':<14} | {'BLEU-4':<14} | {'ROUGE-L':<14} | {'BERTScore':<14}"
    print(header)
    print("-" * 100)

    for s in summary:
        row = f"{s['cfg']:<10} | " \
              f"{s['bleu1_mean']:.2f}±{s['bleu1_std']:.2f} | " \
              f"{s['bleu2_mean']:.2f}±{s['bleu2_std']:.2f} | " \
              f"{s['bleu4_mean']:.2f}±{s['bleu4_std']:.2f} | " \
              f"{s['rougeL_mean']:.2f}±{s['rougeL_std']:.2f} | " \
              f"{s['bert_f1_mean']:.2f}±{s['bert_f1_std']:.2f}"
        print(row)
    print("=========================================================================================")

    with open("output/pic/final_pro_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
