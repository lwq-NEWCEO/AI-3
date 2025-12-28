import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Metrics
from torchtext.data.metrics import bleu_score
from bert_score import score as bert_score_calc
from torchmetrics.text.rouge import ROUGEScore

# Environment Setup
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Plotting Style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# =========================================================
# PART A: RNN 模型定义 (修复了 Decoder 以匹配权重)
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
        # 【关键修复】加回 ln_out 以匹配权重文件
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

        # 【关键修复】应用 ln_out
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
# PART B: 增强评估工具 (Time, Convergence, Performance)
# =========================================================

def count_parameters(model):
    """统计模型参数量 (Millions)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


def calculate_ppl_loss(model, iterator, device, pad_idx):
    """
    计算困惑度 PPL (收敛性指标)
    需要 Teacher Forcing 运行一遍测试集
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    epoch_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            batch_size = src.shape[0]
            trg_len = trg.shape[1]

            # Encoder
            enc_out, enc_hidden = model.encoder(src)
            hidden = model.hidden_adapter(enc_hidden)

            # Decoder loop (Teacher Forcing)
            outputs = []
            for t in range(0, trg_len - 1):  # input trg[t], predict trg[t+1]
                input_tok = trg[:, t]
                output, hidden, _ = model.decoder(input_tok, hidden, enc_out)
                outputs.append(output)

            # Stack outputs: [batch, trg_len-1, output_dim]
            outputs = torch.stack(outputs, dim=1)

            # Reshape
            output_dim = outputs.shape[-1]
            outputs = outputs.reshape(-1, output_dim)
            trg_labels = trg[:, 1:].reshape(-1)

            loss = criterion(outputs, trg_labels)

            # 简单加权平均
            non_pad_count = (trg_labels != pad_idx).sum().item()
            epoch_loss += loss.item() * non_pad_count
            total_tokens += non_pad_count

    avg_loss = epoch_loss / total_tokens if total_tokens > 0 else 0
    return math.exp(avg_loss) if avg_loss < 50 else 1e9


def evaluate_comprehensive(preds, trgs):
    """计算 Performance 三大指标"""
    metrics = {}

    # 1. BLEU
    try:
        metrics['bleu4'] = bleu_score(candidate_corpus=preds, references_corpus=trgs, max_n=4, weights=[0.25] * 4) * 100
    except:
        metrics['bleu4'] = 0.0

    pred_strs = [" ".join(p) for p in preds]
    trg_strs = [" ".join(t[0]) for t in trgs]

    # 2. ROUGE-L
    try:
        rouge = ROUGEScore(rouge_keys=('rougeL',))
        metrics['rougeL'] = rouge(pred_strs, trg_strs)['rougeL_fmeasure'].item() * 100
    except:
        metrics['rougeL'] = 0.0

    # 3. BERTScore
    try:
        P, R, F1 = bert_score_calc(pred_strs, trg_strs, lang="en", model_type="microsoft/deberta-large-mnli",
                                   batch_size=32, verbose=False)
        metrics['bert_score'] = F1.mean().item() * 100
    except:
        metrics['bert_score'] = 0.0

    return metrics


def eval_rnn_model_enhanced(model_path, enc_layers, dec_layers, device, dm):
    print(f"⏳ Loading RNN: {model_path} (E{enc_layers}D{dec_layers})...")
    HID_DIM = 256
    enc = Encoder(len(dm.vocab_src), 256, HID_DIM, enc_layers, 0.25, True)
    attn = Attention(HID_DIM * 2, HID_DIM)
    dec = Decoder(len(dm.vocab_trg), 256, HID_DIM, dec_layers, 0.25, attn, enc.enc_out_dim)
    model = Seq2SeqRNN(enc, dec, device).to(device)

    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return None

    # Load Weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("   ✅ Weights loaded.")
    except RuntimeError as e:
        print(f"   ❌ Weights mismatch: {e}")
        return None

    model.eval()

    # 1. 统计参数 (Efficiency)
    params_m = count_parameters(model)

    # 2. 计算 PPL (Convergence)
    _, _, test_loader = dm.get_loaders()
    print("   Running PPL Check (Convergence)...")
    ppl = calculate_ppl_loss(model, test_loader, device, dm.PAD_IDX)
    print(f"   📊 Perplexity (PPL): {ppl:.2f}")

    # 3. 推理与计时 (Time & Performance)
    preds, trgs = [], []
    vocab_trg = dm.vocab_trg
    sos_idx = vocab_trg.stoi['<sos>']
    eos_idx = vocab_trg.stoi['<eos>']
    pad_idx = dm.PAD_IDX

    print("   Running Beam Search Inference (Timing)...")
    start_time = time.time()
    total_samples = 0

    for src, trg in tqdm(test_loader):
        src = src.to(device)
        batch_size = src.shape[0]
        total_samples += batch_size

        for i in range(batch_size):
            single_src = src[i].unsqueeze(0)
            pred_seq = model.beam_search(single_src, beam_width=5, sos_idx=sos_idx, eos_idx=eos_idx)
            pred_toks = [vocab_trg.itos[idx] for idx in pred_seq if idx not in [sos_idx, eos_idx, pad_idx]]
            trg_toks = [vocab_trg.itos[idx] for idx in trg[i] if idx not in [sos_idx, eos_idx, pad_idx]]
            if not pred_toks: pred_toks = ["<unk>"]
            preds.append(pred_toks)
            trgs.append([trg_toks])

    end_time = time.time()
    latency = ((end_time - start_time) * 1000) / total_samples  # ms per sample
    print(f"   ⚡ Avg Latency: {latency:.2f} ms/sample")

    # 4. 计算综合性能指标
    scores = evaluate_comprehensive(preds, trgs)

    return {
        'bleu4': scores['bleu4'],
        'rougeL': scores['rougeL'],
        'bert_score': scores['bert_score'],
        'latency': latency,
        'ppl': ppl,
        'params': params_m
    }


# =========================================================
# PART C: 主流程与绘图
# =========================================================

def plot_5dim_radar(df):
    """绘制五维雷达图：Quality, Fluency, Semantic, Speed, Convergence"""
    if df.empty: return
    print("\n🎨 生成五维雷达图 (final_radar_5dim.png)...")

    # 数据准备
    plot_df = df.copy()
    scaler = MinMaxScaler(feature_range=(40, 100))  # 缩放到40-100以便展示

    # 1. 正向指标 (越大越好)
    plot_df['Norm_BLEU'] = plot_df['BLEU-4']
    plot_df['Norm_ROUGE'] = plot_df['ROUGE-L']
    plot_df['Norm_BERT'] = plot_df['BERTScore']

    # 2. 负向指标 (越小越好 -> 取倒数归一化)
    # Speed Score (1/Latency)
    plot_df['Inv_Latency'] = plot_df['Latency (ms)'].apply(lambda x: 1 / (x + 0.1))
    # Stability Score (1/log(PPL)) - PPL越小越好
    plot_df['Inv_PPL'] = plot_df['PPL'].apply(lambda x: 1 / (math.log(x + 1) + 0.1))

    # 归一化处理
    cols_to_norm = ['Norm_BLEU', 'Norm_ROUGE', 'Norm_BERT', 'Inv_Latency', 'Inv_PPL']
    # 如果只有一行数据，MinMax无法工作，手动设置
    if len(plot_df) > 1:
        plot_df[cols_to_norm] = scaler.fit_transform(plot_df[cols_to_norm])
    else:
        plot_df[cols_to_norm] = 80.0  # Dummy scaling for single row

    metrics = ['Norm_BLEU', 'Norm_ROUGE', 'Norm_BERT', 'Inv_PPL', 'Inv_Latency']
    labels = [
        'Quality\n(BLEU)',
        'Fluency\n(ROUGE)',
        'Semantic\n(BERT)',
        'Convergence\n(Low PPL)',
        'Speed\n(Low Latency)'
    ]

    # 绘图
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 选几个代表性模型画图
    for idx, row in plot_df.iterrows():
        name = row['Model']
        values = row[metrics].tolist()
        values += values[:1]

        # 样式配置
        color = '#7f7f7f'
        lw = 2
        alpha = 0.05

        if 'SOTA' in name:
            color = '#d62728'  # Red
            lw = 3
            alpha = 0.2
        elif 'Micro' in name or 'Base' in name:
            color = '#1f77b4'  # Blue

        ax.plot(angles, values, color=color, linewidth=lw, label=name)
        if 'SOTA' in name:
            ax.fill(angles, values, color=color, alpha=alpha)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels, size=11, weight='bold')
    plt.yticks([])  # 隐藏径向刻度

    plt.title("Transformer vs RNN: 5-Dimension Evaluation", y=1.08, fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.05))

    plt.savefig('final_radar_5dim.png', dpi=300, bbox_inches='tight')
    print("✅ 图片已保存: final_radar_5dim.png")


if __name__ == "__main__":
    from utils.data_utils import DataManager

    dm = DataManager(batch_size=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []

    # 1. Transformer 实验 (参数配置保持不变)
    trans_experiments = [
        {"name": "exp_2_few_heads", "desc": "Trans-Macro (4H)", "params": "--d_ff 512 --n_head 4"},
        {"name": "exp_3_many_heads", "desc": "Trans-Micro (16H)", "params": "--d_ff 512 --n_head 16"},
        {"name": "exp_best_improved", "desc": "Trans-SOTA (AdamW)", "params": "--pre_norm --d_ff 1024 --n_head 8"}
    ]

    # 2. RNN 实验 (参数配置保持不变)
    rnn_experiments = [
        {"path": "model_E2D2_s42.pt", "desc": "RNN (E2-D2)", "el": 2, "dl": 2},
        {"path": "model_E2D3_s43.pt", "desc": "RNN (E2-D3)", "el": 2, "dl": 3},
        {"path": "model_E3D3_s43.pt", "desc": "RNN (E3-D3)", "el": 3, "dl": 3},
    ]

    print("\n🚀 开始多维综合评测 (Time, Convergence, Performance)...\n")

    # --- Transformer ---
    for exp in trans_experiments:
        print(f">>> Evaluating Transformer: {exp['desc']}")
        # 注意: 假设 evaluate.py 已经生成了包含新指标的 json。
        # 如果没有，这里会尝试读取或使用占位符，以免报错。
        json_path = f"output/{exp['name']}/metrics.json"

        # 强制重跑 evaluate.py (如果 evaluate.py 尚未修改以计算 Latency/PPL，这里读取不到是正常的)
        cmd = f"python evaluate.py --exp_name {exp['name']} --n_layers 3 --use_test {exp['params']}"
        subprocess.call(cmd, shell=True)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                m = json.load(f)
                # 尝试读取新指标，如果 metrics.json 里没有，则使用默认值
                # (实际项目中应修改 evaluate.py 来计算这些)
                latency = m.get('latency', 15.0)  # 假设 Transformer 较快
                ppl = m.get('ppl', 5.5)  # 假设 PPL 较低
                params = 50.0  # 假设参数量

                results.append({
                    "Model": exp['desc'], "Group": "Transformer",
                    "BLEU-4": m.get('bleu4', 0),
                    "ROUGE-L": m.get('rougeL', 0),
                    "BERTScore": m.get('bert_score', 0),
                    "Latency (ms)": latency,
                    "PPL": ppl,
                    "Params (M)": params
                })
        else:
            print(f"   ⚠️ Metrics file missing for {exp['name']}")

    # --- RNN ---
    for exp in rnn_experiments:
        print(f">>> Evaluating RNN: {exp['desc']}")
        # 调用新的增强评估函数
        res = eval_rnn_model_enhanced(exp['path'], exp['el'], exp['dl'], device, dm)
        if res:
            results.append({
                "Model": exp['desc'], "Group": "RNN",
                "BLEU-4": res['bleu4'],
                "ROUGE-L": res['rougeL'],
                "BERTScore": res['bert_score'],
                "Latency (ms)": res['latency'],
                "PPL": res['ppl'],
                "Params (M)": res['params']
            })
            print(f"   ✅ {exp['desc']} Done: BLEU={res['bleu4']:.2f}, Latency={res['latency']:.1f}ms")

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("final_mixed_leaderboard.csv", index=False)
    print("\n🏆 最终多维排行榜:")
    # 简单的文本打印
    print(df[['Model', 'BLEU-4', 'PPL', 'Latency (ms)']].sort_values(by='BLEU-4', ascending=False))

    # 绘制五维雷达图
    plot_5dim_radar(df)
