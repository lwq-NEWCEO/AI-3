import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchmetrics.text import BLEUScore  # 新增：导入BLEU评估工具

plt.switch_backend('agg')

from utils.data_utils import DataManager
from models.gru import Encoder, Decoder, Attention, Seq2SeqRNN


def beam_search_decode(model, src_tensor, dm, device, beam_size=3, max_len=50):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    # Beam Search 初始化
    # 每个 beam 包含 (累计log概率, 解码序列, 解码器隐状态)
    beams = [(0.0, [dm.vocab_trg.stoi['<sos>']], hidden)]

    for _ in range(max_len):
        new_beams = []
        for log_prob_sum, seq, hidden_state in beams:
            # 如果序列已结束，直接加入新 beams 列表，留待下一轮
            if seq[-1] == dm.vocab_trg.stoi['<eos>']:
                new_beams.append((log_prob_sum, seq, hidden_state))
                continue

            last_token = torch.LongTensor([seq[-1]]).to(device)

            with torch.no_grad():
                output, new_hidden, _ = model.decoder(last_token, hidden_state, encoder_outputs)

            log_probs = torch.nn.functional.log_softmax(output, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=1)

            for i in range(beam_size):
                new_seq = seq + [topk_indices[0, i].item()]
                new_log_prob_sum = log_prob_sum + topk_log_probs[0, i].item()
                new_beams.append((new_log_prob_sum, new_seq, new_hidden))

        # 对所有候选进行排序，选择最好的 beam_size 个
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

    # 选择最好的 beam
    best_log_prob, best_seq, _ = beams[0]
    trg_tokens = [dm.vocab_trg.itos[i] for i in best_seq]
    return trg_tokens[1:]  # 忽略 <sos>

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LabelSmoothingLoss(nn.Module):
    # ... (这部分代码不变，为节省空间省略，请保留您原来的代码) ...
    def __init__(self, classes, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
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
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.detach())


def calculate_accuracy(output, target, padding_idx):
    preds = output.argmax(dim=1)
    non_pad_mask = target != padding_idx
    correct = (preds == target)[non_pad_mask].sum()
    total = non_pad_mask.sum()
    return correct.float() / total.float() if total > 0 else 0.0


def train(model, iterator, optimizer, criterion, clip, device, padding_idx):
    # ... (这部分代码不变，为节省空间省略，请保留您原来的代码) ...
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for src, trg in tqdm(iterator, desc="Training"):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output, _ = model(src, trg)
        output_dim = output.shape[-1]

        output_flat = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        output_log_softmax = nn.functional.log_softmax(output_flat, dim=-1)
        loss = criterion(output_log_softmax, trg_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        acc = calculate_accuracy(output_flat, trg_flat, padding_idx)
        epoch_acc += acc.item()

        non_pad_elements = trg_flat.ne(padding_idx).sum()
        epoch_loss += loss.item() / (non_pad_elements.item() if non_pad_elements > 0 else 1)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# --- 重点修改：evaluate 函数 ---
def evaluate(model, iterator, criterion, dm, device):
    """
    使用 Beam Search 进行验证，仅计算 BLEU 分数。
    """
    model.eval()

    # 初始化 BLEU 指标计算器
    bleu_metric = BLEUScore(n_gram=4).to(device)

    all_preds_text = []
    all_trgs_text = []

    with torch.no_grad():
        # --- 修改点开始：直接解包 src 和 trg ---
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            # --- 修改点结束 ---

            # src shape: [batch_size, src_len]

            # 2. 对当前 batch 中的每一句话进行 Beam Search
            for i in range(src.shape[0]):
                single_src = src[i, :].unsqueeze(0)  # [1, src_len]

                # 调用 Beam Search 解码
                translation_tokens = beam_search_decode(model, single_src, dm, device, beam_size=3)

                # 处理真实文本 (Ground Truth)
                trg_sentence = []
                # trg[i, 1:] 跳过 <sos>
                for idx in trg[i, 1:]:
                    if idx.item() == dm.vocab_trg.stoi['<eos>']: break
                    trg_sentence.append(dm.vocab_trg.itos[idx.item()])

                # --- 收集结果 ---
                # 预测结果
                pred_str = " ".join(translation_tokens).replace(' <eos>', '')
                all_preds_text.append(pred_str)

                # 真实结果
                trg_str = " ".join(trg_sentence)
                all_trgs_text.append([trg_str])

    # 3. 计算整个验证集的 BLEU 分数
    final_bleu = bleu_metric(all_preds_text, all_trgs_text)

    # 4. 返回 0, 0 作为 Loss 和 Acc 的占位符
    return 0, 0, final_bleu.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (参数部分不变) ...
    parser.add_argument('--exp_name', type=str, required=True, help='e.g., rnn_base, rnn_bi, rnn_attn')
    parser.add_argument('--bidirectional', action='store_true', help='Use Bi-LSTM Encoder')
    parser.add_argument('--attention', action='store_true', help='Use Attention Mechanism')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = 'output'  # 直接保存到 output 文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dm = DataManager(batch_size=128)
    train_loader, valid_loader, test_loader = dm.get_loaders()

    # --- 超参数 (建议使用 Attention 模型获得更好效果) ---
    INPUT_DIM = len(dm.vocab_src)
    OUTPUT_DIM = len(dm.vocab_trg)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5  # 恢复到0.5，对于更深的模型，正则化更重要
    DEC_DROPOUT = 0.5

    print(f"初始化模型: Bi-LSTM={args.bidirectional}, Attention={args.attention}")
    # ... (模型初始化部分不变) ...
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, args.bidirectional)
    attn = Attention(HID_DIM) if args.attention else None
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2SeqRNN(enc, dec, device).to(device)


    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.constant_(param.data, 0)


    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = LabelSmoothingLoss(classes=OUTPUT_DIM, padding_idx=dm.PAD_IDX, smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

    N_EPOCHS = 100
    CLIP = 1
    EARLY_STOPPING_PATIENCE = 5

    best_valid_loss = float('inf')
    best_bleu_score = 0.0  # 新增：记录最好的BLEU

    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    bleu_scores = []  # 新增：记录每个epoch的BLEU
    epochs_no_improve = 0

    print("开始终极版训练 (引入BLEU)...")
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, CLIP, device, dm.PAD_IDX)
        # --- evaluate 函数现在返回3个值 ---
        valid_loss, valid_acc, valid_bleu = evaluate(model, valid_loader, criterion, dm, device)

        scheduler.step(valid_loss)
        end_time = time.time()

        train_losses.append(train_loss);
        train_accs.append(train_acc)
        valid_losses.append(valid_loss);
        valid_accs.append(valid_acc)
        bleu_scores.append(valid_bleu)  # 记录BLEU

        # --- 现在以 BLEU 分数为主要保存依据 ---
        if valid_bleu > best_bleu_score:
            best_bleu_score = valid_bleu
            torch.save(model.state_dict(), os.path.join(output_dir, f'{args.exp_name}_model.pt'))
            print(f"  -> New Best BLEU! Model Saved! (Best Val BLEU: {valid_bleu * 100:.2f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement in BLEU for {epochs_no_improve} epoch(s).")

        print(f'Epoch: {epoch + 1:02} | Time: {end_time - start_time:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% |  Val. BLEU: {valid_bleu * 100:.2f}')

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- 终极版绘图，三条曲线 ---
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(train_losses, label='Train Loss', color='tab:blue', linestyle='-')
    ax1.plot(valid_losses, label='Validation Loss', color='tab:blue', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Metrics', color='tab:red')
    # ax2.plot(valid_accs, label='Validation Accuracy', color='tab:red', linestyle=':')
    ax2.plot(bleu_scores, label='Validation BLEU', color='tab:green', linestyle='-')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    title_text = (f'Training Curve: {args.exp_name}\n'
                  f'Best Validation BLEU Score: {best_bleu_score * 100:.2f}')
    plt.title(title_text)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{args.exp_name}_curve.png'))
    plt.close()

    print(f"训练结束，结果已保存至 {output_dir}")

