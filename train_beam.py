import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchmetrics.text import BLEUScore

plt.switch_backend('agg')

# 确保从原始的 LSTM 模型文件导入
from models.rnn_model import Encoder, Decoder, Attention, Seq2SeqRNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LabelSmoothingLoss(nn.Module):
    # ... (这部分代码不变，请保留) ...
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


# --- 新增: Beam Search 解码函数 (适配LSTM) ---
def beam_search_decode(model, src_tensor, dm, device, beam_size=3, max_len=50):
    model.eval()
    with torch.no_grad():
        # encoder_outputs, (hidden, cell)
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    # Beam Search 初始化
    # 每个 beam 包含 (累计log概率, 解码序列, (hidden, cell))
    beams = [(0.0, [dm.vocab_trg.stoi['<sos>']], (hidden, cell))]

    for _ in range(max_len):
        new_beams = []
        all_ended = True
        for log_prob_sum, seq, (h, c) in beams:
            if seq[-1] == dm.vocab_trg.stoi['<eos>']:
                new_beams.append((log_prob_sum, seq, (h, c)))
                continue

            all_ended = False  # 只要有一个还没结束，就继续
            last_token = torch.LongTensor([seq[-1]]).to(device)

            with torch.no_grad():
                output, new_h, new_c, _ = model.decoder(last_token, h, c, encoder_outputs)

            log_probs = torch.nn.functional.log_softmax(output, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=1)

            for i in range(beam_size):
                new_seq = seq + [topk_indices[0, i].item()]
                new_log_prob_sum = log_prob_sum + topk_log_probs[0, i].item()
                new_beams.append((new_log_prob_sum, new_seq, (new_h, new_c)))

        if all_ended:
            break

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

    best_log_prob, best_seq, _ = beams[0]
    trg_tokens = [dm.vocab_trg.itos[i] for i in best_seq]
    return trg_tokens[1:]


# --- train 函数保持不变 (只为完整性保留) ---
def train(model, iterator, optimizer, criterion, clip, device, padding_idx):
    # ... (这部分代码不变，请保留您原来的代码) ...
    model.train()
    epoch_loss = 0
    # 我们在Beam Search实验中不关心训练准确率，但保留以防万一
    for src, trg in tqdm(iterator, desc="Training"):
        src, trg = src.to(device).long(), trg.to(device).long()
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
        non_pad_elements = trg_flat.ne(padding_idx).sum()
        epoch_loss += loss.item() / (non_pad_elements.item() if non_pad_elements > 0 else 1)
    return epoch_loss / len(iterator)


# --- 重点修改：重写 evaluate 函数，只为计算 BLEU ---
def evaluate(model, iterator, device, dm):
    model.eval()
    bleu_score_metric = BLEUScore(n_gram=4).to(device)
    all_preds_text = []
    all_trgs_text = []

    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Evaluating with Beam Search"):
            # --- 关键修复：必须先把数据移动到 GPU ---
            src, trg = src.to(device), trg.to(device)
            # -------------------------------------

            # 逐句进行 beam search
            for i in range(src.shape[0]):
                single_src = src[i, :].unsqueeze(0)

                # 这里的 single_src 现在已经在 GPU 上了，不会再报错
                translation_tokens = beam_search_decode(model, single_src, dm, device, beam_size=3)

                trg_sentence = []
                for idx in trg[i, 1:]:
                    if idx.item() == dm.vocab_trg.stoi['<eos>']: break
                    trg_sentence.append(dm.vocab_trg.itos[idx.item()])

                # 清理 <eos>
                pred_text = " ".join(translation_tokens).replace(" <eos>", "")

                all_preds_text.append(pred_text)
                all_trgs_text.append([" ".join(trg_sentence)])

    final_bleu = bleu_score_metric(all_preds_text, all_trgs_text)
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

    # --- 修改文件保存路径 ---
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 数据加载 (使用您修改过的data_utils) ---
    from utils.data_utils import DataManager

    dm = DataManager(batch_size=128)
    train_loader, valid_loader, test_loader = dm.get_loaders()

    # --- 模型配置 (保持 LSTM 配置) ---
    INPUT_DIM = len(dm.vocab_src)
    OUTPUT_DIM = len(dm.vocab_trg)
    ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS = 256, 256, 512, 2
    ENC_DROPOUT, DEC_DROPOUT = 0.5, 0.5

    print(f"初始化模型: Bi-LSTM={args.bidirectional}, Attention={args.attention}")
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, args.bidirectional)
    attn = Attention(HID_DIM) if args.attention else None
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2SeqRNN(enc, dec, device).to(device)


    # ... (权重初始化、优化器、损失函数等部分不变) ...
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

    best_bleu_score = 0.0
    train_losses, bleu_scores = [], []
    epochs_no_improve = 0

    print("开始 Beam Search 实验训练...")
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # 训练过程不变
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, device, dm.PAD_IDX)

        # 评估过程使用新的 evaluate 函数
        _, _, valid_bleu = evaluate(model, valid_loader, device, dm)

        # scheduler 仍然可以基于 train_loss 更新，因为 valid_loss 现在是0
        scheduler.step(train_loss)
        end_time = time.time()

        train_losses.append(train_loss)
        bleu_scores.append(valid_bleu)

        if valid_bleu > best_bleu_score:
            best_bleu_score = valid_bleu
            torch.save(model.state_dict(), os.path.join(output_dir, f'{args.exp_name}_model.pt'))
            print(f"  -> New Best BLEU! Model Saved! (Best Val BLEU: {valid_bleu * 100:.2f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement in BLEU for {epochs_no_improve} epoch(s).")

        print(f'Epoch: {epoch + 1:02} | Time: {end_time - start_time:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. BLEU: {valid_bleu * 100:.2f}')

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # --- 修改绘图逻辑，只绘制 Train Loss 和 Val BLEU ---
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:blue')
    ax1.plot(train_losses, label='Train Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation BLEU', color='tab:green')
    ax2.plot(bleu_scores, label='Validation BLEU', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')

    title_text = (f'Training Curve: {args.exp_name}\n'
                  f'Best Validation BLEU Score: {best_bleu_score * 100:.2f}')
    plt.title(title_text)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{args.exp_name}_curve.png'))
    plt.close()

    print(f"训练结束，结果已保存至 {output_dir}")
