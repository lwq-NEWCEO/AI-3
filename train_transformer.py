import sys
import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import numpy as np
from tqdm import tqdm

# 绘图后端设置
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 🔧 路径修复
# ==========================================
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_utils import DataManager
from models.transformer_model import Seq2SeqTransformer


# ============================
# 🏃 训练循环
# ============================
def train(model, iterator, optimizer, criterion, clip, device, scheduler=None):
    model.train()
    epoch_loss = 0

    for src, trg in tqdm(iterator, desc="Train"):
        src, trg = src.to(device).long(), trg.to(device).long()

        # trg input: <sos>, x, y, z
        # trg label: x, y, z, <eos>
        trg_input = trg[:, :-1]
        trg_label = trg[:, 1:]

        optimizer.zero_grad()

        # 注意：这里假设你的 model 内部自动处理了 mask 生成
        # 如果你的 model 需要外部传入 mask，这里需要修改
        output, _ = model(src, trg_input)

        # output: [batch size, trg len - 1, output dim]
        output_dim = output.shape[-1]

        output = output.reshape(-1, output_dim)
        trg_label = trg_label.reshape(-1)

        loss = criterion(output, trg_label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # 更新学习率
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device, vocab_trg=None):
    model.eval()
    epoch_loss = 0

    # 用于打印示例翻译
    example_printed = False

    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device).long(), trg.to(device).long()
            trg_input = trg[:, :-1]
            trg_label = trg[:, 1:]

            output, attention = model(src, trg_input)

            # --- 简单的Debug打印: 看看模型到底预测了什么 ---
            if not example_printed and vocab_trg:
                try:
                    # 取第一个样本的预测结果 (Greedy)
                    pred_token = output.argmax(2)[0]  # [seq_len]
                    target_token = trg_label[0]  # [seq_len]

                    # 简单的 token 转 string (假设 vocab 有 itos 或 lookup)
                    # 即使没有 itos，打印 ID 也能看出是否全是同一个数字
                    # print(f"DEBUG Sample Pred IDs: {pred_token.cpu().numpy()[:10]}")
                    example_printed = True
                except:
                    pass
            # -----------------------------------------------

            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg_label = trg_label.reshape(-1)

            loss = criterion(output, trg_label)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# ============================
# 🚀 主程序
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--n_head', type=int, default=8, help="Number of attention heads")

    # 新增优化选项
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay for AdamW")
    parser.add_argument('--use_adamw', action='store_true', help="Use AdamW optimizer instead of Adam")

    parser.add_argument('--pre_norm', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dm = DataManager(batch_size=128)
    train_loader, valid_loader, test_loader = dm.get_loaders()

    # 2. Config
    # 适当增加 Hidden Dim，256 对于 Transformer 稍小，512 是标准
    INPUT_DIM = len(dm.vocab_src)
    OUTPUT_DIM = len(dm.vocab_trg)
    HID_DIM = 256
    ENC_LAYERS = args.n_layers
    DEC_LAYERS = args.n_layers
    ENC_HEADS = 8
    ENC_PF_DIM = 512
    ENC_DROPOUT = 0.1

    # 3. 准备 Pad Index (从 DataManager 获取)
    SRC_PAD_IDX = dm.PAD_IDX
    TRG_PAD_IDX = dm.PAD_IDX

    # 构建模型
    # 【修正】直接实例化，不再需要 try-except，因为 transformer_model.py 已经更新
    model = Seq2SeqTransformer(
        src_vocab=len(dm.vocab_src), trg_vocab=len(dm.vocab_trg),
        d_model=256,
        n_head=args.n_head,  # <--- 动态参数
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=200,
        dropout=args.dropout,
        device=device, src_pad_idx=dm.PAD_IDX, trg_pad_idx=dm.PAD_IDX,
        pre_norm=args.pre_norm
    ).to(device)

    print("Model initialized with Pre-Norm configuration.")


    # 权重初始化 (Xavier)
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)


    model.apply(initialize_weights)

    print(f'Model Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # 【关键修正 1】Optimizer 设置
    # Transformer 需要特定的 beta 参数
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)

    # 【关键修正 2】Label Smoothing Loss
    # 这能防止模型过度自信地预测 <pad> 或 <eos>
    criterion = nn.CrossEntropyLoss(ignore_index=dm.PAD_IDX, label_smoothing=0.1)

    # 【关键修正 3】Scheduler (Warmup)
    # 使用 OneCycleLR 进行预热，这是修复不收敛神器的关键
    EPOCHS = 30
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0005,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS
    )

    save_dir = f'output/{args.exp_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print(f"🚀 开始训练: {args.exp_name} (OneCycleLR + LabelSmoothing)")

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, 1, device, scheduler)
        valid_loss = evaluate(model, valid_loader, criterion, device, dm.vocab_trg)

        end_time = time.time()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(valid_loss)

        with open(f'{save_dir}/history.json', 'w') as f:
            json.dump(history, f)

        # 只要 Loss 下降就保存，防止过拟合
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{save_dir}/model.pt')

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        print(f'Epoch: {epoch + 1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        print(f'\tBest Val Loss: {best_valid_loss:.3f}')

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(valid_losses, label='Val Loss')
        plt.title(f'Loss Curve: {args.exp_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/loss.png')
        plt.close()

    print("Training Complete.")
