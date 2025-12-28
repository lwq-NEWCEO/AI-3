import matplotlib.pyplot as plt
import json
import os
import numpy as np


def plot_experiment_analysis(exp_name):
    path = f'output/{exp_name}/history.json'
    if not os.path.exists(path):
        print(f"❌ 找不到 {path}，请确保训练脚本里保存了 json")
        return

    with open(path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # 创建画布
    plt.figure(figsize=(15, 5))

    # 1. Loss Curve
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    plt.title(f'Loss Curve ({exp_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Gradient Norm (检查梯度是否消失/爆炸)
    if 'grad_norm' in history:
        plt.subplot(1, 3, 2)
        plt.plot(epochs, history['grad_norm'], color='purple', label='Grad Norm')
        plt.title('Gradient Norm Analysis')
        plt.xlabel('Epoch')
        plt.ylabel('L2 Norm')
        plt.grid(True, alpha=0.3)
        plt.legend()

    # 3. PPL (困惑度)
    plt.subplot(1, 3, 3)
    perplexity = np.exp(history['val_loss'])
    plt.plot(epochs, perplexity, color='green', label='Val PPL')
    plt.title('Perplexity (Lower is Better)')
    plt.xlabel('Epoch')
    plt.ylabel('PPL')
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path = f'output/{exp_name}/analysis_report.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📊 分析图表已保存至 {save_path}")


if __name__ == "__main__":
    # 分析刚才失败的实验
    plot_experiment_analysis('trans_warmup')
