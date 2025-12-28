import matplotlib.pyplot as plt
import json
import os


def load_history(exp_name):
    path = f'output/{exp_name}/history.json'
    if not os.path.exists(path):
        print(f"Warning: No history found for {exp_name}")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def plot_comparison(experiments):
    """
    experiments: dict, key=Label, value=exp_name
    Example: {'Baseline': 'trans_base_post', 'Optimized': 'trans_optimized'}
    """
    plt.figure(figsize=(12, 5))

    # 1. Train Loss Comparison
    plt.subplot(1, 2, 1)
    for label, exp_name in experiments.items():
        hist = load_history(exp_name)
        if hist:
            plt.plot(hist['train_loss'], label=f'{label} (Train)', linestyle='--')

    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Validation Loss Comparison (关键)
    plt.subplot(1, 2, 2)
    for label, exp_name in experiments.items():
        hist = load_history(exp_name)
        if hist:
            # 找到最低点
            min_loss = min(hist['val_loss'])
            min_epoch = hist['val_loss'].index(min_loss)
            p = plt.plot(hist['val_loss'], label=f'{label} (Val)')

            # 标记最低点
            plt.scatter(min_epoch, min_loss, c=p[0].get_color())
            plt.text(min_epoch, min_loss + 0.1, f'{min_loss:.2f}', fontsize=9)

    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_result.png', dpi=300)
    print("对比图已保存为 comparison_result.png")


if __name__ == "__main__":
    # 你需要把之前实验的 history.json 找回来，或者重新跑一遍生成 json
    # 如果之前的实验没存 json，这图画不出来

    # 假设你接下来的实验是这样命名的：
    experiments = {
        'Baseline (Post-Norm)': 'trans_base_post',  # 之前的
        'Optimized (Pre-Norm + Warmup)': 'trans_warmup'  # 接下来要跑的
    }

    plot_comparison(experiments)
