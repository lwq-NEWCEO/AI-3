import matplotlib.pyplot as plt
import json
import os
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec

# ============================
# 🎨 科研配色方案
# ============================
COLORS = {
    'baseline': '#410e73',  # 深紫色 (Post-Norm)
    'improved': '#df4a68',  # 粉红色 (Pre-Norm)
    'grid': '#eaeaea',
    'text': '#333333'
}

# 设置绘图风格
sns.set_theme(style="white", rc={"axes.grid": True, "grid.color": COLORS['grid'], "grid.linestyle": '--'})
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 适配中文
plt.rcParams['axes.unicode_minus'] = False


def load_data(exp_name):
    """读取 History 和 Metrics"""
    hist_path = f'output/{exp_name}/history.json'
    metr_path = f'output/{exp_name}/metrics.json'

    # 默认空数据结构
    data = {
        'loss': [],
        'metrics': {'bleu1': 0, 'bleu2': 0, 'bleu4': 0, 'bert_score': 0}
    }

    if os.path.exists(hist_path):
        with open(hist_path, 'r') as f:
            h = json.load(f)
            data['loss'] = h['val_loss']

    if os.path.exists(metr_path):
        with open(metr_path, 'r') as f:
            data['metrics'] = json.load(f)
    else:
        print(f"⚠️  警告: 未找到 {exp_name} 的 metrics.json，图表中对应数据将为 0。")

    return data


def plot_dashboard(base_name, impr_name):
    # 1. 读取数据
    base_data = load_data(base_name)
    impr_data = load_data(impr_name)

    # 2. 创建画布布局 (左边是大图Loss，右边是Metrics对比)
    # figsize=(宽, 高)
    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])  # 调整比例

    # === 左图：Validation Loss Curve ===
    ax_loss = plt.subplot(gs[0])
    epochs = range(1, len(base_data['loss']) + 1)

    # 绘制曲线
    ax_loss.plot(epochs, base_data['loss'], color=COLORS['baseline'],
                 linestyle='--', linewidth=2, marker='o', markersize=3, label='Baseline (Post-Norm)')
    ax_loss.plot(epochs, impr_data['loss'], color=COLORS['improved'],
                 linestyle='-', linewidth=2.5, label='Improved (Pre-Norm)')

    # 装饰左图
    ax_loss.set_title('Training Dynamics: Loss Convergence\n(训练动态：损失收敛曲线)', fontsize=14, fontweight='bold',
                      color=COLORS['text'], pad=15)
    ax_loss.set_xlabel('Epochs', fontsize=12)
    ax_loss.set_ylabel('Validation Loss (Cross Entropy)', fontsize=12)
    ax_loss.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)

    # 标注 Loss 优化点
    final_base = base_data['loss'][-1] if base_data['loss'] else 0
    final_impr = impr_data['loss'][-1] if impr_data['loss'] else 0
    if final_base > 0:
        drop_pct = (final_base - final_impr) / final_base * 100
        ax_loss.annotate(f'Lower is better\nLoss ↓ {drop_pct:.1f}%',
                         xy=(len(epochs), final_impr), xytext=(len(epochs) - 10, final_impr + 1.0),
                         arrowprops=dict(facecolor=COLORS['improved'], shrink=0.05),
                         fontsize=10, color=COLORS['improved'], fontweight='bold')

    # === 右图：Comprehensive Metrics Comparison ===
    ax_bar = plt.subplot(gs[1])

    # 定义要展示的指标顺序
    metric_keys = ['bleu1', 'bleu2', 'bleu4', 'bert_score']
    metric_labels = ['BLEU-1\n(词汇)', 'BLEU-2\n(流畅度)', 'BLEU-4\n(翻译质量)', 'BERTScore\n(语义)']

    base_scores = [base_data['metrics'].get(k, 0) for k in metric_keys]
    impr_scores = [impr_data['metrics'].get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    # 绘制柱状图
    rects1 = ax_bar.bar(x - width / 2, base_scores, width, label='Baseline', color=COLORS['baseline'], alpha=0.85)
    rects2 = ax_bar.bar(x + width / 2, impr_scores, width, label='Improved', color=COLORS['improved'], alpha=0.95)

    # 装饰右图
    ax_bar.set_title('Evaluation Metrics Comparison\n(全方位性能评估对比)', fontsize=14, fontweight='bold',
                     color=COLORS['text'], pad=15)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontsize=11)
    ax_bar.set_ylim(0, 105)  # 稍微留点空间给数字
    ax_bar.set_ylabel('Score (0-100)', fontsize=12)
    ax_bar.legend(fontsize=11)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.grid(axis='x')  # 隐藏X轴网格，保留Y轴

    # 自动标数值函数
    def autolabel(rects, color):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax_bar.annotate(f'{height:.1f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

    autolabel(rects1, COLORS['baseline'])
    autolabel(rects2, COLORS['improved'])

    plt.tight_layout()
    save_path = 'output/pic/scientific_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 科研级对比图已保存至: {save_path}")


if __name__ == "__main__":
    # 请确保这里的名字和你训练、评估时的 exp_name 一致
    plot_dashboard('exp_baseline_post', 'exp_improved_pre')
