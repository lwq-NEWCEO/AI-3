import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# 设置中文字体
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_final_radar():
    csv_path = "output/pic/final_mixed_leaderboard.csv"
    if not os.path.exists(csv_path):
        print("❌ 数据文件不存在")
        return

    df = pd.read_csv(csv_path)

    # 提取指标
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-4', 'BERTScore']
    labels = ['BLEU-1\n(词汇)', 'BLEU-2\n(流畅度)', 'BLEU-4\n(翻译质量)', 'BERTScore\n(语义)']

    # 准备雷达图数据
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 定义样式映射
    # 格式: (Color, LineStyle, LineWidth, Alpha)
    styles = {
        'Trans-SOTA': ('#d62728', '-', 3.0, 1.0),  # 红色粗实线 (主角)
        'Trans-Macro': ('#1f77b4', '-.', 1.5, 0.7),  # 蓝色点划线
        'Trans-Micro': ('#17becf', '-.', 1.5, 0.7),  # 青色点划线
        'RNN (E2-D2)': ('#7f7f7f', '--', 2.0, 0.8),  # 灰色虚线 (最强RNN)
        'RNN (E2-D3)': ('#df4a68', ':', 1.5, 0.6),
        'RNN (E3-D3)': ('#410e73', ':', 1.5, 0.6)  # 更浅灰
    }

    # 排序：把 SOTA 放到最后画，确保它在最上层
    # 简单的逻辑：先画RNN，再画Trans变体，最后画SOTA
    df['order'] = df['Model'].apply(lambda x: 2 if 'SOTA' in x else (0 if 'RNN' in x else 1))
    df = df.sort_values('order')

    for index, row in df.iterrows():
        name = row['Model']
        values = row[metrics].tolist()
        values += values[:1]

        # 模糊匹配样式
        color, ls, lw, alpha = ('black', '-', 1, 0.5)
        for key, style in styles.items():
            if key in name:
                color, ls, lw, alpha = style
                break

        ax.plot(angles, values, color=color, linewidth=lw, linestyle=ls, label=name, alpha=alpha)

        # 只填充 SOTA
        if 'SOTA' in name:
            ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, size=13, weight='bold')

    # Y轴刻度
    plt.ylim(30, 90)
    plt.yticks([40, 50, 60, 70, 80], ["40", "50", "60", "70", "80"], color="grey", size=10)

    plt.title("Transformer SOTA vs RNN 最佳模型全方位对比", y=1.08, fontsize=18, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=11, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig('final_mixed_radar.png', dpi=300, bbox_inches='tight')
    print("✅ 终极混合对比图已保存: final_mixed_radar.png")


if __name__ == "__main__":
    plot_final_radar()
