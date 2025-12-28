import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. 数据准备 (更新为你最新的优化实验数据)
# ==========================================
data = [
    {"cfg": "E2-D2", "mean": 33.25,"std": 0.19},
    {"cfg": "E2-D3", "mean": 33.12,"std": 0.13},
    {"cfg": "E3-D3", "mean": 32.40,"std": 0.16}, # 基准
    {"cfg": "E3-D4", "mean": 30.95,"std": 0.21}
]

df = pd.DataFrame(data)

# ==========================================
# 2. 绘图风格设置 (模拟 LaTeX 科研论文风)
# ==========================================
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    plt.style.use('ggplot')

# 强制使用 Times New Roman 字体 (最符合 LaTeX/PDF 输出效果)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体风格


# ==========================================
# 3. 核心绘图逻辑
# ==========================================
def plot_scientific_chart(df, baseline_cfg="E3-D3"):
    # 创建画布，DPI=300 适合论文印刷
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 1. 提取基准线数值
    baseline_row = df[df['cfg'] == baseline_cfg]
    if not baseline_row.empty:
        baseline_val = baseline_row['mean'].values[0]
    else:
        baseline_val = 0
        print(f"Warning: Baseline {baseline_cfg} not found.")

    # 2. 颜色映射：数值越高颜色越深 (Viridis)
    # 即使 Std=0，深浅不一的颜色也有助于视觉区分
    norm = plt.Normalize(df['mean'].min() * 0.95, df['mean'].max())
    colors = plt.cm.viridis(norm(df['mean']))

    # 3. 绘制柱状图
    # 注意：因为 std=0，yerr 虽然传入但不会画出线
    bars = ax.bar(
        df['cfg'],
        df['mean'],
        yerr=df['std'],
        capsize=5,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.9,
        width=0.55,
        error_kw={'elinewidth': 1.5, 'ecolor': 'black'}
    )

    # 4. 绘制基准线 (Baseline Line)
    # 使用红色虚线标记 E3-D3 的位置
    ax.axhline(y=baseline_val, color='#D62728', linestyle='--', linewidth=1.5, zorder=0)

    # 基准线标注文字
    ax.text(
        len(df) - 0.6, baseline_val + 0.3,
        f'Baseline ({baseline_cfg}): {baseline_val:.2f}',
        color='#D62728',
        fontweight='bold',
        fontsize=10,
        ha='right',
        va='bottom',
        family='serif'
    )

    # 5. 数值标注 (Bar Labels)
    for bar, mean_val, std_val in zip(bars, df['mean'], df['std']):
        height = bar.get_height()

        # 标注文本
        label_text = f'{mean_val:.2f}'

        # 智能位置计算
        text_y = height + std_val + 0.2 if std_val > 0 else height + 0.2

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            label_text,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='black',
            family='serif'
        )

    # 6. 图表细节修饰 (符合学术规范)
    ax.set_xlabel('Layer Configuration (Encoder-Decoder)', fontsize=12, fontweight='bold', labelpad=8, family='serif')
    ax.set_ylabel('Best Validation BLEU (%)', fontsize=12, fontweight='bold', labelpad=8, family='serif')

    # 标题可选，有些论文直接在图注(Caption)里写，这里保留作为演示
    ax.set_title('Impact of Network Depth on Translation Quality (Optimized)', fontsize=13, fontweight='bold', pad=15,
                 family='serif')

    # Y轴范围设置：稍微留出顶部空间
    max_y = (df['mean'] + df['std']).max()
    ax.set_ylim(28, max_y * 1.1)  # 技巧：既然分都很高，Y轴起点可以不从0开始(如从28开始)以放大差异，或者保持0
    # ax.set_ylim(0, max_y * 1.2) # 如果想从0开始展示整体感，用这行

    # 网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.xaxis.grid(False)

    # 去掉上方和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    plt.tight_layout()

    # 保存高分辨率图片
    save_path = 'output/pic/optimized_layer_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Chart saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_scientific_chart(df, baseline_cfg="E3-D3")
