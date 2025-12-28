import os
import subprocess
import pandas as pd
import json
import time

# ==========================
# 🧪 实验配置网格
# ==========================
# 我们基于 "Improved" (Pre-Norm) 的配置进行微调
base_cmd = "python train_ablation.py --epochs 35 --n_layers 3 --pre_norm --use_adamw --lr 0.001 --weight_decay 0.0001"
eval_cmd_template = "python evaluate.py --n_layers 3 --use_test --pre_norm"

# 定义要搜索的参数组合
experiments = [
    # 实验组 1: 探索更大的前馈网络 (d_ff)
    {"name": "search_dff_1024", "d_ff": 1024, "dropout": 0.2, "d_model": 256, "heads": 8},
    {"name": "search_dff_2048", "d_ff": 2048, "dropout": 0.3, "d_model": 256, "heads": 8},  # 更大的模型需要更大的dropout

    # 实验组 2: 探索头数 (Multi-Head Impact) - 参考 PPT
    # d_model=256, heads=4 -> d_k=64 (关注更宏观的信息)
    {"name": "search_head_4", "d_ff": 512, "dropout": 0.2, "d_model": 256, "heads": 4},
    # d_model=256, heads=16 -> d_k=16 (关注更微观的细节)
    {"name": "search_head_16", "d_ff": 512, "dropout": 0.2, "d_model": 256, "heads": 16},

    # 实验组 3: 激进的正则化
    {"name": "search_drop_03", "d_ff": 1024, "dropout": 0.3, "d_model": 256, "heads": 8},
]

results = []

print(f"🚀 开始自动化网格搜索，共 {len(experiments)} 组实验...")

for exp in experiments:
    exp_name = exp['name']
    print(f"\n==========================================")
    print(f"🧪 Running Experiment: {exp_name}")
    print(f"Params: {exp}")
    print(f"==========================================")

    # 1. 拼接训练命令
    train_cmd = f"{base_cmd} --exp_name {exp_name} --d_ff {exp['d_ff']} --dropout {exp['dropout']} --n_head {exp['heads']}"

    # 2. 运行训练
    try:
        print(f"Running: {train_cmd}")
        subprocess.check_call(train_cmd, shell=True)
    except subprocess.CalledProcessError:
        print(f"❌ Training failed for {exp_name}")
        continue

    # 3. 拼接评估命令 (注意评估时也要传入 d_ff 和 n_head 以便正确加载模型)
    # 注意：你需要先修改 evaluate.py 和 train_ablation.py 支持 --n_head 参数（见下文）
    current_eval_cmd = f"{eval_cmd_template} --exp_name {exp_name} --d_ff {exp['d_ff']} --n_head {exp['heads']}"

    # 4. 运行评估
    try:
        print(f"Running Evaluation...")
        subprocess.check_call(current_eval_cmd, shell=True)
    except subprocess.CalledProcessError:
        print(f"❌ Evaluation failed for {exp_name}")
        continue

    # 5. 读取结果
    metrics_path = f"output/{exp_name}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            # 合并结果
            row = {**exp, **metrics}
            results.append(row)
            print(f"✅ Result: BLEU-4 = {metrics['bleu4']:.2f}")

# ==========================
# 📊 保存汇总报告
# ==========================
df = pd.DataFrame(results)
# 按 BLEU-4 降序排列
df = df.sort_values(by='bleu4', ascending=False)
print("\n🏆 网格搜索结果汇总:")
print(df[['name', 'd_ff', 'heads', 'dropout', 'bleu4', 'bert_score']])

df.to_csv("grid_search_results.csv", index=False)
print("\n详细结果已保存至 grid_search_results.csv")
