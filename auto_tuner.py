import os
import subprocess
import pandas as pd
import json

# ==========================
# 🧪 5组全方位对比实验配置
# ==========================
base_train_cmd = "python train_ablation.py --epochs 35 --n_layers 3"
base_eval_cmd = "python evaluate.py --n_layers 3 --use_test"

experiments = [
    # 1. Baseline: 标准配置 (参照物)
    {
        "name": "exp_1_baseline",
        "d_ff": 512, "n_head": 8, "dropout": 0.1, "lr": 0.0005,
        "extra_train": "",
        "extra_eval": "",
        "desc": "Baseline (8 Heads)"
    },

    # 2. Few Heads: 4头 (探索粗粒度注意力)
    {
        "name": "exp_2_few_heads",
        "d_ff": 512, "n_head": 4, "dropout": 0.1, "lr": 0.0005,
        "extra_train": "",
        "extra_eval": "",
        "desc": "Macro (4 Heads)"
    },

    # 3. Many Heads: 16头 (探索细粒度注意力)
    {
        "name": "exp_3_many_heads",
        "d_ff": 512, "n_head": 16, "dropout": 0.1, "lr": 0.0005,
        "extra_train": "",
        "extra_eval": "",
        "desc": "Micro (16 Heads)"
    },

    # 4. High Capacity: 2048维 FFN (大容量 + 高Dropout)
    {
        "name": "exp_4_high_cap",
        "d_ff": 2048, "n_head": 8, "dropout": 0.3, "lr": 0.0005,
        "extra_train": "",
        "extra_eval": "",
        "desc": "Fat FFN (2048)"
    },

    # 5. SOTA Config: 现代优化组合 (Pre-Norm + AdamW + WD)
    {
        "name": "exp_5_sota",
        "d_ff": 1024, "n_head": 8, "dropout": 0.2, "lr": 0.001,
        "extra_train": "--pre_norm --use_adamw --weight_decay 0.0001",
        "extra_eval": "--pre_norm",
        "desc": "SOTA (AdamW+PreNorm)"
    }
]

results = []

print(f"🚀 开始 5 组全方位对比实验...")

for exp in experiments:
    exp_name = exp['name']
    print(f"\n==========================================")
    print(f"🧪 Experiment: {exp['desc']} ({exp_name})")
    print(f"==========================================")

    # --- 训练 ---
    if os.path.exists(f"output/{exp_name}/model.pt"):
        print("✅ 模型已存在，跳过训练...")
    else:
        cmd = f"{base_train_cmd} --exp_name {exp_name} --d_ff {exp['d_ff']} --n_head {exp['n_head']} --dropout {exp['dropout']} --lr {exp['lr']} {exp['extra_train']}"
        print(f"Command: {cmd}")
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            print(f"❌ 训练崩溃: {exp_name}")
            continue

    # --- 评估 ---
    if os.path.exists(f"output/{exp_name}/metrics.json"):
        print("✅ 评估结果已存在，读取中...")
    else:
        cmd = f"{base_eval_cmd} --exp_name {exp_name} --d_ff {exp['d_ff']} --n_head {exp['n_head']} {exp['extra_eval']}"
        print(f"Evaluating: {cmd}")
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            print(f"❌ 评估崩溃: {exp_name}")
            continue

    # --- 读取数据 ---
    try:
        with open(f"output/{exp_name}/metrics.json", 'r') as f:
            metrics = json.load(f)
            row = {
                "Experiment": exp['desc'],
                "Heads": exp['n_head'],
                "FFN": exp['d_ff'],
                "Optim": "AdamW" if "AdamW" in exp['extra_train'] else "Adam",
                "BLEU-4": metrics['bleu4'],
                "BERTScore": metrics['bert_score']
            }
            results.append(row)
            print(f"📈 {exp['desc']} -> BLEU-4: {metrics['bleu4']:.2f}")
    except:
        print("❌ 读取结果失败")

# ==========================
# 📊 输出排行榜
# ==========================
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by="BLEU-4", ascending=False)
    print("\n🏆 最终实验排行榜:")
    print(df.to_string(index=False))
    df.to_csv("final_5_experiments.csv", index=False)
