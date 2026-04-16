import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== 基本設定 =====
method = "avg_token_logprob_margin"
category = "Goodreads"
seed = 1
lr = "1e-6"

# 要畫的 p
p_list = [0, 25, 50, 75, 100]

# metric 設定
metric_ = "loss"   # chosen / rejected / margins
metric = metric_
# metric = f"rewards/{metric_}"   # e.g. logps/chosen

plt.figure(figsize=(10, 6))

# ===== loop 每個 p =====
for p in p_list:
    checkpoint_dir = f"./centered_percentile_experiments_4096_1000/{category}/models/seed{seed}/{method}/p{p}/checkpoint-96"
    filename = f"{checkpoint_dir}/trainer_state.json"

    if not os.path.exists(filename):
        print(f"❌ Missing: p{p}")
        continue

    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        log_history = data.get("log_history", data)

        eval_data = []

        for entry in log_history:
            
            eval_data.append({
                "step": entry["step"],
                "value": entry[metric]
            })

        if len(eval_data) == 0:
            print(f"⚠️ No eval data for p{p}")
            continue

        df_eval = pd.DataFrame(eval_data)

        # 畫線
        plt.plot(
            df_eval["step"],
            df_eval["value"],
            marker='o',
            label=f"p{p}"
        )

    except Exception as e:
        print(f"❌ Error in p{p}: {e}")

# ===== 圖設定 =====
plt.title(f"{category} using {method} sampling: {metric} across p values")
plt.xlabel("Step")
plt.ylabel(metric)
plt.legend()
plt.grid(True)

# ===== 存圖 =====
save_dir = f"./plot_compare_{method}"
os.makedirs(save_dir, exist_ok=True)

save_path = f"/data2/chuanhsin0110/CHES_DPO/centered_percentile_experiments_4096_1000/{category}/summary_results/seed{seed}/{method}_{metric_}_p_compare.png"
plt.savefig(save_path)
plt.show()

print(f"✅ Saved to {save_path}")