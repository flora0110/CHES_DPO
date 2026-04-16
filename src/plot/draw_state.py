import json
import pandas as pd
import matplotlib.pyplot as plt
import os
# 設定檔案名稱
method = "Max_ln_ches_scores"
category = "Goodreads"
it = 0
lr = "1e-6"
# filename = f'models/SPRec/{category}_2048_{lr}_MAX_EPOCH{MAX_EPOCH}/it{it}/{method}/checkpoint-320/trainer_state.json'
checkpoint_dir = f'./experiments_new/models/{category}_sftlr_0.00002_lr_{lr}/it{it}/{method}/checkpoint-256'
filename = f'{checkpoint_dir}/trainer_state.json'

# filename = f'models/SPRec/{category}_2048_{lr}/it{it}/{method}/checkpoint-64/trainer_state.json'
# filename = f'experiments/{category_2048_256_ep3/checkpoint-192/trainer_state.json'
# method = "nearest_ln_ches_scores"
metric_ = "accuracies"  # 或 "eval_loss" 根據你的需求選擇
metric = f"rewards/{metric_}"
# metric = f"logps/{metric_}"
# metric = metric_  # 假設 json 中直接使用 "loss" 和 "eval_loss" 作為 key
try:
    # 1. 讀取檔案
    with open(filename, 'r') as f:
        data = json.load(f)

    # 2. 解析數據
    # 檢查 log_history 是否存在
    if 'log_history' in data:
        log_history = data['log_history']
    else:
        # 如果檔案本身就是 list 格式
        log_history = data

    # 分離 Training 和 Evaluation 數據
    train_data = []
    eval_data = []

    for entry in log_history:
        # 提取 Training Loss (通常 key 為 'loss')
        if metric in entry and 'step' in entry:
            # print(entry[metric])
            train_data.append({'step': entry['step'], 'loss': entry[metric]})
        
        # 提取 Evaluation Loss (通常 key 為 'eval_loss')
        if f'eval_{metric}' in entry and 'step' in entry:
            eval_data.append({'step': entry['step'], 'eval_loss': entry[f'eval_{metric}']})

    # 轉換為 DataFrame 方便繪圖
    df_train = pd.DataFrame(train_data)
    df_eval = pd.DataFrame(eval_data)

    # 3. 畫圖
    plt.figure(figsize=(10, 6))

    # 繪製 Training Loss
    if not df_train.empty:
        plt.plot(df_train['step'], df_train['loss'], marker='o', label='Training Loss')
        

    # 繪製 Validation Loss
    if not df_eval.empty:
        plt.plot(df_eval['step'], df_eval['eval_loss'], marker='x', label='Validation Loss')
        for x, y in zip(df_eval['step'], df_eval['eval_loss']):
            plt.text(x, y, f"{y:.3f}", fontsize=8, ha='center', va='bottom')

    plt.title('Training and Validation Loss per Step')
    plt.xlabel('Step')
    plt.ylabel(metric_)
    plt.legend()
    plt.grid(True)
    
    # 顯示圖表
    # plt.savefig(f'plot/{category}_it{it}_{method}_{metric_}_{lr}_MAX_EPOCH{MAX_EPOCH}_curve.png') # 存檔
    # plt.show() 

    # print(f"圖表已繪製並儲存為 plot/{category}_it{it}_{method}_{metric_}_{lr}_MAX_EPOCH{MAX_EPOCH}_curve.png")
    os.makedirs(f'{checkpoint_dir}/plot', exist_ok=True)
    plt.savefig(f'{checkpoint_dir}/plot/{metric_}_curve.png') # 存檔
    plt.show() 

    print(f"圖表已繪製並儲存為 {checkpoint_dir}/plot/{category}_it{it}_{method}_{metric_}_{lr}_curve.png")

except Exception as e:
    print(f"發生錯誤: {e}")
    print("請確認您的 json 檔案格式是否正確。")