import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import Counter

# ============================
# 1. 入口 (Entry)：設定參數與讀取資料
# ============================
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir", type=str, required=True, help="輸入：模型預測結果的 JSON 檔案路徑")
parse.add_argument("--output_dir", type=str, default="./gini_result.json", help="輸出：單次評估結果 JSON 路徑")
parse.add_argument("--topk", type=int, default=5, help="參數：Top K 值")
parse.add_argument("--category", type=str, default="CDs_and_Vinyl", help="參數：資料類別 (用於尋找 id2name)")
parse.add_argument("--gamma", type=float, default=0, help="參數：佔位符 (sh檔有傳入但此處不需使用)")
parse.add_argument("--exp_csv", type=str, default=None, help="輸出：匯總結果的 CSV 路徑 (選填)")
parse.add_argument("--model", type=str, default="Unknown", help="模型名稱 (若 sh 沒傳入，程式碼嘗試從路徑解析)")
parse.add_argument("--id2name_path", type=str, default="./eval/CDs_and_Vinyl/id2name.json", help="id2name.json 的路徑")
args = parse.parse_args()

def read_json(json_file: str):
    """ 讀取 JSON 的輔助函式 """
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} not found.")
        return None
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- 嘗試從路徑解析 Model Name (如果 args.model 是預設值) ---
if args.model == "Unknown" and "/" in args.input_dir:
    # 假設路徑結構為 .../ModelName/test_result.json
    try:
        args.model = args.input_dir.split("/")[-2]
    except:
        pass

# --- 讀取輔助映射表 (關鍵步驟) ---
# Gini Index 需要知道「總物品數(N)」，所以必須讀取 id2name
id2name_path = args.id2name_path
print(f"Loading auxiliary data from: {id2name_path}")
id2name = read_json(id2name_path)

if id2name is None:
    raise FileNotFoundError(f"Critical: id2name.json not found at {id2name_path}. Cannot calculate Gini Index correctly.")

num_total_items = len(id2name)
print(f"Total items in dataset (N): {num_total_items}")

# --- 讀取主要測試數據 ---
print(f"Reading input predictions from: {args.input_dir}")
predictions = read_json(args.input_dir)

if predictions is None:
    raise FileNotFoundError("Prediction file not found.")

# ============================
# 2. 處理核心 (Process)：計算 Gini Index
# ============================
print(f"Calculating Gini Index @ {args.topk}...")

# 1. 截斷預測列表 (Slice Top-K)
# 確保我們只計算前 K 個推薦結果
sliced_preds = [user_list[:args.topk] for user_list in predictions]

# 2. 展平列表 (Flatten)
all_recommended_items = [item for sublist in sliced_preds for item in sublist]

# 3. 統計次數
item_counts = Counter(all_recommended_items)
sorted_count = np.array(sorted(item_counts.values()))


# --- [新增] 檢查點：驗證數據完整性 ---
# 計算 sorted_count 的總和
sum_counts = np.sum(sorted_count)
# 計算 列表的總長度
total_len = len(all_recommended_items)

# 使用 assert 進行強制檢查
# 如果兩者不相等，程式會報錯並停止，防止算出錯誤的 Gini Index
assert sum_counts == total_len, \
    f"數據異常！統計總和 ({sum_counts}) 與 推薦列表總長度 ({total_len}) 不符。"

print(f"Check Passed: Sum of counts ({sum_counts}) matches total recommendations.")
# -------------------------------------

# 4. Gini 計算邏輯
num_recommended_unique = sorted_count.shape[0]  # 有被推薦到的獨特物品數
total_recommendations = len(all_recommended_items) # 總推薦次數 (Users * K)

if total_recommendations == 0:
    gini_index = 0.0
    print("Warning: No recommendations found.")
else:
    # 建立索引：從 (N - 推薦數 + 1) 到 N
    # 對應那些「有被推薦到的物品」在完整排序中的位置
    idx = np.arange(num_total_items - num_recommended_unique + 1, num_total_items + 1)
    
    # 公式： sum((2i - n - 1) * yi) / (n * sum(yi))
    gini_index = np.sum((2 * idx - num_total_items - 1) * sorted_count) / total_recommendations
    gini_index /= num_total_items

print(f"Gini Index result: {gini_index:.6f}")

# 準備寫入的結果字典
eval_dic = {
    "model": args.model,
    "category": args.category,
    "topk": args.topk,
    "GiniIndex": gini_index,
    "num_recommended_unique": num_recommended_unique,
    "total_recommendations": total_recommendations,
    "num_total_items": num_total_items,
    "coverage": num_recommended_unique / num_total_items
}

# ============================
# 3. 出口 (Exit)：輸出結果
# ============================

# --- 出口 A: 寫入詳細 JSON 結果檔 ---
print(f"Writing detailed results to: {args.output_dir}")

data = []
# 讀取舊檔 (若存在) 以便 Append
if os.path.exists(args.output_dir) and os.path.getsize(args.output_dir) > 0:
    with open(args.output_dir, 'r') as file:
        try:
            data = json.load(file)
            if not isinstance(data, list): data = []
        except json.JSONDecodeError:
            data = []

# 避免重複寫入同樣的設定 (Optional check)
# data = [d for d in data if not (d['model'] == args.model and d['topk'] == args.topk)]

data.append(eval_dic)

with open(args.output_dir, 'w') as file:
    json.dump(data, file, separators=(',', ': '), indent=2)


# --- 出口 B: 更新匯總 CSV 報表 (若有指定) ---
def update_csv(dataset_name, model_name, metrics_dict, csv_file):
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Dataset", "Model"])

        # 尋找是否已有該 Dataset + Model 的紀錄
        condition = (df["Dataset"] == dataset_name) & (df["Model"] == model_name)
        
        if not condition.any():
            new_row = {"Dataset": dataset_name, "Model": model_name}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            condition = (df["Dataset"] == dataset_name) & (df["Model"] == model_name)

        # 填入指標數值
        for metric, value in metrics_dict.items():
            if metric not in df.columns:
                df[metric] = np.nan # 初始化新欄位
            df.loc[condition, metric] = value

        df.to_csv(csv_file, index=False)
        print(f"CSV updated: {csv_file}")
    except Exception as e:
        print(f"Error updating CSV: {e}")

if args.exp_csv:
    # 準備 CSV 專用的字典格式
    metric_dic_csv = {
        f"GiniIndex@{args.topk}": gini_index
    }
    update_csv(args.category, args.model, metric_dic_csv, args.exp_csv)

print("Process Completed.")