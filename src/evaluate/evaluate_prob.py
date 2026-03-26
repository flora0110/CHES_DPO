import os
import re
import math
import json
import argparse
import numpy as np
from tqdm import tqdm

parse = argparse.ArgumentParser()
parse.add_argument("--input_dir", type=str, default="./temp.json", help="result file containing output_log_probability")
parse.add_argument("--category", type=str, default="CDs_and_Vinyl", help="category name")
parse.add_argument("--output_dir", type=str, default="./log_prob_result.json", help="result output path")
# 保留原本的參數結構以免報錯，雖然此腳本可能用不到
parse.add_argument("--model", type=str, default="SPRec", help="model name")
parse.add_argument("--exp_csv", type=str, default=None, help="csv file")
parse.add_argument("--topk", type=str, default="10", help="topk") 
parse.add_argument("--gamma", type=float, default=0.0, help="gamma")
args = parse.parse_args()

def read_json(json_file:str) -> dict:
    with open(json_file, 'r') as f:
        return json.load(f)

# 讀取基礎設定
category = args.category
# 假設路徑結構與您提供的一致
try:
    id2name = read_json(f"./eval/{category}/id2name.json")
    name2id = read_json(f"./eval/{category}/name2id.json")
except FileNotFoundError:
    print(f"Error: Mapping files not found in ./eval/{category}/. Please check the path.")
    exit()

# =========================
# Step 1: 復用原本的 Head/Tail 切分邏輯
# =========================
def build_head_tail_from_train_input(train_json_path: str, name2id: dict, id2name: dict, head_ratio: float = 0.2):
    """
    以 train data 的 input(history) 統計 item popularity
    """
    if not os.path.exists(train_json_path):
        print(f"Warning: Train file {train_json_path} not found. Returning empty sets.")
        return set(), set(), {}

    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    # popularity_count[item_id] = number_of_train_examples_that_contain_it
    popularity_count = {int(i): 0 for i in id2name.keys()}

    for ex in train_data:
        input_text = ex.get("input", "")
        names = re.findall(r'"([^"]+)"', input_text)

        unique_ids_in_example = set()
        for name in names:
            if name in name2id:
                unique_ids_in_example.add(int(name2id[name]))

        for iid in unique_ids_in_example:
            if iid in popularity_count:
                popularity_count[iid] += 1

    # 依 popularity 由高到低排序
    all_item_ids = list(popularity_count.keys())
    all_item_ids.sort(key=lambda x: popularity_count[x], reverse=True)

    # Head = 前 20% item
    n_items = len(all_item_ids)
    n_head = max(1, math.ceil(n_items * head_ratio))
    head_items = set(all_item_ids[:n_head])
    tail_items = set(all_item_ids[n_head:])

    return head_items, tail_items, popularity_count

# 執行切分
train_json_path = f"./data/{category}/train.json"
head_items, tail_items, popularity_count = build_head_tail_from_train_input(
    train_json_path=train_json_path,
    name2id=name2id,
    id2name=id2name,
    head_ratio=0.2
)
print(f"[Head/Tail Split] total_items={len(id2name)} head={len(head_items)} tail={len(tail_items)}")

# =========================
# Step 2: 讀取測試結果並計算 Log Probability 平均
# =========================
result_json = args.input_dir
if not os.path.exists(result_json):
    print(f"Error: Input file {result_json} not found.")
    exit()

f = open(result_json, 'r')
test_data = json.load(f)

# 儲存數值列表
log_probs_head = []
log_probs_tail = []
log_probs_overall = []

missing_prob_count = 0
skipped_target_count = 0

head_sampe_count = 0
tail_sample_count = 0

for i in tqdm(range(len(test_data)), desc="Calculating Log Probability Stats......"):
    data = test_data[i]
    
    # 1. 取得 output_log_probability
    # 這裡假設 key 是 "output_log_probability"，如果是 "log_prob" 請自行修改
    log_prob = data.get('output_log_probability', None)
    
    if log_prob is None:
        missing_prob_count += 1
        continue

    # 2. 取得 Target Item ID 以判斷 Head/Tail
    target_name = data.get('output', "")
    target_name = target_name.strip().strip('"') # 清理字串，與原 code 保持一致

    if target_name in name2id:
        target_id = int(name2id[target_name])
        
        # 加入 Overall
        log_probs_overall.append(log_prob)

        # 加入 Head 或 Tail
        if target_id in head_items:
            log_probs_head.append(log_prob)
        else:
            log_probs_tail.append(log_prob)
    else:
        # Target 不在 ID Mapping 中，跳過不計
        skipped_target_count += 1

# =========================
# Step 3: 計算平均並輸出
# =========================

def safe_mean(l):
    return sum(l) / len(l) if len(l) > 0 else 0.0

avg_head = safe_mean(log_probs_head)
avg_tail = safe_mean(log_probs_tail)
avg_overall = safe_mean(log_probs_overall)

print("\n" + "="*30)
print(f"Category: {category}")
print(f"Total Samples Processed: {len(log_probs_overall)}")
print(f"Skipped (Target not in mapping): {skipped_target_count}")
print(f"Missing Log Prob Field: {missing_prob_count}")
print("-" * 30)
print(f"Average Output Log Probability (Overall): {avg_overall:.4f}")
print(f"Average Output Log Probability (Head)   : {avg_head:.4f} (count: {len(log_probs_head)})")
print(f"Average Output Log Probability (Tail)   : {avg_tail:.4f} (count: {len(log_probs_tail)})")
print("="*30 + "\n")

# =========================
# Step 4: 儲存結果 (JSON)
# =========================
output_res = {
    "category": category,
    "input_file": args.input_dir,
    "stats": {
        "overall": {
            "mean_log_prob": avg_overall,
            "count": len(log_probs_overall)
        },
        "head": {
            "mean_log_prob": avg_head,
            "count": len(log_probs_head)
        },
        "tail": {
            "mean_log_prob": avg_tail,
            "count": len(log_probs_tail)
        }
    }
}

# 確保輸出目錄存在
output_dir_path = os.path.dirname(args.output_dir)
if output_dir_path and not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

with open(args.output_dir, 'w') as f_out:
    json.dump(output_res, f_out, indent=4)

print(f"Results saved to {args.output_dir}")