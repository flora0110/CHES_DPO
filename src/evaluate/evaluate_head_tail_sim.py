import os
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import re
import math
import json
from peft import PeftModel
import argparse
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm # 確保 tqdm 被正確引入

parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="result file")
parse.add_argument("--model",type=str, default="SPRec", help="result file")
parse.add_argument("--exp_csv",type=str, default=None, help="result file")
parse.add_argument("--output_dir",type=str, default="./", help="eval_result")
parse.add_argument("--topk",type=str, default="10", help="topk") # 修改 default 為 "10" 避免錯誤，原為 "./"
parse.add_argument("--gamma",type=float,default=0.0,help="gamma")
parse.add_argument("--category",type=str,default="CDs_and_Vinyl",help="gamma")
parse.add_argument("--train_data_for_head_tail", type=str, default="./data_sprec/CDs_and_Vinyl_train.json", help="用於統計 Head/Tail 的訓練資料路徑")
parse.add_argument("--id2name_path", type=str, default="./eval/CDs_and_Vinyl/id2name.json", help="id2name.json 的路徑")
parse.add_argument("--name2id_path", type=str, default="./eval/CDs_and_Vinyl/name2id.json", help="name2id.json 的路徑")
parse.add_argument("--embeddings_path", type=str, default="./eval/CDs_and_Vinyl/embeddings.pt", help="item embeddings 的路徑")
args = parse.parse_args()

def read_json(json_file:str) -> dict:
    with open(json_file, 'r', encoding='utf-8') as f: # [修改] 加上 encoding
        return json.load(f)

category = args.category
# 確保路徑正確
id2name = read_json(args.id2name_path)
name2id = read_json(args.name2id_path)
embeddings = torch.load(args.embeddings_path)

# [修改] 移除了 name2genre 和 genre_dict 的讀取
# name2genre = read_json(f"./eval/{category}/name2genre.json")
# genre_dict = read_json(f"./eval/{category}/genre_dict.json")

def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

# [修改] 移除了 gh 函數 (Genre History Distribution)

# =========================
# [新增] Step 1: 用 train.json 統計 Head/Tail 分組
# =========================
def build_head_tail_from_train_input(train_json_path: str, name2id: dict, id2name: dict, head_ratio: float = 0.2):
    """
    統計 train.json 中 input 欄位出現過的 item 次數，定義 Head/Tail。
    """
    print(f"Loading training data from {train_json_path} for popularity check...")
    try:
        with open(train_json_path, "r", encoding='utf-8') as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Train file not found at {train_json_path}")
        return set(), set(), {}

    # 初始化計數器 (確保所有 item 都有 key)
    popularity_count = {int(i): 0 for i in id2name.keys()}

    for ex in train_data:
        input_text = ex.get("input", "")
        # 解析 input 中的歷史紀錄
        names = re.findall(r'"([^"]+)"', input_text)
        
        # 確保同一個 user history 中重複提到的只算一次
        unique_ids_in_example = set()
        for name in names:
            if name in name2id:
                unique_ids_in_example.add(int(name2id[name]))

        for iid in unique_ids_in_example:
            popularity_count[iid] += 1

    # 排序
    all_item_ids = list(popularity_count.keys())
    # 由大到小排序 (Most Popular -> Least Popular)
    all_item_ids.sort(key=lambda x: popularity_count[x], reverse=True)

    # 切分 Head/Tail
    n_items = len(all_item_ids)
    n_head = max(1, math.ceil(n_items * head_ratio))
    
    head_items = set(all_item_ids[:n_head])
    tail_items = set(all_item_ids[n_head:])
    
    return head_items, tail_items, popularity_count

# =========================
# Main Process
# =========================

result_json = args.input_dir
f = open(result_json, 'r', encoding='utf-8')
test_data = json.load(f)
total = 0

# Identify your sentence-embedding model
# [注意] 請確保此路徑在您的環境是正確的
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

embeddings = torch.tensor(embeddings).cuda()
text = []

print("Extracting predictions...")
for i,_ in tqdm(enumerate(test_data)):
    if(len(_["predict"])>0):
        if(len(_['predict'][0])==0):
            text.append("NAN")
            # print("Empty prediction!")
        else:
            match = re.search(r'"([^"]*)', _['predict'][0])
            if match:
                name = match.group(1)
                text.append(name)
            else:
                text.append(_['predict'][0].split('\n', 1)[0])
    else:
        text.append("NAN")
        print("Empty:")

print("Encoding predictions...")
predict_embeddings = []
for i, batch_input in tqdm(enumerate(batch(text, 32))): # [修改] batch_size 加大一點加快速度
    predict_embeddings.append(torch.tensor(model.encode(batch_input)))
predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()

print("Calculating Distances...")
dist = torch.cdist(predict_embeddings, embeddings, p=2)

batch_size = 1
num_batches = (dist.size(0) + batch_size - 1) // batch_size
rank_list = []
print("Ranking...")
for i in tqdm(range(num_batches), desc="Processing Batches"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, dist.size(0))
    batch_dist = dist[start_idx:end_idx]

    # argsort 兩次得到 rank (名次)
    batch_rank = batch_dist.argsort(dim=-1).argsort(dim=-1)
    torch.cuda.empty_cache()
    rank_list.append(batch_rank)

rank_list = torch.cat(rank_list, dim=0)

# =========================
# [新增] 執行 Head/Tail 分組邏輯
# =========================
# 假設 train.json 在 data 目錄下，格式如 data/CDs_and_Vinyl/train.json
train_json_path = args.train_data_for_head_tail

head_items, tail_items, popularity_count = build_head_tail_from_train_input(
    train_json_path=train_json_path,
    name2id=name2id,
    id2name=id2name,
    head_ratio=0.2
)

print(f"[Split Info] Total Items: {len(id2name)}")
print(f"[Split Info] Head Items (Top 20%): {len(head_items)}")
print(f"[Split Info] Tail Items (Bottom 80%): {len(tail_items)}")

# =========================
# [修改] 計算 Metrics (區分 Head/Tail)
# =========================
topk_list = [int(args.topk)]

# 儲存每個樣本的 topk 推薦結果
topk_recs_all = [] 

NDCG_head = []
HR_head = []
NDCG_tail = []
HR_tail = []

for topk in topk_list:
    S_ndcg_head = 0.0
    S_hr_head = 0.0
    cnt_head = 0 # Head 類別的樣本數

    S_ndcg_tail = 0.0
    S_hr_tail = 0.0
    cnt_tail = 0 # Tail 類別的樣本數

    for i in tqdm(range(len(test_data)), desc="Calculating Metrics......"):
        rank = rank_list[i]
        
        # 1. 處理 Top-K 推薦列表輸出 (不論 target 是否存在都輸出，保持 index 一致)
        # 取出距離最近 (rank 最小) 的前 K 個 ID
        topk_vals, topk_indices = torch.topk(rank, k=topk, largest=False)
        current_recs = [id2name[str(idx.item())] for idx in topk_indices]
        topk_recs_all.append(current_recs)

        # 2. 處理 Metrics 計算
        target_name = test_data[i]['output']
        # 清理字串
        target_name = target_name.strip().strip('"')

        if target_name in name2id:
            target_id = int(name2id[target_name]) # 轉 int
            total += 1
        else:
            continue
                
        rankId = rank[target_id].item() # 轉 scalar

        # 判斷是 Head 還是 Tail
        is_head = target_id in head_items
        
        # 計算分數
        score_ndcg = 0.0
        score_hr = 0.0
        
        if rankId < topk:
            score_ndcg = (1 / math.log(rankId + 2))
            score_hr = 1.0
        
        # 歸類累加
        if is_head:
            S_ndcg_head += score_ndcg
            S_hr_head += score_hr
            cnt_head += 1
        else:
            S_ndcg_tail += score_ndcg
            S_hr_tail += score_hr
            cnt_tail += 1

    # 計算平均值 (Normalize)
    # 避免除以 0
    avg_ndcg_head = (S_ndcg_head / cnt_head / (1 / math.log(2))) if cnt_head > 0 else 0.0
    avg_hr_head = (S_hr_head / cnt_head) if cnt_head > 0 else 0.0
    
    avg_ndcg_tail = (S_ndcg_tail / cnt_tail / (1 / math.log(2))) if cnt_tail > 0 else 0.0
    avg_hr_tail = (S_hr_tail / cnt_tail) if cnt_tail > 0 else 0.0

    NDCG_head.append(avg_ndcg_head)
    HR_head.append(avg_hr_head)
    NDCG_tail.append(avg_ndcg_tail)
    HR_tail.append(avg_hr_tail)

print(f"Head Items Count in Test: {cnt_head}")
print(f"Tail Items Count in Test: {cnt_tail}")
print(f"Head: NDCG@{topk}={NDCG_head[0]:.4f}, HR@{topk}={HR_head[0]:.4f}")
print(f"Tail: NDCG@{topk}={NDCG_tail[0]:.4f}, HR@{topk}={HR_tail[0]:.4f}")

# =========================
# [修改] 輸出結果檔案
# =========================
eval_dic = {}
eval_dic["model"] = args.input_dir
eval_dic["category"] = category

# 儲存 Head/Tail 結果
eval_dic[f"NDCG_head@{topk_list[0]}"] = NDCG_head[0]
eval_dic[f"HR_head@{topk_list[0]}"] = HR_head[0]
eval_dic[f"NDCG_tail@{topk_list[0]}"] = NDCG_tail[0]
eval_dic[f"HR_tail@{topk_list[0]}"] = HR_tail[0]

# 補充資訊
eval_dic["Count_Head"] = cnt_head
eval_dic["Count_Tail"] = cnt_tail

# 1. 儲存主要的 eval_result.json
file_path = args.output_dir
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = []
else:
    data = []

data.append(eval_dic)

with open(args.output_dir, 'w') as file:
    json.dump(data, file, separators=(',', ': '), indent=2)

print(f"Evaluation results saved to {args.output_dir}")

# 2. [新增] 儲存 Top-K 推薦列表 (這對 Qualitative Analysis 很有用)
topk_path = args.output_dir.replace(".json", f"_top{topk_list[0]}_recs.json")
with open(topk_path, 'w', encoding='utf-8') as f:
    json.dump(topk_recs_all, f, indent=2, ensure_ascii=False)
print(f"Top-{topk_list[0]} recommendations saved to {topk_path}")

# =========================
# [修改] CSV 更新邏輯 (只存 Head/Tail)
# =========================
def update_csv(dataset_name, model_name, metrics_dict, csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Dataset", "Model"]) # 創建新檔

    required_columns = ["Dataset", "Model"]
    # 確保基本欄位存在
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    condition = (df["Dataset"] == dataset_name) & (df["Model"] == model_name)
    if not condition.any():
        new_row = {col: None for col in df.columns}
        new_row["Dataset"] = dataset_name
        new_row["Model"] = model_name
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        condition = (df["Dataset"] == dataset_name) & (df["Model"] == model_name)

    for metric, value in metrics_dict.items():
        if metric not in df.columns:
            # print(f"注意：指標 '{metric}' 不在 CSV 文件列中，已添加該列。")
            df[metric] = 0.0
        df.loc[condition, metric] = value

    df.to_csv(csv_file, index=False)
    print(f"CSV 文件已更新：{csv_file}")

if args.exp_csv != None:
    metric_dic = {}
    metric_dic[f"NDCG_head@{args.topk}"] = eval_dic[f"NDCG_head@{topk_list[0]}"]
    metric_dic[f"HR_head@{args.topk}"] = eval_dic[f"HR_head@{topk_list[0]}"]
    metric_dic[f"NDCG_tail@{args.topk}"] = eval_dic[f"NDCG_tail@{topk_list[0]}"]
    metric_dic[f"HR_tail@{args.topk}"] = eval_dic[f"HR_tail@{topk_list[0]}"]
    
    update_csv(category, args.model, metric_dic, args.exp_csv)