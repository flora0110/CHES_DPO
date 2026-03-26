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

parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="result file")
parse.add_argument("--model",type=str, default="SPRec", help="result file")
parse.add_argument("--exp_csv",type=str, default=None, help="result file")
parse.add_argument("--output_dir",type=str, default="./", help="eval_result")
parse.add_argument("--topk",type=str, default="./", help="topk")
parse.add_argument("--gamma",type=float,default=0.0,help="gamma")
parse.add_argument("--category",type=str,default="CDs_and_Vinyl",help="gamma")
args = parse.parse_args()

def read_json(json_file:str) -> dict:
    f = open(json_file, 'r')
    return json.load(f)

category = args.category
id2name = read_json(f"../SPRec/eval/{category}/id2name.json")
name2id = read_json(f"../SPRec/eval/{category}/name2id.json")
embeddings = torch.load(f"../SPRec/eval/{category}/embeddings.pt")
name2genre = read_json(f"../SPRec/eval/{category}/name2genre.json")
genre_dict = read_json(f"../SPRec/eval/{category}/genre_dict.json")

def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

def sum_of_first_i_keys(sorted_dic, i):
    keys = list(sorted_dic.values())[:i]
    return sum(keys)

def gh(category:str,test_data):
    notin_count = 0
    in_count = 0
    name2genre=read_json(f"../SPRec/eval/{category}/name2genre.json")
    genre_dict = read_json(f"../SPRec/eval/{category}/genre_dict.json")
    for data in tqdm(test_data,desc="Processing category data......"):
        input = data['input']
        names = re.findall(r'"([^"]+)"', input)
        for name in names:
            if name in name2genre:
                in_count += 1
                genres = name2genre[name]
            else:
                notin_count += 1
                continue
            select_genres = []
            for genre in genres:
                if genre in genre_dict:
                        select_genres.append(genre)
            if(len(select_genres)>0):
                for genre in select_genres:
                    genre_dict[genre] += 1/len(select_genres)
    gh = [genre_dict[x] for x in genre_dict]
    gh_normalize = [x/sum(gh) for x in gh]
    print(f"InCount:{in_count}\nNotinCount:{notin_count}")
    return gh_normalize


# =========================
# [新增] Step 1: 用 train.json 的 input 統計 item 被多少「人/樣本」點擊過
# =========================
def build_head_tail_from_train_input(train_json_path: str, name2id: dict, id2name: dict, head_ratio: float = 0.2):
    """
    [新增]
    以 train data 的 input(history) 統計 item popularity：
      - 每筆 train example 視為一個 user/session
      - item 出現在該 example 的 input 中 => popularity +1（同一筆內重複只算一次）
    依 popularity 排序切分：
      - Head: 前 head_ratio（預設 20%）
      - Tail: 後 (1-head_ratio)
    """
    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    # popularity_count[item_id] = number_of_train_examples_that_contain_it
    popularity_count = {int(i): 0 for i in id2name.keys()}  # [新增] universe 以 id2name 為準，未出現的自然為 0

    for ex in train_data:
        input_text = ex.get("input", "")
        names = re.findall(r'"([^"]+)"', input_text)

        # [新增] 同一個 train example 內只算一次（避免 history 裡同 item 重複出現）
        unique_ids_in_example = set()
        for name in names:
            if name in name2id:
                unique_ids_in_example.add(int(name2id[name]))

        for iid in unique_ids_in_example:
            # [新增] iid 一定在 popularity_count 的 universe 內才加；不在則忽略
            if iid in popularity_count:
                popularity_count[iid] += 1

    # [新增] 依 popularity 由高到低排序
    all_item_ids = list(popularity_count.keys())
    all_item_ids.sort(key=lambda x: popularity_count[x], reverse=True)

    # [新增] Head = 前 20% item（用 ceil 避免太少）
    n_items = len(all_item_ids)
    n_head = max(1, math.ceil(n_items * head_ratio))
    head_items = set(all_item_ids[:n_head])
    tail_items = set(all_item_ids[n_head:])

    return head_items, tail_items, popularity_count


result_json = args.input_dir
f = open(result_json, 'r')
test_data = json.load(f)

total = 0

# Identify your sentence-embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

from tqdm import tqdm
embeddings = torch.tensor(embeddings).cuda()

text = []
for i,_ in tqdm(enumerate(test_data)):
    if(len(_["predict"])>0):
        if(len(_['predict'][0])==0):
            text.append("NAN")
            print("Empty prediction!")
        else:
            match = re.search(r'"([^"]*)', _['predict'][0])
            if match:
                name = match.group(1)
                text.append(name)
            else:
                text.append(_['predict'][0].split('\n', 1)[0])
    else:
        print("Empty:")

predict_embeddings = []
for i, batch_input in tqdm(enumerate(batch(text, 8))):
    predict_embeddings.append(torch.tensor(model.encode(batch_input)))
predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
predict_embeddings.size()
dist = torch.cdist(predict_embeddings, embeddings, p=2)

batch_size = 1
num_batches = (dist.size(0) + batch_size - 1) // batch_size
rank_list = []
for i in tqdm(range(num_batches), desc="Processing Batches"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, dist.size(0)
)
    batch_dist = dist[start_idx:end_idx]

    batch_rank = batch_dist.argsort(dim=-1).argsort(dim=-1)
    torch.cuda.empty_cache ()
    rank_list.append(batch_rank)

rank_list = torch.cat(rank_list, dim=0)

# =========================
# [新增] Step 1.5: 由 train input 切分 Head/Tail items
# =========================
# [新增] 你指定的路徑：./data/Goodreads/train.json（用 category 組合，方便未來換 dataset）
train_json_path = f"./sampled_data/{category}/train.json"
head_items, tail_items, popularity_count = build_head_tail_from_train_input(
    train_json_path=train_json_path,
    name2id=name2id,
    id2name=id2name,
    head_ratio=0.2
)
print(f"[Head/Tail Split] total_items={len(id2name)} head={len(head_items)} tail={len(tail_items)}")
# 你如果想 sanity check，可印幾個最熱門 item：
# top10 = sorted(popularity_count.items(), key=lambda x: x[1], reverse=True)[:10]
# print("Top-10 popular items (by train input):", [(id2name[str(i)], c) for i, c in top10])

# =========================
# [修改] Step 2: 只計算 Head/Tail 的 HR@K / NDCG@K，其餘不需要
# =========================
topk_list = [int(args.topk)]

# [新增] 分組累加器
NDCG_head = []
HR_head = []
NDCG_tail = []
HR_tail = []

# =========================
# [新增] 存 Top-K recommendation（每個 sample 一個 list）以及全域統計
# =========================
# [新增] 注意：若你之後想支援多個 topk（topk_list 有多個值），建議改成 dict[topk]=...
#        目前 topk_list 只有一個值，所以用單一 list/counter 即可。
topk_recs_all = []                 # [新增] list of list[str]，與 test_data 對齊（跳過 target 不在 name2id 的樣本會造成長度不等，見下方處理）
topk_recs_all_meta = []            # [新增] 保留每筆對應的 index/target，避免跳過樣本後對不上
topk_global_counter = Counter()    # [新增] 全域 topk 推薦頻次統計（name-level）


for topk in topk_list:
    S_ndcg_head = 0.0
    S_hr_head = 0.0
    cnt_head = 0

    S_ndcg_tail = 0.0
    S_hr_tail = 0.0
    cnt_tail = 0

    for i in tqdm(range(len(test_data)), desc="Calculating Head/Tail HR/NDCG......"):
        rank = rank_list[i]

        # Target name
        target_name = test_data[i]['output']
        target_name = target_name.strip().strip('"')

        if target_name in name2id:
            target_id = int(name2id[target_name])
            total += 1
        else:
            # [保留原行為] target 不在 name2id => 跳過
            continue

        rankId = rank[target_id].item()  # [修改] 明確取 python scalar

        # =========================
        # [新增] 取出 Top-K recommendation（由 rank 向量取最小的 K 個）
        # =========================
        # rank: 每個 item 的名次(0..N-1)，越小越好 => largest=False
        topk_item_ids = torch.topk(rank, k=topk, largest=False).indices  # [新增] tensor shape [topk]
        topk_item_names = [id2name[str(int(_id))] for _id in topk_item_ids]  # [新增] 轉成 item name list

        # [新增] 記錄每筆 sample 的 topk recs（注意：只記錄 target 存在於 name2id 的樣本）
        topk_recs_all.append(topk_item_names)
        topk_recs_all_meta.append({  # [新增] 加 meta，避免之後你要對回原 test index 時混亂
            "test_index": i,
            "target_name": target_name,
            "target_id": target_id
        })
        topk_global_counter.update(topk_item_names)  # [新增] 全域統計

        # [新增] 判斷 Head/Tail（以 target item 分組）
        if target_id in head_items:
            cnt_head += 1
            if rankId < topk:
                S_ndcg_head += (1 / math.log(rankId + 2))
                S_hr_head += 1
        else:
            # [新增] universe 裡除了 head 就是 tail（含 popularity=0）
            cnt_tail += 1
            if rankId < topk:
                S_ndcg_tail += (1 / math.log(rankId + 2))
                S_hr_tail += 1

    # [新增] 正規化：和你原本一致（NDCG 再除以 ideal=1/log(2)）
    # 注意：分母改成各自 group 的樣本數（cnt_head/cnt_tail），避免 head/tail 測試樣本數不一樣時被稀釋
    if cnt_head > 0:
        NDCG_head.append((S_ndcg_head / cnt_head) / (1 / math.log(2)))
        HR_head.append(S_hr_head / cnt_head)
    else:
        NDCG_head.append(0.0)
        HR_head.append(0.0)

    if cnt_tail > 0:
        NDCG_tail.append((S_ndcg_tail / cnt_tail) / (1 / math.log(2)))
        HR_tail.append(S_hr_tail / cnt_tail)
    else:
        NDCG_tail.append(0.0)
        HR_tail.append(0.0)

print(f"Head: NDCG@{topk_list[0]}={NDCG_head} HR@{topk_list[0]}={HR_head}")
print(f"Tail: NDCG@{topk_list[0]}={NDCG_tail} HR@{topk_list[0]}={HR_tail}")

# =========================
# [停用] 原本的多種 metrics（genre/diversity/ORRatio/CSV）不再需要：保留但不執行
# =========================
# NDCG = []
# HR = []
# diversity = []
# diversity_dic = {}
# MGU_genre = []
# DGU_genre = []
# pop_count = {}
# genre_count = {}
# notin = 0
# notin_count = 0
# in_count = 0
# diversity_set = set()
# ...（整段原本計算 diversity/genre fairness/ORRatio 的迴圈已由上方 Head/Tail 版本取代）

# gh_genre = gh(category,test_data)
# gp_genre = [genre_dict[x] for x in genre_dict]
# gp_genre = [x/sum(gp_genre) for x in gp_genre]
# dis_genre = [gp_genre[i]-gh_genre[i] for i in range(len(gh_genre))]
# DGU_genre = max(dis_genre)-min(dis_genre)
# dis_abs_genre = [abs(x) for x in dis_genre]
# MGU_genre = sum(dis_abs_genre) / len(dis_genre)
# print(f"DGU:{DGU_genre}")
# print(f"MGU:{MGU_genre}")
# div_ratio = diversity[0] / (total*topk)
# print(f"DivRatio:{div_ratio}")

# =========================
# [修改] 輸出 eval 結果：只保留 Head/Tail 的 HR/NDCG
# =========================
eval_dic = {}
eval_dic["model"] = args.input_dir
eval_dic["category"] = category  # [新增] 方便追蹤
eval_dic[f"NDCG_head@{topk_list[0]}"] = NDCG_head
eval_dic[f"HR_head@{topk_list[0]}"] = HR_head
eval_dic[f"NDCG_tail@{topk_list[0]}"] = NDCG_tail
eval_dic[f"HR_tail@{topk_list[0]}"] = HR_tail
# [新增] 可選：把 head/tail 的 item 數量也寫進去
eval_dic["num_head_items"] = len(head_items)
eval_dic["num_tail_items"] = len(tail_items)
eval_dic["head_test_sample_count"] = cnt_head    # [新增]
eval_dic["tail_test_sample_count"] = cnt_tail    # [新增]

# =========================
topk_output_path = args.output_dir + f".top{topk_list[0]}.json"  # [新增]
with open(topk_output_path, "w") as f_topk:  # [新增]
    json.dump(topk_recs_all, f_topk, separators=(',', ': '), indent=2)  # [新增]


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

# =========================
# [停用] 原本 CSV 更新：你說「其餘不需要」
# =========================
# def update_csv(dataset_name, model_name, metrics_dict, csv_file):
#     df = pd.read_csv(csv_file)
#     required_columns = ["Dataset", "Model"]
#     if not all(col in df.columns for col in required_columns):
#         raise ValueError("CSV 文件必须包含 'Dataset' 和 'Model' 列")
#     condition = (df["Dataset"] == dataset_name) & (df["Model"] == model_name)
#     if not condition.any():
#         new_row = {col: None for col in df.columns}
#         new_row["Dataset"] = dataset_name
#         new_row["Model"] = model_name
#         new_row_df = pd.DataFrame([new_row])
#         df = pd.concat([df, new_row_df], ignore_index=True)
#         condition = (df["Dataset"] == dataset_name) & (df["Model"] == model_name)
#     for metric, value in metrics_dict.items():
#         if metric not in df.columns:
#             print(f"注意：指标 '{metric}' 不在 CSV 文件列中，已添加该列并初始化为0。")
#             df[metric] = 0
#         df.loc[condition, metric] = value
#     df.to_csv(csv_file, index=False)
#     print(f"CSV 文件已更新：{csv_file}")
#
# if args.exp_csv != None:
#     metric_dic = {}
#     metric_dic[f"NDCG_head@{args.topk}"] = eval_dic[f"NDCG_head@{topk_list[0]}"]
#     metric_dic[f"HR_head@{args.topk}"] = eval_dic[f"HR_head@{topk_list[0]}"]
#     metric_dic[f"NDCG_tail@{args.topk}"] = eval_dic[f"NDCG_tail@{topk_list[0]}"]
#     metric_dic[f"HR_tail@{args.topk}"] = eval_dic[f"HR_tail@{topk_list[0]}"]
#     update_csv(category, args.model, metric_dic, args.exp_csv)
