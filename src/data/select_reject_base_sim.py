import re
import json
import sys
import fire
# import gradio as gr
import numpy as np
import torch
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer
import random
import transformers
from tqdm import tqdm
import json
import os
import glob 
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig,AutoTokenizer
from transformers import LlamaForCausalLM

def main(
    similarity_chunk_dir : str = "", 
    output_dir : str = "", 
    data_type: str = "train"
):
    
    # [新增] 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # [新增] 定義我們要處理的 Metrics 列表
    metrics = [
        "minus_normalized_edit_distances",
        "ches_scores",
        "ln_ches_scores",
        "last_hidden_embedding_inner_prods"
    ]
    
    # [新增] 初始化結果容器
    results = {}
    for metric in metrics:
        results[f"Max_{metric}"] = [] # Max
        results[f"Min_{metric}"] = []  # Min

    # [修改] 步驟 1: 讀取 Similarity Chunk Files (不依賴 SFT 檔案)
    print(f"Loading Similarity Chunks from {similarity_chunk_dir}...")
    
    all_similarity_items = [] # [新增] 用來儲存所有 chunk 的資料列表
    # chunk_files = sorted(glob.glob(os.path.join(similarity_chunk_dir, f"{data_type}_item_pref_similarity_chunk*.json")))
    
    # for chunk_file in tqdm(chunk_files, desc="Reading Chunks"):
    #     with open(chunk_file, 'r') as f:
    #         chunk_data = json.load(f)
    #         # 直接將資料加入總列表
    #         all_similarity_items.extend(chunk_data)

    chunk_file = os.path.join(similarity_chunk_dir, f"{data_type}_item_pref_similarity.json")
    with open(chunk_file, 'r') as f:
        chunk_data = json.load(f)
        # 直接將資料加入總列表
        all_similarity_items.extend(chunk_data)

    # [修改] 步驟 2: 直接處理 Similarity Data (不需對齊 SFT)
    print(f"Processing {len(all_similarity_items)} items...")
    
    # 直接遍歷所有讀取到的 Similarity Item
    for sim_item in tqdm(all_similarity_items, desc="Processing"):
        
        prompt_key = sim_item['prompt']
        
        # 準備基礎 DPO 結構
        base_dpo_entry = {
            "prompt": prompt_key,
            "chosen": sim_item['chosen'], 
            # rejected 待填入
        }

        # 針對每一個 Metric 計算 Max 和 Min
        for metric in metrics:
            scores_dict = sim_item.get(metric, {})
            if not scores_dict:
                continue

            # 找出最大值與最小值的 Candidate
            # key 是 candidate string, value 是分數
            
            #  (Max Score)
            max_candidate = max(scores_dict, key=scores_dict.get)
            
            #  (Min Score)
            min_candidate = min(scores_dict, key=scores_dict.get)

            # Farthest Entry
            farthest_entry = base_dpo_entry.copy()
            farthest_entry['rejected'] = f"\"{max_candidate}\"\n"
            results[f"Max_{metric}"].append(farthest_entry)

            # Nearest Entry
            nearest_entry = base_dpo_entry.copy()
            nearest_entry['rejected'] = f"\"{min_candidate}\"\n"
            results[f"Min_{metric}"].append(nearest_entry)

    # [新增] 步驟 3: 寫入檔案
    print("Saving results...")
    for key, data_list in results.items():
        output_subdir = os.path.join(output_dir, key)
        
        # [關鍵修正]：如果該子資料夾不存在，則建立它
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)

        # 設定完整的檔案路徑 (這裡根據上一段錯誤訊息，推測你是要存成 train.jsonl 還是 valid.jsonl?)
        # 假設是要存成 train.jsonl
        output_filename = os.path.join(output_subdir, f"{data_type}.jsonl")
        
        with open(output_filename, 'w') as f:
            for item in data_list:
                json.dump(item, f)
                f.write('\n')
        
        print(f"Saved {len(data_list)} items to {output_filename}")

if __name__ == "__main__":
    fire.Fire(main)