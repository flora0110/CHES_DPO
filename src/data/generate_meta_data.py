import json
import os
import re
import torch
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ==========================================
# Part 1: 生成 Metadata (id2name, name2id)
# ==========================================

def extract_titles_from_text(text):
    """
    使用 Regex 從文字中提取被雙引號包圍的標題。
    例如: The user has watched ... "Toy Story (1995)", "Die Hard (1988)"
    會提取出: ['Toy Story (1995)', 'Die Hard (1988)']
    """
    # 尋找所有被雙引號包圍的字串: "(...)"
    # (.*?) 是非貪婪匹配，避免一次抓到跨越多個標題的長字串
    pattern = r'"(.*?)"'
    titles = re.findall(pattern, text)
    return titles

def generate_meta_files(dataset_path, output_dir, category):
    print(f"=== 步驟 1: 從 {dataset_path} 讀取資料並建立 ID 映射 ===")
    
    unique_titles = set()
    files_to_process = [f'{category}_train.json', f'{category}_test.json', f'{category}_valid.json']
    
    # 1. 遍歷三個檔案收集所有出現過的標題
    for filename in files_to_process:
        file_path = os.path.join(dataset_path, filename)
        if not os.path.exists(file_path):
            print(f"警告: 找不到 {filename}，跳過。")
            continue
        
        print(f"正在處理: {filename}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for entry in tqdm(data):
                # 處理 input 欄位 (歷史紀錄)
                if 'input' in entry:
                    titles = extract_titles_from_text(entry['input'])
                    unique_titles.update(titles)
                
                # 處理 output 欄位 (目標項目)
                if 'output' in entry:
                    # output 通常只有一個標題，但也可能有引號
                    titles = extract_titles_from_text(entry['output'])
                    unique_titles.update(titles)

    print(f"總共找到 {len(unique_titles)} 個唯一的項目標題。")

    # 2. 建立映射字典 (排序以確保順序固定)
    sorted_titles = sorted(list(unique_titles))
    
    id2name = {}
    name2id = {}
    
    for idx, title in enumerate(sorted_titles):
        id_str = str(idx) # ID 使用字串格式，符合一般 JSON key 習慣
        id2name[id_str] = title
        name2id[title] = int(idx)

    # 3. 儲存檔案
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'id2name.json'), 'w', encoding='utf-8') as f:
        json.dump(id2name, f, indent=4, ensure_ascii=False)
        
    with open(os.path.join(output_dir, 'name2id.json'), 'w', encoding='utf-8') as f:
        json.dump(name2id, f, indent=4, ensure_ascii=False)
        
    print(f"Metadata 已儲存至 {output_dir}")
    return id2name

# ==========================================
# Part 2: 生成 Embeddings
# ==========================================

def generate_embeddings(id2name, output_dir, model_name_or_path, batch_size=32):
    print(f"\n=== 步驟 2: 使用模型 {model_name_or_path} 生成 Embeddings ===")
    
    output_path = os.path.join(output_dir, "embeddings.pt")

    # 1. 準備標題列表 (必須確保依照 ID 0, 1, 2... 的順序排列)
    # id2name 的 key 是字串，需要轉成 int 排序
    sorted_ids = sorted(id2name.keys(), key=lambda x: int(x))
    all_titles = [id2name[id_str] for id_str in sorted_ids]
    
    print(f"準備編碼 {len(all_titles)} 個標題...")
    print(f"範例 (ID 0): {all_titles[0]}")

    # 2. 加載模型
    print("Loading SentenceTransformer model...")
    # 這裡會使用您指定的本地路徑或 HuggingFace 名稱
    model = SentenceTransformer(model_name_or_path)
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Model moved to CUDA")
    else:
        print("CUDA not available, using CPU")

    # 3. 計算 Embeddings
    print("Encoding titles to embeddings...")
    embeddings = model.encode(
        all_titles, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_tensor=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 4. 儲存結果
    print(f"Saving embeddings to {output_path}...")
    torch.save(embeddings, output_path)
    print(f"Done! Embeddings shape: {embeddings.shape}")

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset_path = "./data_sprec"
    # output_dir = "./eval/CDs_and_Vinyl_new/"
    model_path = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
    batch_size = 32
    for category in ["MovieLens", "Goodreads", "CDs_and_Vinyl", "Steam"]:
        print(f"\n=== 處理資料類別: {category} ===")

        output_dir = f"./eval/{category}/"
        generated_id2name = generate_meta_files(dataset_path, output_dir,category)
        
        # 執行 Part 2
        generate_embeddings(generated_id2name, output_dir, model_path, batch_size)