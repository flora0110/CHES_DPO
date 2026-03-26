import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Gini evaluation results.")
    parser.add_argument("--base_dir", type=str, required=True, 
                        help="The base directory containing iteration folders (e.g., ./models/SPRec/Goodreads_2048_2e-05)")
    parser.add_argument("--iterations", type=int, required=True, 
                        help="Total number of iterations to check")
    parser.add_argument("--metrics", type=str, default="",
                        help="string of method names (folder names)")
    parser.add_argument("--topks", nargs='+', type=int, default=[1, 5, 10], 
                        help="List of Top K values to aggregate")
    return parser.parse_args()

def read_json_result(filepath):
    """Safely reads the JSON file. Handles list of dicts by taking the last one."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是 list，取最後一個 (最新的結果)
        if isinstance(data, list):
            if len(data) > 0:
                return data[-1]
            else:
                return None
        return data
    except Exception as e:
        print(f"[Warning] Error reading {filepath}: {e}")
        return None

def main():
    args = parse_args()
    
    # 輸出的 Markdown 檔案路徑
    output_md_path = os.path.join(args.base_dir, f"{args.metrics}_gini_metrics.md")
    
    md_lines = []
    md_lines.append(f"# Gini Index & Coverage Summary")
    md_lines.append(f"**Directory:** `{args.base_dir}`\n")

    # 1. 第一層迴圈：遍歷每個方法 (Metric/Method)
    method = args.metrics
    
    md_lines.append(f"## Method: {method}")
    
    # 2. 第二層迴圈：遍歷每個 Top K
    for k in args.topks:
        md_lines.append(f"### Top {k}")
        
        # 建立表格 Header
        md_lines.append("| its | GiniIndex ↓ | coverage ↑ | num_recommended_unique ↑ |")
        md_lines.append("| :--- | ----------: | ----------: | -----------------------: |")
        # 用來儲存數值以便計算平均
        gini_values = []
        cov_values = []
        unique_values = []
        # 3. 第三層迴圈：遍歷 Iterations
        for i in range(args.iterations):
            # 建構檔案路徑: {base_dir}/it{i}/{method}/eval_gini_top{k}.json
            file_path = os.path.join(args.base_dir, f"it{i}", method, f"eval_top{k}.json")
            
            result = read_json_result(file_path)
            
            if result:
                # 取得數值，若無則顯示 N/A
                gini = result.get("GiniIndex", "N/A")
                cov = result.get("coverage", "N/A")
                num_unique = result.get("num_recommended_unique", "N/A")

                # 收集有效數據
                if isinstance(gini, (int, float)): gini_values.append(gini)
                if isinstance(cov, (int, float)): cov_values.append(cov)
                if isinstance(num_unique, (int, float)): unique_values.append(num_unique)
                
                # 格式化小數點 (如果是數字的話)
                if isinstance(gini, (int, float)): gini = f"{gini:.4f}"
                if isinstance(cov, (int, float)): cov = f"{cov:.4f}"
                
                md_lines.append(f"| {i} | {gini} | {cov} | {num_unique} |")
            else:
                # 檔案不存在或讀取失敗
                md_lines.append(f"| {i} | - | - | - |")
        
        # --- 計算並加入 Average 行 ---
        if gini_values: # 只要有數據就計算平均
            avg_gini = sum(gini_values) / len(gini_values)
            avg_cov = sum(cov_values) / len(cov_values) if cov_values else 0
            avg_unique = sum(unique_values) / len(unique_values) if unique_values else 0
            
            # 使用粗體 (**...**) 標示平均值
            md_lines.append(f"| **Avg** | **{avg_gini:.4f}** | **{avg_cov:.4f}** | **{avg_unique:.1f}** |")
        else:
            md_lines.append(f"| **Avg** | - | - | - |")

        md_lines.append("") # 表格後空一行
    
    md_lines.append("---") # 方法之間的分隔線

    # 寫入檔案
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    
    print(f"Successfully generated summary at: {output_md_path}")
    # 同時印出到 Console 方便查看
    # print("\n".join(md_lines))

if __name__ == "__main__":
    main()