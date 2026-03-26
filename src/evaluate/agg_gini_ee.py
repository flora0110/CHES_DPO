import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Gini and EE evaluation results.")
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
    output_md_path = os.path.join(args.base_dir, f"{args.metrics}_metrics_summary.md")
    
    md_lines = []
    md_lines.append(f"# Evaluation Summary (Gini & EE)")
    md_lines.append(f"**Directory:** `{args.base_dir}`\n")

    # 1. 第一層迴圈：遍歷每個方法 (Metric/Method)
    method = args.metrics
    
    md_lines.append(f"## Method: {method}")
    
    # 2. 第二層迴圈：遍歷每個 Top K
    for k in args.topks:
        md_lines.append(f"### Top {k}")
        
        # [修改] 建立表格 Header，加入 EE
        md_lines.append("| its | GiniIndex ↓ | coverage ↑ | num_unique ↑ | EE ↑ |")
        md_lines.append("| :--- | ----------: | ----------: | -----------: | ---: |")
        
        # 用來儲存數值以便計算平均
        gini_values = []
        cov_values = []
        unique_values = []
        ee_values = [] # [新增] 儲存 EE 數值

        # 3. 第三層迴圈：遍歷 Iterations
        for i in range(args.iterations):
            # 建構檔案路徑
            # 假設 Gini 檔案: {base_dir}/it{i}/{method}/eval_gini_top{k}.json
            gini_path = os.path.join(args.base_dir, f"it{i}", method, f"eval_gini_top{k}.json")
            
            # [新增] 假設 EE 檔案: {base_dir}/it{i}/{method}/eval_ee_top{k}.json
            # 注意：如果你上一部的檔名不同 (例如 ee_result_top{k}.json)，請在此處修改檔名
            ee_path = os.path.join(args.base_dir, f"it{i}", method, f"eval_ee_top{k}.json")
            
            # 讀取檔案
            res_gini = read_json_result(gini_path)
            res_ee = read_json_result(ee_path)
            
            # 初始化顯示字串
            val_gini, val_cov, val_unique, val_ee = "-", "-", "-", "-"
            
            # --- 處理 Gini 數據 ---
            if res_gini:
                g = res_gini.get("GiniIndex")
                c = res_gini.get("coverage")
                u = res_gini.get("num_recommended_unique")

                if isinstance(g, (int, float)): 
                    gini_values.append(g)
                    val_gini = f"{g:.4f}"
                
                if isinstance(c, (int, float)): 
                    cov_values.append(c)
                    val_cov = f"{c:.4f}"
                    
                if isinstance(u, (int, float)): 
                    unique_values.append(u)
                    val_unique = f"{u}"

            # --- [新增] 處理 EE 數據 ---
            if res_ee:
                # 兼容上一部程式碼產生的 {"value": 0.xxx, "metric": "EE"} 結構
                # 或是標準的 {"EE": 0.xxx} 結構
                e_val = res_ee.get("value") if "value" in res_ee else res_ee.get("EE")
                
                if isinstance(e_val, (int, float)):
                    ee_values.append(e_val)
                    val_ee = f"{e_val:.4f}"

            # 寫入一行表格
            md_lines.append(f"| {i} | {val_gini} | {val_cov} | {val_unique} | {val_ee} |")
        
        # --- 計算並加入 Average 行 ---
        avg_gini_str = "-"
        avg_cov_str = "-"
        avg_unique_str = "-"
        avg_ee_str = "-"

        if gini_values: avg_gini_str = f"**{sum(gini_values) / len(gini_values):.4f}**"
        if cov_values: avg_cov_str = f"**{sum(cov_values) / len(cov_values):.4f}**"
        if unique_values: avg_unique_str = f"**{sum(unique_values) / len(unique_values):.1f}**"
        if ee_values: avg_ee_str = f"**{sum(ee_values) / len(ee_values):.4f}**" # [新增] EE 平均
            
        md_lines.append(f"| **Avg** | {avg_gini_str} | {avg_cov_str} | {avg_unique_str} | {avg_ee_str} |")

        md_lines.append("") # 表格後空一行
    
    md_lines.append("---") 

    # 寫入檔案
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    
    print(f"Successfully generated summary at: {output_md_path}")

if __name__ == "__main__":
    main()