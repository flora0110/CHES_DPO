# 檔案位置: train/utils.py
import os
import random
import numpy as np
import torch
import json

def safe_write_json(file_path, data, indent=2):
    """
    Safely write data to a JSON file. If the directory does not exist,
    it will be created automatically. If the file already exists,
    a warning will be shown and a numeric suffix will be appended to avoid overwriting.

    Args:
        file_path (str): Target file path for saving the JSON file.
        data: The data to be serialized using json.dump.
        indent (int, optional): Indentation level for JSON formatting. Defaults to 2.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    base_path, ext = os.path.splitext(file_path)
    final_path = file_path
    counter = 1

    while os.path.exists(final_path):
        print(f"Warning: {final_path} already exists. Generating new file name...")
        final_path = f"{base_path}_{counter}{ext}"
        counter += 1

    with open(final_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    print(f"Saved data to {final_path}")



def safe_load_json(file_path):
    """
    Safely load a JSON file. If the file does not exist or is not valid JSON, raise an error.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from {file_path}: {e}")
    
    return data

def prepare_output_dir(output_path: str, check_subdir: str = "final_model", allow_existing: bool = False) -> str:
    """
    Check if a specified subdirectory (default "final_model") exists within the output path.
    If it exists, append a numeric suffix to the output directory to avoid overwriting.
    
    If check_subdir is None and the output_path already exists, then:
      - If the output_path is empty, print a warning and return it as is.
      - Otherwise, generate a new directory name with a numeric suffix.
    
    Additionally, if allow_existing is True:
      - If check_subdir is None and output_path exists, directly return output_path regardless of contents.
      - If check_subdir is provided and it exists under output_path, return output_path.
      - Otherwise, create the missing subdirectory and return output_path.
    
    Args:
        output_path (str): The desired directory path where output will be saved.
        check_subdir (str or None): The subdirectory name to check. If None, the output_path itself is checked.
        allow_existing (bool): If True, use the existing directory (or create missing subdirectory) without appending a suffix.
        
    Returns:
        str: The final output directory path that can be used safely.
    """
    # Create the base output directory if it does not exist.
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Created base output dir: {output_path}")

    # If allow_existing is True, then use the existing directory
    if allow_existing:
        if check_subdir is None:
            print(f"Using existing output directory: {output_path}")
            return output_path
        else:
            subdir = os.path.join(output_path, check_subdir)
            if not os.path.exists(subdir):
                os.makedirs(subdir, exist_ok=True)
                print(f"Created subdirectory: {subdir}")
            else:
                print(f"Using existing subdirectory: {subdir}")
            return subdir

    # When allow_existing is False, follow the original logic.
    if check_subdir is None:
        if len(os.listdir(output_path)) == 0:
            print(f"Warning: Output directory '{output_path}' exists and is empty. No need to create a new directory.")
            return output_path
        else:
            base_path = output_path
            counter = 1
            final_output_path = base_path
            while os.path.exists(final_output_path) and os.listdir(final_output_path):
                print(f"Warning: '{final_output_path}' already exists and is not empty. Generating a new output directory name...")
                final_output_path = f"{base_path}_{counter}"
                counter += 1
            os.makedirs(final_output_path, exist_ok=True)
            print(f"Using output directory: {final_output_path}")
            return final_output_path
    else:
        if check_subdir:
            final_model_dir = os.path.join(output_path, check_subdir)
        else:
            final_model_dir = output_path

        counter = 1
        final_output_path = output_path
        while os.path.exists(final_model_dir):
            print(f"Warning: '{final_model_dir}' already exists. Generating a new output directory name...")
            final_output_path = f"{output_path}_{counter}"
            if check_subdir:
                final_model_dir = os.path.join(final_output_path, check_subdir)
            else:
                final_model_dir = final_output_path
            counter += 1

        os.makedirs(final_output_path, exist_ok=True)
        print(f"Using output directory: {final_output_path}")
        return final_output_path

def set_seed(seed):
    """
    固定所有必要的隨機種子以確保可重現性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 如果需要極致的確定性 (會犧牲一點效能)，可以打開下面這兩行
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def save_run_script_content(sh_file_path, output_file_path):
    """
    讀取 sh_file_path 的內容，並將其寫入到 output_file_path 所在的目錄中。
    檔名會儲存為 'run_config_backup.sh'
    """
    print(f"Backing up shell script from {sh_file_path} to output directory: {output_file_path}...")
    if not sh_file_path:
        return

    if not os.path.exists(sh_file_path):
        print(f"Warning: Shell script not found at {sh_file_path}")
        return

    try:
        # 取得存放目標檔案的資料夾
        dest_dir = os.path.dirname(output_file_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(dest_dir, "run_config_backup.sh")
        
        # 讀取並寫入
        with open(sh_file_path, 'r', encoding='utf-8') as f_src:
            content = f_src.read()
        
        with open(dest_path, 'w', encoding='utf-8') as f_dest:
            f_dest.write(content)
            
        print(f"Shell script has been backed up to: {dest_path}")
        
    except Exception as e:
        print(f"Warning: Failed to backup shell script. Error: {e}")