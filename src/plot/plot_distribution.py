import json
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import torch
from sentence_transformers import SentenceTransformer


def clean_item_name(text):
    if text is None:
        return None
    text = str(text).strip()
    text = text.replace("</s>", "").strip()
    # print("[DEBUG] Raw text:", text)
    # print(text)

    # 優先抓雙引號內內容，跟 evaluate 比較一致
    match = re.search(r'"([^"]+)"', text)
    if match:
        return match.group(1).strip()

    # 沒抓到就退回整行第一行
    text = text.split("\n", 1)[0].strip()

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    return text.strip()


def build_popularity(train_json_path, name2id, id2name):
    """
    popularity_count[item_id] = number of train examples whose input history contains this item
    same item repeated in one example only counts once
    """
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    popularity_count = {int(i): 0 for i in id2name.keys()}

    for ex in train_data:
        input_text = ex.get("input", "")
        names = re.findall(r'"([^"]+)"', input_text)

        unique_ids = set()
        for name in names:
            if name in name2id:
                unique_ids.add(int(name2id[name]))

        for iid in unique_ids:
            if iid in popularity_count:
                popularity_count[iid] += 1
    for ex in train_data:
        input_text = ex.get("output", "")
        names = re.findall(r'"([^"]+)"', input_text)

        unique_ids = set()
        for name in names:
            if name in name2id:
                unique_ids.add(int(name2id[name]))

        for iid in unique_ids:
            if iid in popularity_count:
                popularity_count[iid] += 1

    return popularity_count


def build_item_embedding_lookup(
    embeddings_path,
    id2name,
    sbert_model_path,
    device=None
):
    """
    載入：
      - item embeddings
      - SentenceTransformer model
    回傳：
      - embeddings_tensor: [num_items, dim]
      - sbert_model
      - ordered_item_ids: embeddings row index -> item_id
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_embeddings = torch.load(embeddings_path)
    embeddings_tensor = torch.tensor(raw_embeddings, dtype=torch.float32, device=device)

    # 假設 embeddings 的 row index 對應 item_id 0..N-1
    # 而 id2name 的 key 是字串型別 item_id
    ordered_item_ids = list(range(embeddings_tensor.size(0)))

    sbert_model = SentenceTransformer(sbert_model_path, device=device)

    return embeddings_tensor, sbert_model, ordered_item_ids


def nearest_item_id_by_text(text, sbert_model, item_embeddings, ordered_item_ids, device=None):
    """
    evaluate 同款 fallback:
      text -> SBERT embedding -> cdist to all item embeddings -> nearest top1
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if text is None or len(text.strip()) == 0:
        return None

    vec = sbert_model.encode([text], convert_to_numpy=True)
    vec = torch.tensor(vec, dtype=torch.float32, device=device)   # [1, dim]

    dist = torch.cdist(vec, item_embeddings, p=2)  # [1, num_items]
    nearest_idx = torch.argmin(dist, dim=1).item()

    if nearest_idx < 0 or nearest_idx >= len(ordered_item_ids):
        return None

    return ordered_item_ids[nearest_idx]


def resolve_name_to_item_id(
    raw_text,
    name2id,
    sbert_model=None,
    item_embeddings=None,
    ordered_item_ids=None,
    device=None
):
    """
    優先 exact match，失敗再用 nearest top1 fallback
    回傳:
      item_id, match_type
    其中 match_type ∈ {"exact", "fallback", "fail"}
    """
    cleaned = clean_item_name(raw_text)

    if cleaned in name2id:
        return int(name2id[cleaned]), "exact"

    if sbert_model is not None and item_embeddings is not None and ordered_item_ids is not None:
        fallback_id = nearest_item_id_by_text(
            cleaned,
            sbert_model=sbert_model,
            item_embeddings=item_embeddings,
            ordered_item_ids=ordered_item_ids,
            device=device
        )
        if fallback_id is not None:
            return int(fallback_id), "fallback"

    return None, "fail"


def extract_prediction_scores(
    result_json_path,
    name2id,
    popularity_count,
    sbert_model=None,
    item_embeddings=None,
    ordered_item_ids=None,
    device=None
):
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred_scores = []

    exact_count = 0
    fallback_count = 0
    fail_count = 0

    for ex in data:
        item_id, match_type = resolve_name_to_item_id(
            ex.get("predict"),
            name2id=name2id,
            sbert_model=sbert_model,
            item_embeddings=item_embeddings,
            ordered_item_ids=ordered_item_ids,
            device=device
        )

        if item_id is None:
            fail_count += 1
            continue

        if match_type == "exact":
            exact_count += 1
        elif match_type == "fallback":
            fallback_count += 1

        freq = popularity_count.get(int(item_id), 0)
        pred_scores.append(math.log1p(freq))

    print(f"[INFO] {result_json_path}")
    print(f"       prediction exact    : {exact_count}")
    print(f"       prediction fallback : {fallback_count}")
    print(f"       prediction fail     : {fail_count}")

    return np.array(pred_scores, dtype=float)


def extract_ground_truth_scores(
    result_json_path,
    name2id,
    popularity_count,
    sbert_model=None,
    item_embeddings=None,
    ordered_item_ids=None,
    device=None
):
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_scores = []

    exact_count = 0
    fallback_count = 0
    fail_count = 0

    for ex in data:
        item_id, match_type = resolve_name_to_item_id(
            ex.get("output"),
            name2id=name2id,
            sbert_model=sbert_model,
            item_embeddings=item_embeddings,
            ordered_item_ids=ordered_item_ids,
            device=device
        )

        if item_id is None:
            fail_count += 1
            continue

        if match_type == "exact":
            exact_count += 1
        elif match_type == "fallback":
            fallback_count += 1

        freq = popularity_count.get(int(item_id), 0)
        gt_scores.append(math.log1p(freq))

    print(f"[INFO] {result_json_path}")
    print(f"       ground_truth exact    : {exact_count}")
    print(f"       ground_truth fallback : {fallback_count}")
    print(f"       ground_truth fail     : {fail_count}")

    return np.array(gt_scores, dtype=float)


def safe_kde(scores, x_grid):
    if len(scores) == 0:
        return None

    if len(np.unique(scores)) == 1:
        center = scores[0]
        y = np.exp(-0.5 * ((x_grid - center) / 0.05) ** 2)
        area = np.trapz(y, x_grid)
        if area > 0:
            y = y / area
        return y

    kde = gaussian_kde(scores)
    return kde(x_grid)


def plot_kde_multi(
    gt_scores,
    prediction_dict,
    output_path,
    title="Popularity Distribution Comparison",
    colors=None
):
    all_arrays = [gt_scores] + [arr for arr in prediction_dict.values() if len(arr) > 0]
    if not all_arrays or all(len(arr) == 0 for arr in all_arrays):
        raise ValueError("No valid scores to plot.")

    x_min = min(arr.min() for arr in all_arrays if len(arr) > 0)
    x_max = max(arr.max() for arr in all_arrays if len(arr) > 0)

    if x_min == x_max:
        x_min -= 1
        x_max += 1

    x = np.linspace(x_min, x_max, 400)

    plt.figure(figsize=(9, 6))

    y_gt = safe_kde(gt_scores, x)
    if y_gt is not None:
        plt.plot(x, y_gt, linewidth=2.5, linestyle="--", label="Ground Truth")

    for label, scores in prediction_dict.items():
        y = safe_kde(scores, x)
        if y is not None:
            plt.plot(x, y, linewidth=2, label=label, color=colors.pop(0) if colors else None)

    plt.xlabel("Log Popularity (log(1 + frequency))")
    plt.ylabel("Density of top-1 predicted items")
    plt.title(title)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved plot to {output_path}")


def main(
    train_json_path,
    result_json_paths,
    result_labels,
    colors,
    name2id_path,
    embeddings_path,
    sbert_model_path,
    output_path,
    id2name_path=None,
    device=None
):
    if len(result_json_paths) != len(result_labels):
        raise ValueError("result_json_paths and result_labels must have the same length.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(name2id_path, "r", encoding="utf-8") as f:
        name2id = json.load(f)

    if id2name_path is not None:
        with open(id2name_path, "r", encoding="utf-8") as f:
            raw_id2name = json.load(f)
        id2name = {int(k): v for k, v in raw_id2name.items()}
    else:
        id2name = {int(v): k for k, v in name2id.items()}

    # popularity from train input history
    popularity_count = build_popularity(train_json_path, name2id, id2name)

    # fallback resources (evaluate-style nearest top1)
    item_embeddings, sbert_model, ordered_item_ids = build_item_embedding_lookup(
        embeddings_path=embeddings_path,
        id2name=id2name,
        sbert_model_path=sbert_model_path,
        device=device
    )

    # ground truth from first result file
    gt_scores = extract_ground_truth_scores(
        result_json_paths[0],
        name2id=name2id,
        popularity_count=popularity_count,
        sbert_model=sbert_model,
        item_embeddings=item_embeddings,
        ordered_item_ids=ordered_item_ids,
        device=device
    )

    prediction_dict = {}
    for path, label in zip(result_json_paths, result_labels):
        prediction_dict[label] = extract_prediction_scores(
            path,
            name2id=name2id,
            popularity_count=popularity_count,
            sbert_model=sbert_model,
            item_embeddings=item_embeddings,
            ordered_item_ids=ordered_item_ids,
            device=device
        )

    plot_kde_multi(
        gt_scores=gt_scores,
        prediction_dict=prediction_dict,
        output_path=output_path,
        title="Popularity Distribution Comparison",
        colors=colors
    )


if __name__ == "__main__":
    

    lr = "1e-6"
    training_method = "multi_early_stop"
    category = "Goodreads"  # "Goodreads", "Movies", "Music"
    
    name2id_path=f"./eval/{category}/name2id.json"
    id2name_path = f"./eval/{category}/id2name.json"   # optional but recommended
    embeddings_path = f"./eval/{category}/embeddings.pt"
    sbert_model_path = "sentence-transformers/paraphrase-MiniLM-L3-v2"

    seed = 1

    base_dir = f"./centered_percentile_experiments_4096_1000/{category}"
    train_json_path=f"{base_dir}/sampled_data/seed{seed}/train.json"
    output_path = f"{base_dir}/seed{seed}_{category}_distribution_lr_{lr}.png"
    
    result_json_paths = [
        f"{base_dir}/predicts/seed{seed}/SFT/predicts.json",
        f"{base_dir}/predicts/seed{seed}/sequence_logprob_margin/p0/final/predicts.json",
        f"{base_dir}/predicts/seed{seed}/sequence_logprob_margin/p50/final/predicts.json",
        f"{base_dir}/predicts/seed{seed}/sequence_logprob_margin/p100/final/predicts.json",

        f"{base_dir}/predicts/seed{seed}/ln_ches_score/p0/final/predicts.json",
        f"{base_dir}/predicts/seed{seed}/ln_ches_score/p50/final/predicts.json",
        f"{base_dir}/predicts/seed{seed}/ln_ches_score/p100/final/predicts.json",
    ]

    # result_json_paths = [
    #     f"./experiments/predicts/SFT/{category}/predicts.json",
    #     # f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_1e-5/it0/Min_ln_ches_scores/predicts.json",
    #     f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/DPO_RN1/predicts.json",
    #     f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/Min_ln_ches_scores/predicts.json",
    #     f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/Max_ln_ches_scores/predicts.json",
    #     f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/Min_last_hidden_embedding_inner_prods/predicts.json",

    #     # f"./experiments_new/predicts/old/{category}_sftlr_0.00002_lr_{lr}/it0/Min_ln_ches_scores/predicts.json",
    #     # f"./experiments/predicts/{category}_{lr}/it0/Min_ln_ches_scores/predicts.json",
    #     # f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/DPO_RN1/predicts.json",
    #     # f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/Min_ln_ches_scores/predicts.json",
        
        
    # ]

    # result_json_paths = [
    #     "./re_run_experiments/predictions/clusterout_low_train_full/raw_results_1024.json",
    #     "./re_run_experiments/predictions/SPRec_train_full/raw_results_1024.json",
    #     "./re_run_experiments/predictions/RN1_train/raw_results_1024.json",
    #     "./re_run_experiments/predictions/sft_model_train_full/raw_results_1024.json",
    #     # "./re_run_experiments/predictions/RN1/raw_results_1000.json",
    #     # "./re_run_experiments/predictions/raw_model/raw_results_1000.json",

    # ]
    # result_labels = [
    #     "SFT(Before DPO) prediction",
    #     # f"Min_ln_ches_scores lr=1e-5 prediction",
    #     f"DPO_RN1 lr={lr} prediction",
    #     f"Min_ln_ches_scores lr={lr} prediction",
    #     f"Max_ln_ches_scores lr={lr} prediction",
    #     f"Min_last_hidden_embedding_inner_prods lr={lr} prediction",
    #     # f"Min_ln_ches_scores lr={lr} prediction (without 1st SFT)",
    #     # f"Min_ln_ches_scores lr={lr} prediction (without 2nd SFT)",
    #     # f"Min_ln_ches_scores lr={lr} prediction",
    # ]

    result_labels = [
        "SFT (original model) prediction",
        "Sequence LogProb Margin p0",
        "Sequence LogProb Margin p50",
        "Sequence LogProb Margin p100",
        "ln CHES Score p0",
        "ln CHES Score p50",
        "ln CHES Score p100",
    ]


    colors = [
        "#1f77b4",  # blue
        "#d62728",  # red

        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#8c564b",  # brown
         "#e377c2",  # pink
    ]
    main(
        train_json_path=train_json_path,
        result_json_paths=result_json_paths,
        result_labels=result_labels,
        colors=colors,
        name2id_path=name2id_path,
        id2name_path=id2name_path,
        embeddings_path=embeddings_path,
        sbert_model_path=sbert_model_path,
        output_path=output_path,
        device=None,   # auto detect cuda/cpu
    )