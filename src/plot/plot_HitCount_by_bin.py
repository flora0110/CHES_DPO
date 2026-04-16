import json
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer


def clean_item_name(text):
    if text is None:
        return None
    text = str(text).strip()
    text = text.replace("</s>", "").strip()

    match = re.search(r'"([^"]+)"', text)
    if match:
        return match.group(1).strip()

    text = text.split("\n", 1)[0].strip()

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    return text.strip()


def build_popularity(train_json_path, name2id, id2name):
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
        output_text = ex.get("output", "")
        names = re.findall(r'"([^"]+)"', output_text)

        unique_ids = set()
        for name in names:
            if name in name2id:
                unique_ids.add(int(name2id[name]))

        for iid in unique_ids:
            if iid in popularity_count:
                popularity_count[iid] += 1

    return popularity_count


def nearest_item_id_by_text(text, sbert_model, item_embeddings, ordered_item_ids, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if text is None or len(text.strip()) == 0:
        return None

    vec = sbert_model.encode([text], convert_to_numpy=True)
    vec = torch.tensor(vec, dtype=torch.float32, device=device)

    dist = torch.cdist(vec, item_embeddings, p=2)
    nearest_idx = torch.argmin(dist, dim=1).item()

    if nearest_idx < 0 or nearest_idx >= len(ordered_item_ids):
        return None

    return ordered_item_ids[nearest_idx]


def build_item_embedding_lookup(
    embeddings_path,
    id2name,
    sbert_model_path,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_embeddings = torch.load(embeddings_path)
    embeddings_tensor = torch.tensor(raw_embeddings, dtype=torch.float32, device=device)

    ordered_item_ids = list(range(embeddings_tensor.size(0)))
    sbert_model = SentenceTransformer(sbert_model_path, device=device)

    return embeddings_tensor, sbert_model, ordered_item_ids


def resolve_name_to_item_id(
    raw_text,
    name2id,
    sbert_model=None,
    item_embeddings=None,
    ordered_item_ids=None,
    device=None
):
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


def get_topk_item_ids(text, sbert_model, item_embeddings, ordered_item_ids, k=5, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not text or len(str(text).strip()) == 0:
        return []

    cleaned = clean_item_name(text)
    vec = sbert_model.encode([cleaned], convert_to_numpy=True)
    vec = torch.tensor(vec, dtype=torch.float32, device=device)

    dist = torch.cdist(vec, item_embeddings, p=2)
    topk_indices = torch.topk(dist, k=k, largest=False).indices[0].tolist()

    return [ordered_item_ids[idx] for idx in topk_indices]


def extract_hit_gt_popularity_scores(
    result_json_path,
    name2id,
    popularity_count,
    sbert_model,
    item_embeddings,
    ordered_item_ids,
    k=5,
    device=None
):
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hit_scores = []
    total_count = len(data)
    hit_count = 0

    for ex in data:
        gt_raw = ex.get("output")
        if gt_raw is None:
            print(f"[WARN] Missing 'output' field in example: {ex}")
        gt_id, _ = resolve_name_to_item_id(
            gt_raw, name2id, sbert_model, item_embeddings, ordered_item_ids, device
        )

        if gt_id is None:
            continue

        pred_raw = ex.get("predict")
        if pred_raw is None:
            print(f"[WARN] Missing 'predict' field in example: {ex}")
            continue
        pred_topk_ids = get_topk_item_ids(
            pred_raw, sbert_model, item_embeddings, ordered_item_ids, k=k, device=device
        )

        if int(gt_id) in [int(pid) for pid in pred_topk_ids]:
            hit_count += 1
            freq = popularity_count.get(int(gt_id), 0)
            hit_scores.append(math.log1p(freq))

    print(f"[INFO] {result_json_path}")
    print(f"       Total samples: {total_count}, Hit@{k}: {hit_count} ({hit_count/total_count:.2%})")

    return np.array(hit_scores, dtype=float)


def extract_all_ground_truth_popularity_scores(
    result_json_path,
    name2id,
    popularity_count,
    sbert_model,
    item_embeddings,
    ordered_item_ids,
    device=None
):
    """
    讀取一個 result json，收集所有 ground_truth item 的 popularity score
    用它來建立 equal-frequency bins
    """
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_scores = []
    num_total = len(data)
    num_resolved = 0
    num_failed = 0

    for ex in data:
        gt_raw = ex.get("output")
        gt_id, match_type = resolve_name_to_item_id(
            gt_raw,
            name2id=name2id,
            sbert_model=sbert_model,
            item_embeddings=item_embeddings,
            ordered_item_ids=ordered_item_ids,
            device=device
        )

        if gt_id is None:
            num_failed += 1
            continue

        num_resolved += 1
        freq = popularity_count.get(int(gt_id), 0)
        gt_scores.append(math.log1p(freq))

    print(f"[INFO] Build GT bins from: {result_json_path}")
    print(f"       Total GT samples: {num_total}, resolved: {num_resolved}, failed: {num_failed}")

    return np.array(gt_scores, dtype=float)


def build_equal_frequency_bins_from_ground_truth_scores(gt_scores, num_bins=10):
    """
    根據 ground_truth popularity scores 切成等數量桶
    """
    if len(gt_scores) == 0:
        raise ValueError("gt_scores is empty, cannot build bins.")

    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(gt_scores, quantiles)

    # 避免重複 edge
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-8

    return bin_edges


def count_scores_in_bins(scores, bin_edges):
    counts, _ = np.histogram(scores, bins=bin_edges)
    return counts


def make_bin_labels(bin_edges):
    num_bins = len(bin_edges) - 1
    step = 100 // num_bins
    labels = []
    for i in range(num_bins):
        start = i * step
        end = (i + 1) * step
        labels.append(f"{start}-{end}%")
    return labels


def plot_equal_frequency_hit_counts(
    prediction_dict,
    bin_edges,
    output_path,
    title="Hit Count by GT Popularity Bins (Equal-Frequency 10 Bins)",
    colors=None
):
    """
    x 軸：由 ground_truth 決定的 equal-frequency bins
    y 軸：每個 bin 的 hit count
    """
    bin_labels = make_bin_labels(bin_edges)

    model_names = list(prediction_dict.keys())
    num_models = len(model_names)
    x = np.arange(len(bin_labels))
    width = 0.8 / max(num_models, 1)

    plt.figure(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        scores = prediction_dict[model_name]
        counts = count_scores_in_bins(scores, bin_edges)

        offset = (i - (num_models - 1) / 2) * width
        plt.bar(
            x + offset,
            counts,
            width=width,
            label=model_name,
            color=colors[i] if colors and i < len(colors) else None
        )

    plt.xticks(x, bin_labels)
    plt.xlabel("Ground-Truth Popularity Bins")
    plt.ylabel("Hit Count")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved plot to {output_path}")
    print("[INFO] Bin edges from ground_truth (log(1+freq)):")
    for i in range(len(bin_edges) - 1):
        print(f"  B{i+1}: [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f})")


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
    device=None,
    num_bins=10
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

    popularity_count = build_popularity(train_json_path, name2id, id2name)

    item_embeddings, sbert_model, ordered_item_ids = build_item_embedding_lookup(
        embeddings_path=embeddings_path,
        id2name=id2name,
        sbert_model_path=sbert_model_path,
        device=device
    )

    # 1. 用第一個 result file 的 ground_truth 建 bin
    #    前提是所有 model 共用同一批 samples；通常是這樣
    gt_scores_for_bins = extract_all_ground_truth_popularity_scores(
        result_json_path=result_json_paths[0],
        name2id=name2id,
        popularity_count=popularity_count,
        sbert_model=sbert_model,
        item_embeddings=item_embeddings,
        ordered_item_ids=ordered_item_ids,
        device=device
    )
    bin_edges = build_equal_frequency_bins_from_ground_truth_scores(
        gt_scores_for_bins,
        num_bins=num_bins
    )

    # 2. 統計每個 model 的 hits 落在哪些 GT bins
    prediction_dict = {}
    for path, label in zip(result_json_paths, result_labels):
        prediction_dict[label] = extract_hit_gt_popularity_scores(
            path,
            name2id=name2id,
            popularity_count=popularity_count,
            sbert_model=sbert_model,
            item_embeddings=item_embeddings,
            ordered_item_ids=ordered_item_ids,
            k=5,
            device=device
        )

    # 3. 畫圖
    plot_equal_frequency_hit_counts(
        prediction_dict=prediction_dict,
        bin_edges=bin_edges,
        output_path=output_path,
        colors=colors,
        title="Hit Count by Ground-Truth Exposure Quantile (5 Bins)"
    )


if __name__ == "__main__":
    lr = "1e-6"
    training_method = "multi_early_stop"
    category = "Goodreads"  # "Goodreads", "Movies", "Music"
    # train_json_path=f"./sampled_data/{category}_train.json"
    name2id_path=f"./eval/{category}/name2id.json"
    id2name_path = f"./eval/{category}/id2name.json"   # optional but recommended
    embeddings_path = f"./eval/{category}/embeddings.pt"
    sbert_model_path = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    seed = 0
    metric_for_sampling = "sequence_logprob_margin"
    which_epoch = "epoch1"
    base_dir = f"./centered_percentile_experiments_4096_1000/{category}"
    train_json_path=f"{base_dir}/sampled_data/seed{seed}/train.json"
    output_path = f"{base_dir}/seed{seed}_{category}_{metric_for_sampling}_{which_epoch}_HitCount_bin_lr_{lr}.png"


    # result_json_paths = [
    #     f"./experiments/predicts/SFT/{category}/predicts.json",
    #     # f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_1e-5/it0/Min_ln_ches_scores/predicts.json",
    #     f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/DPO_RN1/predicts.json",
    #     f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/Min_ln_ches_scores/predicts.json",
    #     f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/Max_ln_ches_scores/predicts.json",
    #     f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/Min_last_hidden_embedding_inner_prods/predicts.json",
    #     f"./experiments_new/predicts/{category}_0.00002_single_epoch/it0/DPO_RN1/predicts.json",
    #     f"./experiments_new/predicts/{category}_0.00002_single_epoch/it0/Min_ln_ches_scores/predicts.json",
    #     # f"./experiments_new/predicts/old/{category}_sftlr_0.00002_lr_{lr}/it0/Min_ln_ches_scores/predicts.json",
    #     # f"./experiments/predicts/{category}_{lr}/it0/Min_ln_ches_scores/predicts.json",
    #     # f"./experiments_new/predicts/{category}_sftlr_0.00002_lr_{lr}/it0/Min_ln_ches_scores/predicts.json",
        
        
    # ]

    result_json_paths = [
        f"{base_dir}/predicts/seed{seed}/SFT/predicts.json",
        f"{base_dir}/predicts/seed{seed}/{metric_for_sampling}/p0/{which_epoch}/predicts.json",
        f"{base_dir}/predicts/seed{seed}/{metric_for_sampling}/p25/{which_epoch}/predicts.json",
        f"{base_dir}/predicts/seed{seed}/{metric_for_sampling}/p50/{which_epoch}/predicts.json",
        f"{base_dir}/predicts/seed{seed}/{metric_for_sampling}/p75/{which_epoch}/predicts.json",
        f"{base_dir}/predicts/seed{seed}/{metric_for_sampling}/p100/{which_epoch}/predicts.json",
    ]
    
    # result_json_paths = [
    #     f"{base_dir}/predicts/seed{seed}/SFT/predicts.json",
    #     f"{base_dir}/predicts/seed{seed}/sequence_logprob_margin/p0/final/predicts.json",
    #     f"{base_dir}/predicts/seed{seed}/sequence_logprob_margin/p50/final/predicts.json",
    #     f"{base_dir}/predicts/seed{seed}/sequence_logprob_margin/p100/final/predicts.json",

    #     f"{base_dir}/predicts/seed{seed}/ln_ches_score/p0/final/predicts.json",
    #     f"{base_dir}/predicts/seed{seed}/ln_ches_score/p50/final/predicts.json",
    #     f"{base_dir}/predicts/seed{seed}/ln_ches_score/p100/final/predicts.json",
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
    #     f"DPO_RN1 single_epoch lr=0.00002 prediction",
    #     f"Min_ln_ches_scores single_epoch lr=0.00002 prediction",
    #     # f"Min_ln_ches_scores lr={lr} prediction (old)",
    #     # f"Min_ln_ches_scores lr={lr} prediction (without 2nd SFT)",
    #     # f"Min_ln_ches_scores lr={lr} prediction",
    # ]

    result_labels = [
        "SFT (original model) prediction",
        "Sequence LogProb Margin p0",
            "Sequence LogProb Margin p25",
        "Sequence LogProb Margin p50",
        "Sequence LogProb Margin p75",
        "Sequence LogProb Margin p100",
        # "ln CHES Score p0",
        # "ln CHES Score p50",
        # "ln CHES Score p100",
    ]

    colors = [
        "#1f77b4",  # blues
        "#d62728",  # red
        # "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#8c564b",  # brown
            # "#e377c2",  # pink
            # "#7f7f7f",  # gray
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
        device=None,
        num_bins=5
    )