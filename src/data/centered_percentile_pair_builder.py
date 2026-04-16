import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def generate_prompt(instruction: str, input_text: str = None) -> str:
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""


def normalized_edit_distance(seq1: List[int], seq2: List[int]) -> float:
    len_sent2 = len(seq2)
    dold = list(range(len_sent2 + 1))
    dnew = [0 for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = i
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)
        dnew, dold = dold, dnew

    return float(dold[-1]) / max(len(seq1), len(seq2))


def load_data(data_path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data.extend(loaded)
            else:
                raise ValueError("Expected a list of data points in the JSON file.")
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    return data


def normalize_title_text(text: str) -> str:
    return text.strip().strip('"').strip()


def build_random_negative(chosen_text: str, all_titles: List[str], rng: random.Random) -> str:
    chosen_title = normalize_title_text(chosen_text)
    candidate_pool = [title for title in all_titles if normalize_title_text(title) != chosen_title]
    if not candidate_pool:
        raise ValueError("No valid random negative candidates available after excluding chosen title.")
    random_item = rng.choice(candidate_pool)
    return f'"{random_item}"\n'


def get_response_token_spans(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
) -> Dict[str, Any]:
    query_ids = tokenizer(prompt_text, padding=False, truncation=False, add_special_tokens=False).input_ids
    chosen_full_ids = tokenizer(prompt_text + chosen_text, padding=False, truncation=False, add_special_tokens=False).input_ids
    rejected_full_ids = tokenizer(prompt_text + rejected_text, padding=False, truncation=False, add_special_tokens=False).input_ids

    query_len = len(query_ids)
    chosen_ids = chosen_full_ids[query_len:]
    rejected_ids = rejected_full_ids[query_len:]

    return {
        "query_ids": query_ids,
        "query_len": query_len,
        "chosen_full_ids": chosen_full_ids,
        "rejected_full_ids": rejected_full_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }


def compute_sequence_logprob(
    model: AutoModelForCausalLM,
    input_ids: List[int],
    query_len: int,
    device: str,
) -> Tuple[float, float]:
    ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=ids_tensor)

    logits = outputs.logits[:, :-1, :]
    labels = ids_tensor[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    response_start = max(query_len - 1, 0)
    response_token_log_probs = token_log_probs[:, response_start:]

    if response_token_log_probs.numel() == 0:
        return 0.0, 0.0

    seq_logprob = float(response_token_log_probs.sum().detach().cpu())
    avg_token_logprob = float(response_token_log_probs.mean().detach().cpu())
    return seq_logprob, avg_token_logprob


def compute_hidden_metrics(
    model: AutoModelForCausalLM,
    chosen_full_ids: List[int],
    rejected_full_ids: List[int],
    query_len: int,
    device: str,
) -> Tuple[float, float, float]:
    chosen_tensor = torch.tensor(chosen_full_ids, dtype=torch.long, device=device).unsqueeze(0)
    rejected_tensor = torch.tensor(rejected_full_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        chosen_out = model(input_ids=chosen_tensor, output_hidden_states=True)
        rejected_out = model(input_ids=rejected_tensor, output_hidden_states=True)

    hidden_w = chosen_out.hidden_states[-1][0]
    hidden_l = rejected_out.hidden_states[-1][0]

    preferred_hidden_embed = hidden_w[query_len - 1:]
    dispreferred_hidden_embed = hidden_l[query_len - 1:]

    if preferred_hidden_embed.shape[0] == 0 or dispreferred_hidden_embed.shape[0] == 0:
        raise ValueError("Empty preferred/dispreferred hidden embeddings after slicing.")

    preferred_hidden_embed = preferred_hidden_embed.to(torch.float32)
    dispreferred_hidden_embed = dispreferred_hidden_embed.to(torch.float32)

    s_w = preferred_hidden_embed.sum(dim=0)
    s_l = dispreferred_hidden_embed.sum(dim=0)
    t_w = preferred_hidden_embed.shape[0]
    t_l = dispreferred_hidden_embed.shape[0]

    ches = (s_w * s_l).sum() - torch.norm(s_w) ** 2
    pref_dispref = (s_w * s_l).sum() / (t_w * t_l)
    pref_only = torch.norm(s_w) ** 2 / (t_w ** 2)
    ln_ches = pref_dispref - pref_only
    last_inner = torch.inner(preferred_hidden_embed[-1], dispreferred_hidden_embed[-1])

    return (
        float(ches.detach().cpu()),
        float(ln_ches.detach().cpu()),
        float(last_inner.detach().cpu()),
    )


def build_scored_pair_pool(
    data: List[Dict[str, Any]],
    all_titles: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    scored_pairs: List[Dict[str, Any]] = []

    for sample_id, example in tqdm(list(enumerate(data)), desc="Scoring RN pairs"):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        chosen = example["output"]
        prompt_text = generate_prompt(instruction, input_text)
        prompt_field = instruction + input_text

        try:
            rejected = build_random_negative(chosen, all_titles, rng)
        except ValueError:
            continue

        spans = get_response_token_spans(tokenizer, prompt_text, chosen, rejected)
        chosen_len = len(spans["chosen_ids"])
        rejected_len = len(spans["rejected_ids"])

        if chosen_len == 0 or rejected_len == 0:
            continue

        try:
            chosen_seq_logprob, chosen_avg_logprob = compute_sequence_logprob(
                model=model,
                input_ids=spans["chosen_full_ids"],
                query_len=spans["query_len"],
                device=device,
            )
            rejected_seq_logprob, rejected_avg_logprob = compute_sequence_logprob(
                model=model,
                input_ids=spans["rejected_full_ids"],
                query_len=spans["query_len"],
                device=device,
            )
            ches_score, ln_ches_score, last_hidden_inner = compute_hidden_metrics(
                model=model,
                chosen_full_ids=spans["chosen_full_ids"],
                rejected_full_ids=spans["rejected_full_ids"],
                query_len=spans["query_len"],
                device=device,
            )
        except Exception as e:
            print(f"[Warning] sample_id={sample_id} skipped due to metric computation error: {e}")
            continue

        norm_ed = normalized_edit_distance(spans["chosen_ids"], spans["rejected_ids"])
        seq_margin = chosen_seq_logprob - rejected_seq_logprob
        avg_margin = chosen_avg_logprob - rejected_avg_logprob

        scored_pairs.append(
            {
                "sample_id": sample_id,
                "instruction": instruction,
                "input": input_text,
                "prompt": prompt_field,
                "chosen": chosen,
                "rejected": rejected,
                "ches_score": ches_score,
                "ln_ches_score": ln_ches_score,
                "last_hidden_embedding_inner_prod": last_hidden_inner,
                "sequence_logprob_margin": seq_margin,
                "avg_token_logprob_margin": avg_margin,
                "chosen_seq_logprob": chosen_seq_logprob,
                "rejected_seq_logprob": rejected_seq_logprob,
                "chosen_avg_token_logprob": chosen_avg_logprob,
                "rejected_avg_token_logprob": rejected_avg_logprob,
                "chosen_len": chosen_len,
                "rejected_len": rejected_len,
                "normalized_edit_distance": norm_ed,
            }
        )

    return scored_pairs


def get_centered_window_indices(n: int, percentile: int, window_size: int) -> Tuple[int, int]:
    if window_size > n:
        raise ValueError(f"window_size={window_size} is larger than dataset size n={n}")

    if percentile == 0:
        return 0, window_size
    if percentile == 100:
        return n - window_size, n

    center = int(round((percentile / 100.0) * (n - 1)))
    half = window_size // 2
    start = center - half
    end = start + window_size

    if start < 0:
        start = 0
        end = window_size
    if end > n:
        end = n
        start = n - window_size

    return start, end


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_centered_percentile_datasets(
    scored_pairs: List[Dict[str, Any]],
    output_dir: str,
    window_size: int,
    percentiles: List[int],
) -> None:
    metrics = [
        "ches_score",
        "ln_ches_score",
        "last_hidden_embedding_inner_prod",
        "sequence_logprob_margin",
        "avg_token_logprob_margin",
    ]

    datasets_dir = os.path.join(output_dir, "centered_percentile_datasets")
    metadata_dir = os.path.join(output_dir, "window_metadata")
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    n = len(scored_pairs)

    for metric in metrics:
        sorted_pairs = sorted(scored_pairs, key=lambda x: x[metric])

        for percentile in percentiles:
            start, end = get_centered_window_indices(n=n, percentile=percentile, window_size=window_size)
            window_pairs = sorted_pairs[start:end]

            dataset_records = [
                {
                    "prompt": pair["prompt"],
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                }
                for pair in window_pairs
            ]

            metadata_records = window_pairs

            base_name = f"{metric}_p{percentile}_w{window_size}"
            dataset_path = os.path.join(datasets_dir, f"{base_name}.jsonl")
            metadata_path = os.path.join(metadata_dir, f"{base_name}_metadata.json")

            save_jsonl(dataset_path, dataset_records)
            save_json(metadata_path, metadata_records)

            print(
                f"[Saved] metric={metric}, percentile={percentile}, "
                f"window=({start}, {end}), size={len(window_pairs)}"
            )


def main():
    parser = argparse.ArgumentParser(description="Build fixed RN pair pool, score pairs, and create centered percentile DPO datasets.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to sampled data (expected size ~4096)")
    parser.add_argument("--id2name_path", type=str, required=True, help="Path to id2name JSON file")
    parser.add_argument("--model_path", type=str, required=True, help="Base or merged model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for RN sampling")
    parser.add_argument("--window_size", type=int, default=1024, help="Centered percentile window size")
    parser.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=[0, 25, 50, 75, 100],
        help="Percentile centers to use for window construction",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.data_path}")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} samples")

    print(f"Loading id2name from {args.id2name_path}")
    with open(args.id2name_path, "r", encoding="utf-8") as f:
        id2name = json.load(f)
    all_titles = list(id2name.values())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer/model from {args.model_path} to {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if args.lora_path:
        print(f"Loading LoRA adapter from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()

    scored_pairs = build_scored_pair_pool(
        data=data,
        all_titles=all_titles,
        tokenizer=tokenizer,
        model=model,
        device=device,
        seed=args.random_seed,
    )

    print(f"Built scored pair pool with {len(scored_pairs)} valid pairs")
    if len(scored_pairs) == 0:
        raise ValueError("No valid scored pairs were produced.")
    if args.window_size > len(scored_pairs):
        raise ValueError(
            f"window_size={args.window_size} is larger than number of valid scored pairs={len(scored_pairs)}"
        )

    full_metadata_path = os.path.join(args.output_dir, "full_scored_pair_metadata.json")
    save_json(full_metadata_path, scored_pairs)
    print(f"Saved full scored metadata to {full_metadata_path}")

    build_centered_percentile_datasets(
        scored_pairs=scored_pairs,
        output_dir=args.output_dir,
        window_size=args.window_size,
        percentiles=args.percentiles,
    )

    config_path = os.path.join(args.output_dir, "run_config.json")
    save_json(
        config_path,
        {
            "data_path": args.data_path,
            "id2name_path": args.id2name_path,
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "random_seed": args.random_seed,
            "window_size": args.window_size,
            "percentiles": args.percentiles,
            "num_input_samples": len(data),
            "num_valid_scored_pairs": len(scored_pairs),
        },
    )
    print(f"Saved run config to {config_path}")


if __name__ == "__main__":
    main()
