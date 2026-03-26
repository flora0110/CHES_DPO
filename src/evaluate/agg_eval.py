#!/usr/bin/env python3
# aggregate_eval_top5.py

import json
import re
import sys
from pathlib import Path
from statistics import mean

IT_RE = re.compile(r"(it\d+)")

def _to_float(x):
    """Handle scalar or [scalar] forms."""
    if isinstance(x, list) and len(x) > 0:
        return float(x[0])
    return float(x)

def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# Head/Tail @5 (eval_top5.json)
# -------------------------
def read_head_tail_metrics(eval_top5_path: Path):
    """
    eval_top5.json is typically a JSON list with multiple dicts.
    We pick the one that contains head/tail keys.
    """
    data = _load_json(eval_top5_path)

    if not isinstance(data, list):
        raise ValueError(f"{eval_top5_path} is not a JSON list")

    for obj in data:
        if isinstance(obj, dict) and ("NDCG_head@5" in obj or "HR_head@5" in obj):
            return {
                "NDCG_head@5": _to_float(obj["NDCG_head@5"]),
                "HR_head@5": _to_float(obj["HR_head@5"]),
                "NDCG_tail@5": _to_float(obj["NDCG_tail@5"]),
                "HR_tail@5": _to_float(obj["HR_tail@5"]),
            }

    raise KeyError(f"No head/tail @5 metrics found in {eval_top5_path}")

def aggregate_head_tail(output_dir: Path, metric: str):
    """
    Scan:
      output_dir/it*/<metric>/eval_top5.json
    """
    # print("Aggregating head/tail @5 metrics...")
    # print(output_dir)
    rows = []
    for it_dir in sorted(output_dir.glob("it*")):
        if not it_dir.is_dir():
            continue
        it_match = IT_RE.search(it_dir.name)
        if not it_match:
            continue
        it = it_match.group(1)
        
        eval_path = it_dir / metric / "eval_top5.json"
        # print(f"Processing {eval_path}...")
        if not eval_path.exists():
            print(f"[WARN] missing: {eval_path}", file=sys.stderr)
            continue

        try:
            m = read_head_tail_metrics(eval_path)
        except Exception as e:
            print(f"[WARN] failed reading {eval_path}: {e}", file=sys.stderr)
            continue

        rows.append((it, m["NDCG_head@5"], m["HR_head@5"], m["NDCG_tail@5"], m["HR_tail@5"]))

    rows.sort(key=lambda r: int(r[0].replace("it", "")))
    return rows

def print_markdown_head_tail(rows):
    print("## Head/Tail @5\n")
    header = "| it | NDCG_head@5 | HR_head@5 | NDCG_tail@5 | HR_tail@5 |"
    sep    = "| ----------- | ----------: | ----------: | ----------: | ----------: |"
    print(header)
    print(sep)

    for it, ndcg_h, hr_h, ndcg_t, hr_t in rows:
        print(f"| {it} | {ndcg_h:.5f} | {hr_h:.5f} | {ndcg_t:.5f} | {hr_t:.5f} |")

    if rows:
        avg_ndcg_h = mean(r[1] for r in rows)
        avg_hr_h   = mean(r[2] for r in rows)
        avg_ndcg_t = mean(r[3] for r in rows)
        avg_hr_t   = mean(r[4] for r in rows)
        print(f"| **Average** | **{avg_ndcg_h:.5f}** | **{avg_hr_h:.5f}** | **{avg_ndcg_t:.5f}** | **{avg_hr_t:.5f}** |")
    else:
        print("| **Average** | **nan** | **nan** | **nan** | **nan** |")

# -------------------------
# Overall metrics table (test_result.json)
# -------------------------
def read_overall_metrics(test_result_path: Path):
    """
    test_result.json is expected to be a JSON list with a dict containing:
      DivRatio, ORRatio, MGU, HR (list), NDCG (list)
    We'll take the first dict that has DivRatio/ORRatio/MGU.
    """
    data = _load_json(test_result_path)

    if not isinstance(data, list):
        raise ValueError(f"{test_result_path} is not a JSON list")

    for obj in data:
        if isinstance(obj, dict) and ("MGU" in obj and "DGU" in obj):
            return {
                "DivRatio": float(obj["DivRatio"]),
                "ORRatio": float(obj["ORRatio"]),
                "MGU": float(obj["MGU"]),
                "DGU": float(obj["DGU"]),
                "HR": _to_float(obj["HR"]),
                "NDCG": _to_float(obj["NDCG"]),
            }
        elif isinstance(obj, dict):
            return {
                "DivRatio": float(obj["DivRatio"]),
                "ORRatio": float(obj["ORRatio"]),
                # "MGU": float(obj["MGU"]),
                # "DGU": float(obj["DGU"]),
                "HR": _to_float(obj["HR"]),
                "NDCG": _to_float(obj["NDCG"]),
            }

    raise KeyError(f"No overall metrics found in {test_result_path}")

def aggregate_overall(output_dir: Path, metric: str):
    """
    Scan:
      output_dir/it*/<metric>/test_result.json
    """
    rows = []
    for it_dir in sorted(output_dir.glob("it*")):
        if not it_dir.is_dir():
            continue
        it_match = IT_RE.search(it_dir.name)
        if not it_match:
            continue
        it = it_match.group(1)

        test_path = it_dir / metric / "eval_top5.json"
        if not test_path.exists():
            print(f"[WARN] missing: {test_path}", file=sys.stderr)
            continue

        try:
            m = read_overall_metrics(test_path)
        except Exception as e:
            print(f"[WARN] failed reading {test_path}: {e}", file=sys.stderr)
            continue
        
        if m.get("MGU") is None or m.get("DGU") is None:
            rows.append((it, m["DivRatio"], m["ORRatio"], 0.0, 0.0, m["HR"], m["NDCG"]))
        else:
            rows.append((it, m["DivRatio"], m["ORRatio"], m["MGU"],  m["DGU"], m["HR"], m["NDCG"]))
        # rows.append((it, m["DivRatio"], m["ORRatio"], 0.0, m["HR"], m["NDCG"]))

    rows.sort(key=lambda r: int(r[0].replace("it", "")))
    return rows

def print_markdown_overall(rows):
    print("\n\n## Overall metrics\n")
    header = "| it | DivRatio ↑ | ORRatio ↓ | MGU ↓ | DGU ↓ | HR ↑ | NDCG ↑ |"
    sep    = "| ----------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |"
    print(header)
    print(sep)

    for it, divr, orr, mgu, dgu, hr, ndcg in rows:
        # 你要 Model 欄只留 it0/it1/...，所以直接印 it
        print(f"| {it} | {divr:.5f} | {orr:.5f} | {mgu:.5f} | {dgu:.5f} | {hr:.5f} | {ndcg:.5f} |")

    if rows:
        avg_divr = mean(r[1] for r in rows)
        avg_orr  = mean(r[2] for r in rows)
        avg_mgu  = mean(r[3] for r in rows)
        avg_dgu  = mean(r[4] for r in rows)
        avg_hr   = mean(r[5] for r in rows)
        avg_ndcg = mean(r[6] for r in rows)
        print(f"| **Average** | **{avg_divr:.5f}** | **{avg_orr:.5f}** | **{avg_mgu:.5f}** | **{avg_dgu:.5f}** | **{avg_hr:.5f}** | **{avg_ndcg:.5f}** |")
    else:
        print("| **Average** | **nan** | **nan** | **nan** | **nan** | **nan** |")

# -------------------------
# main
# -------------------------
def main():
    # print("Starting aggregation...")
    """
    Usage:
      python aggregate_eval_top5.py <output_dir> <metric>

    Example:
      python aggregate_eval_top5.py ./models/SPRec/Goodreads_2048_0.00002 nearest_ln_ches_scores
    """
    if len(sys.argv) < 3:
        print("Usage: python aggregate_eval_top5.py <output_dir> <metric>", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(sys.argv[1]).expanduser().resolve()
    metric = sys.argv[2]
    # print(f"Aggregating results from {output_dir} for metric: {metric}")

    head_tail_rows = aggregate_head_tail(output_dir, metric)
    overall_rows = aggregate_overall(output_dir, metric)

    # Write both tables to the same stdout (so same md file when redirected)
    # print(f"# Aggregated results for metric: {metric}\n")
    print_markdown_head_tail(head_tail_rows)
    print_markdown_overall(overall_rows)

if __name__ == "__main__":
    main()
