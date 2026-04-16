import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


PERCENTILE_ORDER = ["p0", "p25", "p50", "p75", "p100"]
PERCENTILE_TO_X = {"p0": 0, "p25": 25, "p50": 50, "p75": 75, "p100": 100}

CORE_FIELDS = [
    "NDCG",
    "HR",
    "diversity",
    "DivRatio",
    "ORRatio",
    "NDCG_head@5",
    "HR_head@5",
    "NDCG_tail@5",
    "HR_tail@5",
    "Count_Head",
    "Count_Tail",
    "GiniIndex",
    "num_recommended_unique",
    "total_recommendations",
    "num_total_items",
    "coverage",
]

DEFAULT_BAR_FIELDS = [
    "NDCG",
    "HR",
    "DivRatio",
    "ORRatio",
    "coverage",
    "GiniIndex",
]

HEAD_TAIL_FIELDS = [
    "NDCG_head@5",
    "HR_head@5",
    "NDCG_tail@5",
    "HR_tail@5",
]


def load_eval_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_scalar(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 0:
            return None
        return value[0]
    return value


def parse_eval_top5(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for record in eval_records:
        for key, value in record.items():
            if key == "model" or key == "category" or key == "topk":
                merged[key] = value
            elif key in CORE_FIELDS:
                merged[key] = extract_scalar(value)
            else:
                merged[key] = extract_scalar(value)
    return merged


def method_name_from_metric_percentile(metric_name: str, percentile_name: str) -> str:
    if percentile_name == "p0":
        prefix = "Min"
    elif percentile_name == "p100":
        prefix = "Max"
    else:
        prefix = percentile_name.upper()
    return f"{prefix}_{metric_name}"


def find_metric_dirs(metrics_root: str, metric_filter: Optional[List[str]] = None) -> List[str]:
    metric_dirs = []
    for item in sorted(os.listdir(metrics_root)):
        full = os.path.join(metrics_root, item)
        if os.path.isdir(full):
            if metric_filter is None or item in metric_filter:
                metric_dirs.append(item)
    return metric_dirs


def collect_results(metrics_root: str, metric_filter: Optional[List[str]] = None, percentiles: Optional[List[str]] = None) -> pd.DataFrame:
    if percentiles is None:
        percentiles = ["p0", "p50", "p100"]

    rows: List[Dict[str, Any]] = []
    metric_dirs = find_metric_dirs(metrics_root, metric_filter)

    for metric_name in metric_dirs:
        for p in percentiles:
            for model in ["epoch1", "epoch2", "final"]:
           
                
                eval_path = os.path.join(metrics_root, metric_name, p, model , "eval_top5.json")
                if not os.path.exists(eval_path):
                    print(f"[Warning] Missing: {eval_path}")
                    continue

                eval_records = load_eval_json(eval_path)
                merged = parse_eval_top5(eval_records)
                row = {
                    "metric": metric_name,
                    "percentile": p,
                    "percentile_value": PERCENTILE_TO_X.get(p, None),
                    "Method": method_name_from_metric_percentile(metric_name, p),
                }
                row.update(merged)
                rows.append(row)

    if not rows:
        raise ValueError(f"No eval_top5.json results found under {metrics_root}")

    df = pd.DataFrame(rows)
    return df


def numeric_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def format_number(x: Any, decimals: int = 5) -> str:
    val = numeric_or_none(x)
    if val is None:
        return ""
    return f"{val:.{decimals}f}"


def save_markdown_table(df: pd.DataFrame, out_path: str, columns: List[str], sort_by: Optional[List[str]] = None) -> None:
    table_df = df.copy()
    if sort_by is not None:
        table_df = table_df.sort_values(sort_by)

    render_df = table_df[columns].copy()
    numeric_cols = [c for c in columns if c != "Method"]
    for c in numeric_cols:
        render_df[c] = render_df[c].apply(lambda x: format_number(x, decimals=5))

    md = render_df.to_markdown(index=False)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
        f.write("\n")
    print(f"[Saved] markdown table -> {out_path}")


def save_all_tables(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Full table
    full_cols = [
        "Method", "metric", "percentile", "NDCG", "HR", "DivRatio", "ORRatio",
        "NDCG_head@5", "HR_head@5", "NDCG_tail@5", "HR_tail@5", "coverage", "GiniIndex"
    ]
    save_markdown_table(
        df=df,
        out_path=os.path.join(output_dir, "all_results.md"),
        columns=full_cols,
        sort_by=["metric", "percentile_value"],
    )

    # Head/tail table like your example
    ht_cols = ["Method", "NDCG_head@5", "HR_head@5", "NDCG_tail@5", "HR_tail@5"]
    save_markdown_table(
        df=df,
        out_path=os.path.join(output_dir, "head_tail_results.md"),
        columns=ht_cols,
        sort_by=["metric", "percentile_value"],
    )

    # Per-metric head/tail tables
    for metric_name, subdf in df.groupby("metric"):
        safe_metric = metric_name.replace("/", "_")
        save_markdown_table(
            df=subdf,
            out_path=os.path.join(output_dir, f"{safe_metric}_head_tail.md"),
            columns=ht_cols,
            sort_by=["percentile_value"],
        )


def plot_bar_comparisons(df: pd.DataFrame, output_dir: str, bar_fields: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, subdf in df.groupby("metric"):
        subdf = subdf.copy().sort_values("percentile_value")
        labels = subdf["percentile"].tolist()

        n_fields = len(bar_fields)
        fig, axes = plt.subplots(n_fields, 1, figsize=(10, 4 * n_fields))
        if n_fields == 1:
            axes = [axes]

        for ax, field in zip(axes, bar_fields):
            values = [numeric_or_none(v) for v in subdf[field].tolist()]
            ax.bar(labels, values)
            ax.set_title(f"{metric_name}: {field} by percentile")
            ax.set_xlabel("Percentile")
            ax.set_ylabel(field)
            ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{metric_name}_bar_comparison.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] bar chart -> {out_path}")


def plot_line_chart(
    df: pd.DataFrame,
    output_dir: str,
    y_field: str,
    metric_filter: Optional[List[str]] = None,
    title: Optional[str] = None,
    model: str = "final",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plot_df = df.copy()

    if metric_filter is not None:
        plot_df = plot_df[plot_df["metric"].isin(metric_filter)].copy()

    metric_names = sorted(plot_df["metric"].unique().tolist())
    if len(metric_names) == 0:
        raise ValueError("No metrics available for line plot after filtering.")

    fig, ax = plt.subplots(figsize=(10, 6))

    for metric_name in metric_names:
        condition = (plot_df["metric"] == metric_name) & (plot_df["model"] == model)
        subdf = plot_df[condition].copy().sort_values("percentile_value")
        # subdf = plot_df[plot_df["metric"] == metric_name].copy().sort_values("percentile_value")
        
        x_vals = subdf["percentile_value"].tolist()
        y_vals = [numeric_or_none(v) for v in subdf[y_field].tolist()]
        ax.plot(x_vals, y_vals, marker="o", label=metric_name)

    ax.set_xlabel("Preference Similarity (Percentile)")
    ax.set_ylabel(y_field)
    ax.set_title(title if title else f"{y_field} by percentile")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    safe_field = re.sub(r"[^A-Za-z0-9_@]+", "_", y_field)
    out_path = os.path.join(output_dir, f"line_{safe_field}_{model}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] line chart -> {out_path}")


def save_csv(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)
    print(f"[Saved] csv -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize eval_top5.json results and make tables/plots.")
    parser.add_argument("--metrics_root", type=str, required=True, help="Root like ./.../Goodreads/metrics")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save tables and plots")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric directory names to include, e.g. sequence_logprob_margin ches_score",
    )
    parser.add_argument(
        "--percentiles",
        nargs="*",
        default=["p0", "p50", "p100"],
        help="Percentiles to include, default: p0 p50 p100",
    )
    parser.add_argument(
        "--line_field",
        type=str,
        default="HR",
        help="Field to use for line chart, e.g. HR, NDCG, HR_head@5, HR_tail@5",
    )
    parser.add_argument(
        "--bar_fields",
        nargs="*",
        default=DEFAULT_BAR_FIELDS,
        help="Fields to include in bar chart subplots",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="final",
        help="Model to use for line chart, e.g. epoch1, epoch2, final",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = collect_results(
        metrics_root=args.metrics_root,
        metric_filter=args.metrics,
        percentiles=args.percentiles,
    )

    df = df.sort_values(["metric", "percentile_value"]).reset_index(drop=True)
    save_csv(df, os.path.join(args.output_dir, "all_results.csv"))
    save_all_tables(df, args.output_dir)

    plots_dir = os.path.join(args.output_dir, "plots")
    bar_dir = os.path.join(plots_dir, "bar")
    line_dir = os.path.join(plots_dir, "line")
    os.makedirs(bar_dir, exist_ok=True)
    os.makedirs(line_dir, exist_ok=True)

    # plot_bar_comparisons(df, bar_dir, args.bar_fields)
    plot_line_chart(
        df=df,
        output_dir=line_dir,
        y_field=args.line_field,
        metric_filter=args.metrics,
        title=f"{args.line_field} by preference similarity percentile",
        model=args.model,
    )


if __name__ == "__main__":
    main()
