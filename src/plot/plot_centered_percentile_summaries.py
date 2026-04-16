import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "ches_score",
    "ln_ches_score",
    "last_hidden_embedding_inner_prod",
    "sequence_logprob_margin",
    "avg_token_logprob_margin",
]

PERCENTILES = [0, 25, 50, 75, 100]

SUMMARY_FIELDS = [
    "chosen_len",
    "rejected_len",
    "normalized_edit_distance",
    "chosen_seq_logprob",
    "rejected_seq_logprob",
    "chosen_avg_token_logprob",
    "rejected_avg_token_logprob",
]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_or_nan(records: List[Dict], key: str) -> float:
    values = [r[key] for r in records if key in r and r[key] is not None]
    if not values:
        return float("nan")
    return float(np.mean(values))


def std_or_nan(records: List[Dict], key: str) -> float:
    values = [r[key] for r in records if key in r and r[key] is not None]
    if not values:
        return float("nan")
    return float(np.std(values))


def build_metric_summary(window_metadata_dir: str, metric: str, window_size: int) -> Dict[str, List[float]]:
    summary = {
        "percentiles": [],
        metric: [],
        f"{metric}_std": [],
    }

    for field in SUMMARY_FIELDS:
        summary[field] = []
        summary[f"{field}_std"] = []

    for p in PERCENTILES:
        path = os.path.join(window_metadata_dir, f"{metric}_p{p}_w{window_size}_metadata.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing metadata file: {path}")

        records = load_json(path)
        summary["percentiles"].append(str(p))
        summary[metric].append(mean_or_nan(records, metric))
        summary[f"{metric}_std"].append(std_or_nan(records, metric))

        for field in SUMMARY_FIELDS:
            summary[field].append(mean_or_nan(records, field))
            summary[f"{field}_std"].append(std_or_nan(records, field))

    return summary


def plot_metric_summary(output_plot_dir: str, metric: str, summary: Dict[str, List[float]]) -> None:
    x = np.arange(len(summary["percentiles"]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # errorbar style (point + vertical std line)
    ax.errorbar(
        x,
        summary[metric],
        yerr=summary[f"{metric}_std"],
        fmt='o',
        capsize=5
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["percentiles"])
    ax.set_xlabel("Percentile Window Center")
    ax.set_ylabel(metric)
    ax.set_title(f"Mean {metric} by centered percentile window")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    png_path = os.path.join(output_plot_dir, f"{metric}_errorbar.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_auxiliary_summary(output_plot_dir: str, metric: str, summary: Dict[str, List[float]]) -> None:
    x = np.arange(len(summary["percentiles"]))
    width = 0.36

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.bar(
        x - width / 2,
        summary["chosen_len"],
        width,
        yerr=summary["chosen_len_std"],
        capsize=4,
        label="chosen_len",
    )
    ax.bar(
        x + width / 2,
        summary["rejected_len"],
        width,
        yerr=summary["rejected_len_std"],
        capsize=4,
        label="rejected_len",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["percentiles"])
    ax.set_xlabel("Percentile Window Center")
    ax.set_ylabel("Token Length")
    ax.set_title(f"chosen/rejected length across {metric} windows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    png_path = os.path.join(output_plot_dir, f"{metric}_length_bar.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_edit_distance_summary(output_plot_dir: str, metric: str, summary: Dict[str, List[float]]) -> None:
    x = np.arange(len(summary["percentiles"]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.bar(
        x,
        summary["normalized_edit_distance"],
        yerr=summary["normalized_edit_distance_std"],
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["percentiles"])
    ax.set_xlabel("Percentile Window Center")
    ax.set_ylabel("normalized_edit_distance")
    ax.set_title(f"normalized edit distance across {metric} windows")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    png_path = os.path.join(output_plot_dir, f"{metric}_normalized_edit_distance_bar.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_seq_logprob_summary(output_plot_dir: str, metric: str, summary: Dict[str, List[float]]) -> None:
    x = np.arange(len(summary["percentiles"]))
    width = 0.36

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.bar(
        x - width / 2,
        summary["chosen_seq_logprob"],
        width,
        yerr=summary["chosen_seq_logprob_std"],
        capsize=4,
        label="chosen_seq_logprob",
    )
    ax.bar(
        x + width / 2,
        summary["rejected_seq_logprob"],
        width,
        yerr=summary["rejected_seq_logprob_std"],
        capsize=4,
        label="rejected_seq_logprob",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["percentiles"])
    ax.set_xlabel("Percentile Window Center")
    ax.set_ylabel("Sequence Log Probability")
    ax.set_title(f"sequence log-prob across {metric} windows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    png_path = os.path.join(output_plot_dir, f"{metric}_seq_logprob_bar.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_avg_logprob_summary(output_plot_dir: str, metric: str, summary: Dict[str, List[float]]) -> None:
    x = np.arange(len(summary["percentiles"]))
    width = 0.36

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.bar(
        x - width / 2,
        summary["chosen_avg_token_logprob"],
        width,
        yerr=summary["chosen_avg_token_logprob_std"],
        capsize=4,
        label="chosen_avg_token_logprob",
    )
    ax.bar(
        x + width / 2,
        summary["rejected_avg_token_logprob"],
        width,
        yerr=summary["rejected_avg_token_logprob_std"],
        capsize=4,
        label="rejected_avg_token_logprob",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["percentiles"])
    ax.set_xlabel("Percentile Window Center")
    ax.set_ylabel("Average Token Log Probability")
    ax.set_title(f"avg token log-prob across {metric} windows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    png_path = os.path.join(output_plot_dir, f"{metric}_avg_token_logprob_bar.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot centered percentile summary bar charts from output_dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory from centered_percentile_pair_builder.py")
    parser.add_argument("--window_size", type=int, default=1024, help="Window size used when building percentile datasets")
    args = parser.parse_args()

    window_metadata_dir = os.path.join(args.output_dir, "window_metadata")
    output_plot_dir = os.path.join(args.output_dir, "summary_plots")
    os.makedirs(output_plot_dir, exist_ok=True)

    for metric in METRICS:
        summary = build_metric_summary(window_metadata_dir, metric, args.window_size)
        plot_metric_summary(output_plot_dir, metric, summary)
        plot_auxiliary_summary(output_plot_dir, metric, summary)
        plot_edit_distance_summary(output_plot_dir, metric, summary)
        plot_seq_logprob_summary(output_plot_dir, metric, summary)
        plot_avg_logprob_summary(output_plot_dir, metric, summary)
        print(f"[Saved] 5 plots for {metric}")

    print(f"All plots saved to: {output_plot_dir}")


if __name__ == "__main__":
    main()
