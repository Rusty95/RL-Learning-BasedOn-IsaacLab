"""Plot standalone PPO training curves from metrics.csv."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "standalone" / "ppo"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def find_latest_metrics(log_root: Path) -> Path:
    candidates = sorted(log_root.rglob("metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv files found under {log_root}")
    return candidates[-1]


def load_metrics(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    series: dict[str, list[float]] = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            series[key].append(float(value))
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot standalone PPO metrics.")
    parser.add_argument("--metrics", type=Path, default=None, help="Path to metrics.csv. Defaults to latest run.")
    parser.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT, help="Root folder to search for runs.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path. Defaults beside metrics.csv.")
    args = parser.parse_args()

    metrics_path = args.metrics or find_latest_metrics(args.log_root)
    output_path = args.output or metrics_path.with_name("training_curves.png")
    data = load_metrics(metrics_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = data["iteration"]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True)
    fig.suptitle(f"Standalone PPO Training Curves\n{metrics_path.parent}", fontsize=13)

    plots = [
        ("mean_reward_100", "Mean Reward / Last 100 Episodes"),
        ("mean_length_100", "Mean Episode Length / Last 100"),
        ("policy_loss", "Policy Loss"),
        ("value_loss", "Value Loss"),
        ("entropy", "Entropy"),
        ("approx_kl", "Approx KL"),
        ("clip_fraction", "Clip Fraction"),
        ("explained_variance", "Value Explained Variance"),
    ]
    colors = ["#0e7c86", "#287447", "#b7552c", "#5e6f67", "#a76a16", "#17211c", "#6b5ca5", "#3d7c4a"]

    for ax, (key, title), color in zip(axes.flat, plots, colors, strict=True):
        if key in data:
            ax.plot(x, data[key], color=color, linewidth=2)
        else:
            ax.text(0.5, 0.5, f"{key} not found", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.28)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
