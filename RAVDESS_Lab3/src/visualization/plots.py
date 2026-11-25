"""Save distribution and ROC plots from paired neutral/sad recording summaries."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


def plot_feature_comparison(neutral: pd.DataFrame, sad: pd.DataFrame,
                            output_dir: str | Path, selected: list[tuple[str, str]]) -> list[Path]:
    """Save an independent paired-data plot for each feature/statistic selection."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for characteristic, statistic in selected:
        x = neutral[(characteristic, statistic)].to_numpy(dtype=float)
        y = sad[(characteristic, statistic)].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        for a, b in zip(x, y):
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([0, 1], [a, b], color="0.75", alpha=0.55, linewidth=0.8)
        ax.scatter(np.zeros(len(x)), x, s=28, color="#284B63", label="Neutral")
        ax.scatter(np.ones(len(y)), y, s=28, color="#B05A3C", label="Sad")
        ax.set_xticks([0, 1], ["Neutral", "Sad"])
        ax.set_ylabel(f"{characteristic}: {statistic}")
        ax.set_title("Paired actor comparison (acted emotions)")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        outfile = destination / f"paired_{characteristic.lower().replace(' ','_')}_{statistic.lower()}.png"
        fig.savefig(outfile, dpi=160)
        plt.close(fig)
        paths.append(outfile)
    return paths


def plot_roc(predictions: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot grouped cross-validated ROC curves (one curve per model)."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, group in predictions.groupby("Model", sort=False):
        fpr, tpr, _ = roc_curve(group["True"], group["P(Sad)"])
        auc = roc_auc_score(group["True"], group["P(Sad)"])
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out-actor ROC curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = destination / "roc_grouped_cv.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_benchmark_comparison(timing: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save a grouped chart of per-stage reference and optimized wall times."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(timing))
    width = 0.37
    ax.bar(x - width / 2, timing["Reference [s]"], width, label="Reference")
    ax.bar(x + width / 2, timing["Optimized [s]"], width, label="Optimized")
    ax.set_xticks(x, timing["Stage"].str.replace(" (excluding identical pYIN)", "", regex=False), rotation=12, ha="right")
    ax.set_ylabel("Elapsed time (s), minimum of repeats")
    ax.set_title("Independent benchmark stages — synthetic demo only")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = destination / "benchmark_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
