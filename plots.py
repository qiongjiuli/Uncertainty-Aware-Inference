"""
Calibration visualization utilities.
All functions return the figure so callers can save or embed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from .metrics import CalibrationResult


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    result: CalibrationResult,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Classic reliability diagram: confidence vs accuracy per bin.
    Gap between bars and diagonal = calibration error.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    bin_conf = np.array(result.bin_confidences)
    bin_acc  = np.array(result.bin_accuracies)
    bin_cnt  = np.array(result.bin_counts)
    nonempty = bin_cnt > 0
    width    = 1.0 / len(bin_conf)

    # Gap bars (calibration error, red)
    ax.bar(
        bin_conf[nonempty],
        np.abs(bin_acc[nonempty] - bin_conf[nonempty]),
        bottom=np.minimum(bin_acc[nonempty], bin_conf[nonempty]),
        width=width * 0.9,
        color="salmon",
        alpha=0.5,
        label="Gap (calibration error)",
    )
    # Accuracy bars (blue)
    ax.bar(
        bin_conf[nonempty],
        bin_acc[nonempty],
        width=width * 0.9,
        color="steelblue",
        alpha=0.6,
        label="Accuracy",
    )
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"{result.model_name} — {result.precision}\n"
        f"ECE = {result.ece:.4f}   MCE = {result.mce:.4f}   "
        f"Acc = {result.accuracy:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)

    if standalone and save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Entropy histogram
# ---------------------------------------------------------------------------

def plot_entropy_comparison(
    result_fp16: CalibrationResult,
    result_quant: CalibrationResult,
    entropies_fp16: np.ndarray,
    entropies_quant: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Side-by-side entropy distributions: FP16 vs quantized.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    hi = max(entropies_fp16.max(), entropies_quant.max()) * 1.05
    bins = np.linspace(0, hi, 50)

    for ax, (ent, res) in zip(
        axes,
        [(entropies_fp16, result_fp16), (entropies_quant, result_quant)],
    ):
        ax.hist(ent, bins=bins, color="steelblue", alpha=0.7, label="All samples")
        ax.axvline(
            res.entropy_correct,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Correct  μ={res.entropy_correct:.3f}",
        )
        ax.axvline(
            res.entropy_incorrect,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Incorrect μ={res.entropy_incorrect:.3f}",
        )
        ax.set_title(f"{res.model_name} — {res.precision}", fontsize=11)
        ax.set_xlabel("Entropy (nats)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Token-Level Entropy Distribution: FP16 vs Quantized", fontsize=13)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Multi-config comparison bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    results: list[CalibrationResult],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing ECE, MCE, Brier, and Accuracy
    across multiple precision configs for one model.
    """
    metrics = ["ece", "mce", "brier", "accuracy"]
    labels  = [r.precision for r in results]
    n       = len(results)
    x       = np.arange(len(metrics))
    width   = 0.8 / n
    colors  = plt.cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [getattr(res, m) for m in metrics]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=color, label=res.precision)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        f"Calibration Metrics — {results[0].model_name}\n"
        f"(ECE/MCE/Brier: lower is better | Accuracy: higher is better)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.2))
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Full dashboard (4-panel) for one config
# ---------------------------------------------------------------------------

def plot_dashboard(
    result: CalibrationResult,
    entropies: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    4-panel dashboard: reliability diagram, metric bars,
    entropy histogram, and Brier decomposition.
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Reliability diagram
    ax1 = fig.add_subplot(gs[0, 0])
    plot_reliability_diagram(result, ax=ax1)

    # 2. Calibration metric bars
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = {
        "ECE": result.ece,
        "MCE": result.mce,
        "Brier": result.brier,
        "1 - Acc": 1 - result.accuracy,
    }
    colors_bar = ["steelblue", "coral", "seagreen", "orchid"]
    bars = ax2.bar(metrics.keys(), metrics.values(), color=colors_bar)
    for bar, v in zip(bars, metrics.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f"{v:.4f}", ha="center", fontsize=10)
    ax2.set_title(f"Key Metrics — {result.precision}", fontsize=11)
    ax2.set_ylabel("Error (lower is better)")
    ax2.set_ylim(0, max(metrics.values()) * 1.3)
    ax2.grid(axis="y", alpha=0.3)

    # 3. Entropy histogram
    ax3 = fig.add_subplot(gs[1, 0])
    if entropies is not None:
        bins = np.linspace(0, entropies.max() * 1.05, 40)
        ax3.hist(entropies, bins=bins, color="steelblue", alpha=0.7)
        ax3.axvline(result.entropy_correct,   color="green", ls="--",
                    label=f"Correct μ={result.entropy_correct:.3f}")
        ax3.axvline(result.entropy_incorrect, color="red",   ls="--",
                    label=f"Incorrect μ={result.entropy_incorrect:.3f}")
        ax3.set_xlabel("Entropy (nats)")
        ax3.set_ylabel("Count")
        ax3.set_title("Entropy Distribution")
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, "No entropy data provided",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Entropy Distribution")

    # 4. Brier decomposition
    ax4 = fig.add_subplot(gs[1, 1])
    components = {
        "Reliability\n(↓ better)": result.brier_reliability,
        "Resolution\n(↑ better)": result.brier_resolution,
        "Uncertainty\n(fixed)":   result.brier_uncertainty,
        "Brier\n(↓ better)":      result.brier,
    }
    ax4.bar(components.keys(), components.values(),
            color=["coral", "seagreen", "gray", "steelblue"])
    ax4.set_title("Brier Score Decomposition")
    ax4.set_ylabel("Value")
    ax4.grid(axis="y", alpha=0.3)
    for i, v in enumerate(components.values()):
        ax4.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    fig.suptitle(
        f"Calibration Dashboard — {result.model_name} [{result.precision}]\n"
        f"Dataset: {result.dataset}  |  N = {result.n_samples}",
        fontsize=14,
        fontweight="bold",
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
