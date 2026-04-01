"""
Pareto analysis and uncertainty-based routing simulation.

Pareto:
  Identifies configurations where no other config is simultaneously
  cheaper, more accurate, and better calibrated.
  Objectives: accuracy ↑, ECE ↓, tokens/sec ↑, GPU memory ↓

Routing:
  Simulates a confidence-based routing strategy:
    if quantized_model.confidence(query) >= threshold:
        serve with quantized model  (cheap)
    else:
        escalate to FP16            (expensive, better calibrated)
  Sweeps threshold and reports cost savings vs quality trade-off.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3d projection)
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified experiment point (joins calibration + profiling)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentPoint:
    model_name    : str
    precision     : str
    dataset       : str
    # Calibration
    ece           : float
    mce           : float
    brier         : float
    accuracy      : float
    mean_confidence: float
    # Profiling
    tokens_per_sec: float
    gpu_mem_gb    : float
    latency_ms    : float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dicts(cls, cal: dict, prof: dict) -> "ExperimentPoint":
        return cls(
            model_name     = cal["model_name"],
            precision      = cal["precision"],
            dataset        = cal.get("dataset", ""),
            ece            = cal["ece"],
            mce            = cal["mce"],
            brier          = cal["brier"],
            accuracy       = cal["accuracy"],
            mean_confidence= cal.get("mean_confidence", 0.0),
            tokens_per_sec = prof.get("tokens_per_sec", 0.0),
            gpu_mem_gb     = prof.get("gpu_mem_peak_gb", 0.0),
            latency_ms     = prof.get("latency_ms_mean", 0.0),
        )


def load_experiment_points(
    calibration_dir : str | Path,
    profiling_dir   : str | Path,
) -> list[ExperimentPoint]:
    """
    Load all (calibration, profiling) JSON pairs from results directories.
    Matches by filename tag: {ModelName}_{Precision}_calibration.json
    """
    cal_dir  = Path(calibration_dir)
    prof_dir = Path(profiling_dir)
    points   = []

    for cal_file in sorted(cal_dir.glob("*_calibration.json")):
        tag        = cal_file.stem.replace("_calibration", "")
        prof_file  = prof_dir / f"{tag}_profiling.json"
        if not prof_file.exists():
            logger.warning(f"No profiling file for {tag}, skipping")
            continue
        with open(cal_file)  as f: cal  = json.load(f)
        with open(prof_file) as f: prof = json.load(f)
        points.append(ExperimentPoint.from_dicts(cal, prof))

    logger.info(f"Loaded {len(points)} experiment points")
    return points


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

def _objectives(p: ExperimentPoint) -> np.ndarray:
    """
    Convert ExperimentPoint to objectives vector (all higher = better).
    """
    return np.array([
        p.accuracy,          # ↑
        -p.ece,              # ↑ (lower ECE is better)
        -p.brier,            # ↑
        p.tokens_per_sec / 1000.0,   # ↑ (normalize scale)
        -p.gpu_mem_gb,       # ↑ (lower memory is better)
    ])


def is_dominated(p: ExperimentPoint, others: list[ExperimentPoint]) -> bool:
    """True if some other point dominates p on all objectives."""
    p_obj = _objectives(p)
    for o in others:
        if o is p:
            continue
        o_obj = _objectives(o)
        if np.all(o_obj >= p_obj) and np.any(o_obj > p_obj):
            return True
    return False


def pareto_front(points: list[ExperimentPoint]) -> list[ExperimentPoint]:
    """Return the non-dominated subset."""
    front = [p for p in points if not is_dominated(p, points)]
    logger.info(f"Pareto front: {len(front)} / {len(points)} configs")
    return front


# ---------------------------------------------------------------------------
# Routing simulation
# ---------------------------------------------------------------------------

@dataclass
class RoutingPoint:
    threshold        : float
    frac_cheap       : float    # fraction served by quantized model
    effective_acc    : float
    effective_ece    : float
    cost_saving_pct  : float    # savings vs all-FP16


def simulate_routing(
    confidences   : np.ndarray,   # max-prob of quantized model per sample  (N,)
    correct_quant : np.ndarray,   # 1/0 correctness of quantized model       (N,)
    correct_fp16  : np.ndarray,   # 1/0 correctness of FP16 model            (N,)
    ece_quant     : float,
    ece_fp16      : float,
    fp16_cost     : float = 1.0,
    quant_cost    : float = 0.25,  # INT4 is ~4x cheaper than FP16
    thresholds    : Optional[np.ndarray] = None,
) -> list[RoutingPoint]:
    """
    Simulate confidence-threshold routing and return metrics at each threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.99, 100)

    results = []
    N = len(confidences)

    for thresh in thresholds:
        cheap_mask  = confidences >= thresh    # bool (N,)
        frac_cheap  = float(cheap_mask.mean())

        n_cheap = cheap_mask.sum()
        n_exp   = (~cheap_mask).sum()

        acc_cheap = float(correct_quant[cheap_mask].mean()) if n_cheap > 0 else 0.0
        acc_exp   = float(correct_fp16[~cheap_mask].mean()) if n_exp   > 0 else 1.0

        effective_acc = frac_cheap * acc_cheap + (1 - frac_cheap) * acc_exp
        effective_ece = frac_cheap * ece_quant + (1 - frac_cheap) * ece_fp16

        effective_cost  = frac_cheap * quant_cost + (1 - frac_cheap) * fp16_cost
        cost_saving_pct = (fp16_cost - effective_cost) / fp16_cost * 100.0

        results.append(RoutingPoint(
            threshold       = float(thresh),
            frac_cheap      = frac_cheap,
            effective_acc   = float(effective_acc),
            effective_ece   = float(effective_ece),
            cost_saving_pct = float(cost_saving_pct),
        ))

    return results


def find_optimal_threshold(
    routing       : list[RoutingPoint],
    fp16_acc      : float,
    fp16_ece      : float,
    acc_tolerance : float = 0.01,
    ece_tolerance : float = 0.005,
) -> Optional[RoutingPoint]:
    """
    Find the highest-savings threshold that stays within quality budgets.

    acc_tolerance: max allowed accuracy drop vs FP16
    ece_tolerance: max allowed ECE increase vs FP16
    """
    candidates = [
        r for r in routing
        if (fp16_acc - r.effective_acc) <= acc_tolerance
        and (r.effective_ece - fp16_ece) <= ece_tolerance
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.cost_saving_pct)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pareto_2d(
    points    : list[ExperimentPoint],
    pareto    : list[ExperimentPoint],
    x_attr    : str,
    y_attr    : str,
    x_label   : str,
    y_label   : str,
    save_path : str | Path,
    title     : str = "",
) -> plt.Figure:
    pareto_ids = {id(p) for p in pareto}

    fig, ax = plt.subplots(figsize=(9, 7))
    models  = sorted({p.model_name for p in points})
    palette = cm.tab10(np.linspace(0, 1, len(models)))
    color_map = {m: palette[i] for i, m in enumerate(models)}

    for p in points:
        c = color_map[p.model_name]
        marker, size, zorder = ("*", 200, 5) if id(p) in pareto_ids else ("o", 80, 3)
        ax.scatter(getattr(p, x_attr), getattr(p, y_attr),
                   c=[c], s=size, marker=marker, zorder=zorder)
        ax.annotate(
            f"{p.model_name}\n{p.precision}",
            (getattr(p, x_attr), getattr(p, y_attr)),
            textcoords="offset points", xytext=(6, 4), fontsize=7,
        )

    # Legend
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[m], markersize=9, label=m)
        for m in models
    ]
    model_handles.append(
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="black", markersize=12, label="Pareto-optimal")
    )
    ax.legend(handles=model_handles, fontsize=9)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title or f"Pareto: {x_label} vs {y_label}", fontsize=12)
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_pareto_3d(
    points    : list[ExperimentPoint],
    pareto    : list[ExperimentPoint],
    save_path : str | Path,
) -> plt.Figure:
    """3D Pareto: latency × accuracy × ECE."""
    pareto_ids = {id(p) for p in pareto}
    models     = sorted({p.model_name for p in points})
    palette    = cm.tab10(np.linspace(0, 1, len(models)))
    color_map  = {m: palette[i] for i, m in enumerate(models)}

    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection="3d")

    for p in points:
        c      = color_map[p.model_name]
        on_pf  = id(p) in pareto_ids
        marker, size = ("*", 250) if on_pf else ("o", 60)
        ax.scatter(p.latency_ms, p.accuracy, p.ece,
                   c=[c], s=size, marker=marker,
                   depthshade=True, zorder=5 if on_pf else 3)
        ax.text(p.latency_ms, p.accuracy, p.ece,
                f"  {p.precision}", fontsize=7)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[m], markersize=9, label=m)
        for m in models
    ]
    handles.append(
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="black", markersize=13, label="Pareto-optimal")
    )
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.set_xlabel("Latency (ms/token)", fontsize=10)
    ax.set_ylabel("Accuracy",           fontsize=10)
    ax.set_zlabel("ECE (↓ better)",     fontsize=10)
    ax.set_title("3D Pareto Frontier: Cost × Accuracy × Calibration", fontsize=12)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_routing(
    routing     : list[RoutingPoint],
    fp16_acc    : float,
    fp16_ece    : float,
    optimal     : Optional[RoutingPoint],
    label       : str,
    save_path   : str | Path,
) -> plt.Figure:
    """4-panel routing analysis plot."""
    thresholds   = [r.threshold       for r in routing]
    frac_cheap   = [r.frac_cheap      for r in routing]
    acc          = [r.effective_acc   for r in routing]
    ece          = [r.effective_ece   for r in routing]
    savings      = [r.cost_saving_pct for r in routing]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Routing rate
    axes[0, 0].plot(thresholds, frac_cheap, "steelblue", lw=2)
    if optimal:
        axes[0, 0].axvline(optimal.threshold, color="red", ls="--",
                           label=f"Optimal θ={optimal.threshold:.2f}")
        axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_xlabel("Confidence Threshold")
    axes[0, 0].set_ylabel("Fraction Served Cheaply")
    axes[0, 0].set_title("Routing Rate vs Threshold")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Cost savings
    axes[0, 1].plot(thresholds, savings, "seagreen", lw=2)
    if optimal:
        axes[0, 1].axvline(optimal.threshold, color="red", ls="--")
        axes[0, 1].scatter([optimal.threshold], [optimal.cost_saving_pct],
                           color="red", s=100, zorder=5,
                           label=f"Max saving = {optimal.cost_saving_pct:.1f}%")
        axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_xlabel("Confidence Threshold")
    axes[0, 1].set_ylabel("Cost Saving vs FP16 (%)")
    axes[0, 1].set_title("Cost Saving vs Threshold")
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Accuracy
    axes[1, 0].plot(thresholds, acc, "coral", lw=2, label="Mixed system")
    axes[1, 0].axhline(fp16_acc, color="gray", ls="--", lw=1.5, label="FP16 baseline")
    if optimal:
        axes[1, 0].axvline(optimal.threshold, color="red", ls="--")
    axes[1, 0].set_xlabel("Confidence Threshold")
    axes[1, 0].set_ylabel("Effective Accuracy")
    axes[1, 0].set_title("Accuracy vs Threshold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Calibration (ECE)
    axes[1, 1].plot(thresholds, ece, "orchid", lw=2, label="Mixed system")
    axes[1, 1].axhline(fp16_ece, color="gray", ls="--", lw=1.5, label="FP16 baseline")
    if optimal:
        axes[1, 1].axvline(optimal.threshold, color="red", ls="--")
    axes[1, 1].set_xlabel("Confidence Threshold")
    axes[1, 1].set_ylabel("Effective ECE")
    axes[1, 1].set_title("Calibration (ECE) vs Threshold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Uncertainty-Based Routing Simulation\n"
        f"Quantized config: {label}",
        fontsize=13, fontweight="bold",
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_cross_model_heatmap(
    points    : list[ExperimentPoint],
    metric    : str,
    save_path : str | Path,
) -> plt.Figure:
    """Heatmap of metric across model × precision."""
    models     = sorted({p.model_name for p in points})
    precisions = sorted({p.precision  for p in points})

    grid = np.full((len(models), len(precisions)), np.nan)
    for p in points:
        i = models.index(p.model_name)
        j = precisions.index(p.precision)
        grid[i, j] = getattr(p, metric)

    lower_better = metric in ("ece", "mce", "brier", "latency_ms", "gpu_mem_gb")
    cmap = "RdYlGn_r" if lower_better else "RdYlGn"

    fig, ax = plt.subplots(figsize=(max(8, len(precisions) * 1.8), max(4, len(models) * 1.2)))
    im = ax.imshow(grid, aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(precisions)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(precisions, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(precisions)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.4f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(grid[i, j]) > 0.5 else "black")

    lower_str = "(↓ better)" if lower_better else "(↑ better)"
    ax.set_title(f"Cross-Model Heatmap: {metric.upper()} {lower_str}", fontsize=12)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
