#!/usr/bin/env python
"""
scripts/run_analysis.py

Cross-model Pareto analysis and routing simulation.
Run AFTER run_ptq_sweep.py and run_profiling.py have completed
for all three models and all precision configs.

Usage:
    python scripts/run_analysis.py \
        --calibration_dirs results/llama2_7b results/mistral_7b results/llama2_13b \
        --profiling_dir results/profiling \
        --output_dir results/analysis
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.pareto   import (
    ExperimentPoint,
    find_optimal_threshold,
    load_experiment_points,
    pareto_front,
    plot_cross_model_heatmap,
    plot_pareto_2d,
    plot_pareto_3d,
    plot_routing,
    simulate_routing,
)
from src.utils.logging     import init_wandb, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-model Pareto + routing analysis")
    p.add_argument("--calibration_dirs", nargs="+", required=True,
                   help="One results dir per model (from run_ptq_sweep.py)")
    p.add_argument("--profiling_dir",    required=True,
                   help="Profiling results dir (from run_profiling.py)")
    p.add_argument("--output_dir",       default="results/analysis")
    p.add_argument("--dataset",          default="arc_challenge",
                   help="Which dataset to use for calibration numbers in Pareto")
    p.add_argument("--acc_tolerance",   type=float, default=0.01)
    p.add_argument("--ece_tolerance",   type=float, default=0.005)
    p.add_argument("--wandb_project",   default="uncertainty-aware-inference")
    p.add_argument("--no_wandb",        action="store_true")
    return p.parse_args()


def _load_all_points(
    cal_dirs: list[Path],
    prof_dir: Path,
    dataset: str,
) -> list[ExperimentPoint]:
    """Load experiment points from multiple model calibration directories."""
    points = []
    for cal_dir in cal_dirs:
        # Filter JSON files to the requested dataset
        dataset_cal_dir = cal_dir  # calibration JSONs are flat in results dir
        for cal_file in sorted(cal_dir.glob(f"*_{dataset}_calibration.json")):
            tag       = cal_file.stem.replace(f"_{dataset}_calibration", "")
            prof_file = prof_dir / f"{tag}_profiling.json"
            if not prof_file.exists():
                logger.warning(f"Missing profiling for {tag}")
                continue
            with open(cal_file)  as f: cal  = json.load(f)
            with open(prof_file) as f: prof = json.load(f)
            points.append(ExperimentPoint.from_dicts(cal, prof))

    logger.info(f"Loaded {len(points)} total experiment points")
    return points


def _make_synthetic_confidences(
    n: int = 1000,
    quant_acc: float = 0.70,
    fp16_acc: float  = 0.75,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic confidence arrays for routing simulation when real
    per-sample confidences are not available.

    In production: replace with actual per-sample softmax-max values
    saved during calibration evaluation.
    """
    rng = np.random.default_rng(seed)
    confidences   = rng.beta(5, 2, n).clip(0.01, 0.99)
    correct_quant = (rng.uniform(size=n) < quant_acc).astype(float)
    correct_fp16  = (rng.uniform(size=n) < fp16_acc).astype(float)
    return confidences, correct_quant, correct_fp16


def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    setup_logging(output_dir / "logs")

    run = None
    if not args.no_wandb:
        run = init_wandb(
            project = args.wandb_project,
            name    = "cross-model-analysis",
            tags    = ["team-c", "pareto", "routing"],
            config  = vars(args),
        )

    cal_dirs = [Path(d) for d in args.calibration_dirs]
    prof_dir = Path(args.profiling_dir)

    # ── Load all experiment points ──
    points = _load_all_points(cal_dirs, prof_dir, args.dataset)
    if not points:
        logger.error("No experiment points loaded. Check that sweep and profiling ran first.")
        sys.exit(1)

    # ── Pareto analysis ──
    pf = pareto_front(points)

    logger.info("\n=== Pareto-Optimal Configurations ===")
    for p in pf:
        logger.info(
            f"  {p.model_name:15s} {p.precision:12s} "
            f"ECE={p.ece:.4f}  Acc={p.accuracy:.4f}  "
            f"Lat={p.latency_ms:.1f}ms  Mem={p.gpu_mem_gb:.1f}GB"
        )

    # Save
    with open(output_dir / "pareto_front.json", "w") as f:
        json.dump([p.to_dict() for p in pf], f, indent=2)

    # 2D Pareto plots
    plot_pairs = [
        ("latency_ms",    "accuracy",      "Latency (ms/token)",    "Accuracy"),
        ("ece",           "accuracy",      "ECE (↓ better)",        "Accuracy"),
        ("gpu_mem_gb",    "ece",           "GPU Memory (GB)",       "ECE (↓ better)"),
        ("tokens_per_sec","ece",           "Throughput (tok/s)",    "ECE (↓ better)"),
    ]
    for x_attr, y_attr, x_label, y_label in plot_pairs:
        plot_pareto_2d(
            points, pf,
            x_attr=x_attr, y_attr=y_attr,
            x_label=x_label, y_label=y_label,
            save_path=output_dir / "plots" / f"pareto_{x_attr}_vs_{y_attr}.png",
            title=f"Pareto: {x_label} vs {y_label}\n(★ = Pareto-optimal)",
        )

    # 3D Pareto
    plot_pareto_3d(
        points, pf,
        save_path=output_dir / "plots" / "pareto_3d.png",
    )

    # Cross-model heatmaps
    for metric in ["ece", "brier", "accuracy", "tokens_per_sec", "gpu_mem_gb"]:
        try:
            plot_cross_model_heatmap(
                points, metric,
                save_path=output_dir / "plots" / f"heatmap_{metric}.png",
            )
        except Exception as e:
            logger.warning(f"Heatmap failed for {metric}: {e}")

    # ── Routing simulation ──
    logger.info("\n=== Routing Simulation ===")
    routing_summary = {}

    # Get FP16 baseline per model
    fp16_by_model: dict[str, ExperimentPoint] = {
        p.model_name: p for p in points if p.precision == "FP16"
    }

    for p in points:
        if p.precision == "FP16":
            continue
        fp16 = fp16_by_model.get(p.model_name)
        if fp16 is None:
            continue

        # Use synthetic confidences (replace with real ones in production)
        confidences, correct_quant, correct_fp16 = _make_synthetic_confidences(
            quant_acc = p.accuracy,
            fp16_acc  = fp16.accuracy,
        )

        routing = simulate_routing(
            confidences   = confidences,
            correct_quant = correct_quant,
            correct_fp16  = correct_fp16,
            ece_quant     = p.ece,
            ece_fp16      = fp16.ece,
        )

        optimal = find_optimal_threshold(
            routing,
            fp16_acc      = fp16.accuracy,
            fp16_ece      = fp16.ece,
            acc_tolerance = args.acc_tolerance,
            ece_tolerance = args.ece_tolerance,
        )

        tag = f"{p.model_name}_{p.precision}".replace(" ", "_").replace("/", "_")
        plot_routing(
            routing, fp16.accuracy, fp16.ece, optimal,
            label     = f"{p.model_name} {p.precision}",
            save_path = output_dir / "plots" / f"routing_{tag}.png",
        )

        if optimal:
            logger.info(
                f"  {p.model_name:15s} {p.precision:12s} | "
                f"θ*={optimal.threshold:.2f}  "
                f"saving={optimal.cost_saving_pct:.1f}%  "
                f"frac_cheap={optimal.frac_cheap:.2f}  "
                f"Δacc={fp16.accuracy - optimal.effective_acc:.4f}  "
                f"ΔECE={optimal.effective_ece - fp16.ece:.4f}"
            )
            routing_summary[tag] = {
                "model"          : p.model_name,
                "precision"      : p.precision,
                "optimal_threshold"  : optimal.threshold,
                "cost_saving_pct": optimal.cost_saving_pct,
                "frac_cheap"     : optimal.frac_cheap,
                "acc_drop"       : fp16.accuracy - optimal.effective_acc,
                "ece_increase"   : optimal.effective_ece - fp16.ece,
            }
            if run:
                run.log({f"routing/{tag}/{k}": v
                         for k, v in routing_summary[tag].items()
                         if isinstance(v, float)})
        else:
            logger.info(
                f"  {p.model_name:15s} {p.precision:12s} | "
                "No threshold meets quality constraints"
            )

    with open(output_dir / "routing_summary.json", "w") as f:
        json.dump(routing_summary, f, indent=2)

    if run:
        run.finish()

    logger.info(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
