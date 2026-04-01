#!/usr/bin/env python
"""
scripts/run_kd.py

Runs the knowledge distillation experiment:
  Teacher: FP16 model
  Student: INT4 quantized model (GPTQ or AWQ)

Trains the student to match teacher soft distributions,
then re-evaluates calibration and computes recovery metrics.

Usage:
    python scripts/run_kd.py \
        --model_id meta-llama/Llama-2-7b-hf \
        --model_name Llama-2-7B \
        --student_precision GPTQ_INT4 \
        --results_dir results/llama2_7b \
        --output_dir results/llama2_7b/kd \
        --n_kd_samples 2000 \
        --n_epochs 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.calibration.datasets   import load_dataset_mcq, load_kd_corpus
from src.calibration.metrics    import CalibrationResult, evaluate_calibration
from src.calibration.plots      import plot_metrics_comparison
from src.distillation.trainer   import KDConfig, KDTrainResult, compute_kd_recovery, train_kd
from src.quantization.loaders   import free_model, load_model
from src.utils.logging          import init_wandb, log_calibration, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Knowledge distillation experiment")
    p.add_argument("--model_id",          required=True)
    p.add_argument("--model_name",         required=True)
    p.add_argument("--student_precision",  default="GPTQ_INT4",
                   choices=["GPTQ_INT4", "AWQ_INT4", "NF4"])
    p.add_argument("--results_dir",        required=True,
                   help="Dir containing pre-computed calibration JSONs from run_ptq_sweep.py")
    p.add_argument("--output_dir",         default="results/kd")
    p.add_argument("--eval_dataset",       default="arc_challenge",
                   choices=["arc_challenge", "hellaswag", "triviaqa"])
    p.add_argument("--n_eval_samples",     type=int, default=500)
    p.add_argument("--n_kd_samples",       type=int, default=2000)
    p.add_argument("--n_epochs",           type=int, default=3)
    p.add_argument("--temperature",        type=float, default=4.0)
    p.add_argument("--alpha",              type=float, default=0.7)
    p.add_argument("--lr",                 type=float, default=2e-5)
    p.add_argument("--batch_size",         type=int, default=4)
    p.add_argument("--max_length",         type=int, default=256)
    p.add_argument("--device",             default="cuda")
    p.add_argument("--wandb_project",      default="uncertainty-aware-inference")
    p.add_argument("--no_wandb",           action="store_true")
    return p.parse_args()


def load_precomputed_result(
    results_dir: Path,
    model_name: str,
    precision: str,
    dataset: str,
) -> CalibrationResult:
    """Load a previously saved calibration JSON."""
    path = results_dir / f"{model_name}_{precision}_{dataset}_calibration.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-computed calibration not found: {path}\n"
            "Run run_ptq_sweep.py first."
        )
    return CalibrationResult.load(path)


def plot_recovery_bars(
    fp16_result    : CalibrationResult,
    pre_kd_result  : CalibrationResult,
    post_kd_result : CalibrationResult,
    recovery       : dict,
    save_path      : Path,
) -> None:
    metrics  = ["ece", "mce", "brier", "accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    for ax, m in zip(axes, metrics):
        vals = [
            getattr(fp16_result,    m),
            getattr(pre_kd_result,  m),
            getattr(post_kd_result, m),
        ]
        colors = ["steelblue", "coral", "seagreen"]
        bars   = ax.bar(["FP16", f"INT4\n(pre-KD)", f"INT4\n(post-KD)"],
                        vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", fontsize=9)
        rec = recovery.get(m, 0.0)
        ax.set_title(f"{m.upper()}\nRecovery: {rec*100:.1f}%", fontsize=11)
        ax.set_ylabel(m)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"KD Calibration Recovery — {fp16_result.model_name}\n"
        f"Teacher: FP16  →  Student: {post_kd_result.precision}",
        fontsize=13, fontweight="bold",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Recovery plot saved: {save_path}")


def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs")

    kd_cfg = KDConfig(
        temperature  = args.temperature,
        alpha        = args.alpha,
        lr           = args.lr,
        n_epochs     = args.n_epochs,
        batch_size   = args.batch_size,
        max_length   = args.max_length,
    )

    run = None
    if not args.no_wandb:
        run = init_wandb(
            project = args.wandb_project,
            name    = f"{args.model_name}-kd-{args.student_precision}",
            tags    = [args.model_name, "kd", args.student_precision],
            config  = {**vars(args), **kd_cfg.__dict__},
        )

    results_dir = Path(args.results_dir)

    # ── Load pre-computed calibration results ──
    logger.info("Loading pre-computed calibration results ...")
    fp16_result = load_precomputed_result(
        results_dir, args.model_name, "FP16", args.eval_dataset
    )
    pre_kd_result = load_precomputed_result(
        results_dir, args.model_name, args.student_precision, args.eval_dataset
    )
    logger.info(f"FP16 baseline  : {fp16_result.summary()}")
    logger.info(f"Pre-KD student : {pre_kd_result.summary()}")

    # ── Load models ──
    logger.info("\nLoading FP16 teacher ...")
    teacher, tokenizer = load_model(args.model_id, "FP16")

    logger.info(f"Loading {args.student_precision} student ...")
    student, _         = load_model(args.model_id, args.student_precision)

    # ── KD corpus ──
    logger.info("Loading KD corpus ...")
    texts = load_kd_corpus(n_samples=args.n_kd_samples)

    # ── Train ──
    logger.info("\nStarting knowledge distillation ...")
    train_result: KDTrainResult = train_kd(
        teacher_model = teacher,
        student_model = student,
        texts         = texts,
        tokenizer     = tokenizer,
        config        = kd_cfg,
        output_dir    = output_dir / "checkpoint",
        device        = args.device,
        wandb_run     = run,
    )
    logger.info(
        f"Training complete | "
        f"final KD loss={train_result.final_kd_loss:.4f} | "
        f"CE loss={train_result.final_ce_loss:.4f} | "
        f"steps={train_result.n_steps}"
    )
    free_model(teacher)

    # ── Re-evaluate student calibration post-KD ──
    logger.info("\nEvaluating post-KD calibration ...")
    eval_samples = load_dataset_mcq(args.eval_dataset, n_samples=args.n_eval_samples)

    post_kd_result = evaluate_calibration(
        model      = student,
        tokenizer  = tokenizer,
        samples    = eval_samples,
        model_name = args.model_name,
        precision  = f"{args.student_precision}_KD",
        dataset    = args.eval_dataset,
        device     = args.device,
    )
    post_kd_result.save(
        output_dir / f"{args.model_name}_{args.student_precision}_KD_{args.eval_dataset}_calibration.json"
    )
    log_calibration(run, post_kd_result)
    free_model(student)

    # ── Recovery analysis ──
    metrics_cfg = {
        "ece"     : False,   # lower is better
        "mce"     : False,
        "brier"   : False,
        "accuracy": True,    # higher is better
    }
    recovery = {}
    logger.info("\n=== KD Recovery Analysis ===")
    for m, higher_better in metrics_cfg.items():
        r = compute_kd_recovery(
            fp16_val         = getattr(fp16_result,    m),
            pre_kd_val       = getattr(pre_kd_result,  m),
            post_kd_val      = getattr(post_kd_result, m),
            metric_name      = m,
            higher_is_better = higher_better,
        )
        recovery[m] = r
        logger.info(f"  {m:12s}: {r*100:+.1f}% recovered")

    # Save recovery metrics
    with open(output_dir / "kd_recovery.json", "w") as f:
        json.dump(recovery, f, indent=2)

    # Plot
    plot_recovery_bars(
        fp16_result, pre_kd_result, post_kd_result,
        recovery,
        save_path=output_dir / "plots" / "kd_recovery.png",
    )

    if run:
        run.log({"recovery/" + k: v for k, v in recovery.items()})
        run.finish()

    logger.info(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
