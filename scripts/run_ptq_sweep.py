#!/usr/bin/env python
"""
scripts/run_ptq_sweep.py

Runs the full PTQ calibration sweep for one model:
  FP16 → GPTQ_INT8 → GPTQ_INT4 → AWQ_INT4 → NF4

For each config: loads model, runs calibration on ARC-Challenge,
HellaSwag, and TriviaQA, saves JSON results and plots.

Usage:
    python scripts/run_ptq_sweep.py \
        --model_id meta-llama/Llama-2-7b-hf \
        --model_name Llama-2-7B \
        --output_dir results/llama2_7b \
        --n_samples 500 \
        --precisions FP16 GPTQ_INT4 AWQ_INT4 NF4

    # GPTQ_INT8 adds bitsandbytes 8-bit; include if bitsandbytes installed
    # Omit --precisions to run all 5 configs
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.datasets  import load_dataset_mcq, DATASET_LOADERS
from src.calibration.metrics   import evaluate_calibration
from src.calibration.plots     import (
    plot_reliability_diagram,
    plot_metrics_comparison,
    plot_dashboard,
)
from src.quantization.loaders  import load_model, free_model, LOADERS
from src.utils.logging         import setup_logging, init_wandb, log_calibration

logger = logging.getLogger(__name__)

ALL_PRECISIONS = list(LOADERS.keys())                    # FP16, GPTQ_INT8, GPTQ_INT4, AWQ_INT4, NF4
EVAL_DATASETS  = list(DATASET_LOADERS.keys())            # arc_challenge, hellaswag, triviaqa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTQ calibration sweep")
    p.add_argument("--model_id",   required=True,
                   help="HuggingFace model ID, e.g. meta-llama/Llama-2-7b-hf")
    p.add_argument("--model_name", required=True,
                   help="Short display name, e.g. Llama-2-7B")
    p.add_argument("--output_dir", default="results",
                   help="Directory for JSON results and plots")
    p.add_argument("--precisions", nargs="+", default=ALL_PRECISIONS,
                   choices=ALL_PRECISIONS,
                   help="Which precision configs to run")
    p.add_argument("--datasets", nargs="+", default=EVAL_DATASETS,
                   choices=EVAL_DATASETS,
                   help="Which eval datasets to use")
    p.add_argument("--n_samples",  type=int, default=500,
                   help="Samples per dataset per config")
    p.add_argument("--n_bins",     type=int, default=15,
                   help="ECE bins")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--wandb_project", default="uncertainty-aware-inference")
    p.add_argument("--no_wandb",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs")

    run = None
    if not args.no_wandb:
        run = init_wandb(
            project = args.wandb_project,
            name    = f"{args.model_name}-ptq-sweep",
            tags    = [args.model_name, "ptq-sweep"],
            config  = vars(args),
        )

    # Load all eval samples once (reused across precisions)
    logger.info("Loading evaluation datasets ...")
    all_samples: dict[str, list[dict]] = {}
    for ds in args.datasets:
        all_samples[ds] = load_dataset_mcq(ds, n_samples=args.n_samples)
        logger.info(f"  {ds}: {len(all_samples[ds])} samples")

    all_results: dict[str, list] = {ds: [] for ds in args.datasets}

    for precision in args.precisions:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Precision: {precision}")
        logger.info(f"{'='*60}")

        try:
            model, tokenizer = load_model(
                args.model_id, precision, device_map="auto"
            )
        except Exception as e:
            logger.error(f"Failed to load {precision}: {e}")
            continue

        for ds_name, samples in all_samples.items():
            logger.info(f"\n--- Dataset: {ds_name} ---")
            try:
                result = evaluate_calibration(
                    model      = model,
                    tokenizer  = tokenizer,
                    samples    = samples,
                    model_name = args.model_name,
                    precision  = precision,
                    dataset    = ds_name,
                    device     = args.device,
                    n_bins     = args.n_bins,
                )
            except Exception as e:
                logger.error(f"Calibration failed for {precision}/{ds_name}: {e}")
                continue

            # Save JSON
            tag       = f"{args.model_name}_{precision}_{ds_name}"
            json_path = output_dir / f"{tag}_calibration.json"
            result.save(json_path)

            # Plots
            plot_reliability_diagram(
                result,
                save_path=output_dir / "plots" / f"{tag}_reliability.png",
            )
            plot_dashboard(
                result,
                save_path=output_dir / "plots" / f"{tag}_dashboard.png",
            )

            all_results[ds_name].append(result)
            log_calibration(run, result)

        free_model(model)

    # Cross-config comparison plot per dataset
    for ds_name, results in all_results.items():
        if len(results) > 1:
            plot_metrics_comparison(
                results,
                save_path=output_dir / "plots" / f"{args.model_name}_{ds_name}_comparison.png",
            )

    logger.info("\n=== Sweep complete ===")
    for ds_name, results in all_results.items():
        if not results:
            continue
        logger.info(f"\nDataset: {ds_name}")
        for r in results:
            logger.info(f"  {r.precision:12s} | {r.summary()}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
