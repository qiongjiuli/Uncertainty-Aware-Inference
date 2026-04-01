#!/usr/bin/env python
"""
scripts/run_profiling.py

Profiles each quantization config for one model:
latency, memory, and Roofline characterization.

Usage:
    python scripts/run_profiling.py \
        --model_id meta-llama/Llama-2-7b-hf \
        --model_name Llama-2-7B \
        --output_dir results/profiling \
        --precisions FP16 GPTQ_INT4 AWQ_INT4 NF4
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.profiling.harness    import ProfilingResult, plot_roofline, profile_model
from src.quantization.loaders import LOADERS, free_model, load_model
from src.utils.logging        import init_wandb, log_profiling, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",    required=True)
    p.add_argument("--model_name",  required=True)
    p.add_argument("--output_dir",  default="results/profiling")
    p.add_argument("--precisions",  nargs="+", default=list(LOADERS.keys()))
    p.add_argument("--device",      default="cuda")
    p.add_argument("--no_profiler_trace", action="store_true",
                   help="Skip chrome trace (faster, less disk)")
    p.add_argument("--wandb_project", default="uncertainty-aware-inference")
    p.add_argument("--no_wandb",    action="store_true")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs")

    run = None
    if not args.no_wandb:
        run = init_wandb(
            project = args.wandb_project,
            name    = f"{args.model_name}-profiling",
            tags    = [args.model_name, "profiling"],
            config  = vars(args),
        )

    all_results: list[ProfilingResult] = []

    for precision in args.precisions:
        logger.info(f"\n{'='*55}\n  {precision}\n{'='*55}")
        try:
            model, tokenizer = load_model(args.model_id, precision)
        except Exception as e:
            logger.error(f"Load failed [{precision}]: {e}")
            continue

        try:
            result = profile_model(
                model       = model,
                tokenizer   = tokenizer,
                model_name  = args.model_name,
                precision   = precision,
                output_dir  = output_dir,
                device      = args.device,
                run_profiler= not args.no_profiler_trace,
            )
            all_results.append(result)
            log_profiling(run, result)
        except Exception as e:
            logger.error(f"Profiling failed [{precision}]: {e}")
        finally:
            free_model(model)

    # Roofline plot
    if len(all_results) > 1:
        plot_roofline(
            all_results,
            save_path=output_dir / f"{args.model_name}_roofline.png",
        )

    logger.info("\n=== Profiling Summary ===")
    for r in all_results:
        logger.info(r.summary())

    if run:
        run.finish()


if __name__ == "__main__":
    main()
