#!/usr/bin/env python
"""
scripts/run_qat.py  [STRETCH GOAL]

Quantization-Aware Training on Llama-2 7B at INT4.
Compares calibration of QAT model vs PTQ GPTQ_INT4 config.

Approach:
  - Load FP16 model
  - Apply bitsandbytes NF4 quantization with LoRA adapters (QLoRA-style)
  - Fine-tune on a small dataset (1-2K samples from WikiText-2)
  - Evaluate calibration and compare to PTQ result

This tests whether training-time quantization awareness inherently
preserves calibration better than post-hoc PTQ methods.

Reference:
  Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023)

NOTE: This requires ~20 GB VRAM (A100 recommended). Estimated: 8-12 GPU-hours.

Usage:
    python scripts/run_qat.py \
        --model_id   meta-llama/Llama-2-7b-hf \
        --model_name Llama-2-7B \
        --results_dir results/llama2_7b \
        --output_dir  results/llama2_7b/qat \
        --n_train_samples 2000 \
        --n_epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from src.calibration.datasets   import load_dataset_mcq, load_kd_corpus
from src.calibration.metrics    import CalibrationResult, evaluate_calibration
from src.calibration.plots      import plot_metrics_comparison
from src.distillation.trainer   import TextDataset, compute_kd_recovery
from src.utils.logging          import init_wandb, log_calibration, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QAT training loop (QLoRA-style fine-tuning)
# ---------------------------------------------------------------------------

def load_qlora_model(model_id: str):
    """
    Load model in NF4 + LoRA configuration (QLoRA-style).
    This is our QAT setup: the model is quantized during training,
    so the forward pass sees INT4 weights throughout.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        raise ImportError("Install peft: pip install peft")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config = bnb_config,
        device_map          = "auto",
        low_cpu_mem_usage   = True,
    )

    # Prepare model for k-bit training (handles gradient checkpointing, etc.)
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = 16,
        lora_alpha     = 32,
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def train_qat(
    model,
    tokenizer,
    texts       : list[str],
    output_dir  : Path,
    n_epochs    : int   = 3,
    batch_size  : int   = 4,
    lr          : float = 2e-4,    # QLoRA uses higher LR than KD
    max_length  : int   = 256,
    device      : str   = "cuda",
    wandb_run   = None,
) -> list[dict]:
    """Standard causal LM fine-tuning on INT4 model + LoRA adapters."""
    import torch.nn as nn
    import numpy as np

    dataset    = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    trainable  = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps= len(dataloader) * n_epochs
    scheduler  = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )

    history     = []
    global_step = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        logger.info(f"QAT Epoch {epoch+1}/{n_epochs}")

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = np.mean(epoch_losses[-50:])
                logger.info(f"  Step {global_step:5d}  loss={avg_loss:.4f}")
                if wandb_run:
                    wandb_run.log({"qat/loss": avg_loss}, step=global_step)

        history.append({"epoch": epoch + 1, "loss": float(np.mean(epoch_losses))})

    # Save
    save_path = output_dir / "qat_checkpoint"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info(f"QAT checkpoint saved: {save_path}")
    return history


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_qat_vs_ptq(
    fp16_result   : CalibrationResult,
    ptq_result    : CalibrationResult,
    qat_result    : CalibrationResult,
    save_path     : Path,
) -> None:
    import matplotlib.pyplot as plt

    metrics = ["ece", "mce", "brier", "accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    for ax, m in zip(axes, metrics):
        vals   = [getattr(fp16_result, m),
                  getattr(ptq_result,  m),
                  getattr(qat_result,  m)]
        colors = ["steelblue", "coral", "seagreen"]
        bars   = ax.bar(["FP16", "PTQ\n(GPTQ_INT4)", "QAT\n(NF4+LoRA)"],
                        vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", fontsize=9)
        ax.set_title(m.upper(), fontsize=12)
        ax.set_ylabel(m)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"QAT vs PTQ Calibration Comparison\n"
        f"{fp16_result.model_name} | Dataset: {fp16_result.dataset}",
        fontsize=13, fontweight="bold",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"QAT vs PTQ plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QAT stretch goal experiment")
    p.add_argument("--model_id",        required=True)
    p.add_argument("--model_name",      required=True)
    p.add_argument("--results_dir",     required=True,
                   help="Dir with pre-computed PTQ calibration JSONs")
    p.add_argument("--output_dir",      default="results/qat")
    p.add_argument("--eval_dataset",    default="arc_challenge",
                   choices=["arc_challenge", "hellaswag", "triviaqa"])
    p.add_argument("--n_eval_samples",  type=int, default=500)
    p.add_argument("--n_train_samples", type=int, default=2000)
    p.add_argument("--n_epochs",        type=int, default=3)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--wandb_project",   default="uncertainty-aware-inference")
    p.add_argument("--no_wandb",        action="store_true")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs")

    logger.warning(
        "STRETCH GOAL: QAT requires ~20 GB VRAM and 8-12 GPU hours. "
        "Ensure you have sufficient GPU budget before proceeding."
    )

    run = None
    if not args.no_wandb:
        run = init_wandb(
            project = args.wandb_project,
            name    = f"{args.model_name}-qat",
            tags    = [args.model_name, "qat", "stretch-goal"],
            config  = vars(args),
        )

    results_dir = Path(args.results_dir)

    # ── Load pre-computed PTQ results ──
    fp16_path = results_dir / f"{args.model_name}_FP16_{args.eval_dataset}_calibration.json"
    ptq_path  = results_dir / f"{args.model_name}_GPTQ_INT4_{args.eval_dataset}_calibration.json"

    if not fp16_path.exists() or not ptq_path.exists():
        raise FileNotFoundError(
            "Run run_ptq_sweep.py first to generate FP16 and GPTQ_INT4 calibration results."
        )

    fp16_result = CalibrationResult.load(fp16_path)
    ptq_result  = CalibrationResult.load(ptq_path)
    logger.info(f"FP16 baseline: {fp16_result.summary()}")
    logger.info(f"PTQ  baseline: {ptq_result.summary()}")

    # ── Load model in QAT config ──
    logger.info("\nLoading model for QAT (NF4 + LoRA) ...")
    model, tokenizer = load_qlora_model(args.model_id)

    # ── Training corpus ──
    texts = load_kd_corpus(n_samples=args.n_train_samples)

    # ── Fine-tune ──
    logger.info("\nStarting QAT fine-tuning ...")
    history = train_qat(
        model      = model,
        tokenizer  = tokenizer,
        texts      = texts,
        output_dir = output_dir,
        n_epochs   = args.n_epochs,
        lr         = args.lr,
        batch_size = args.batch_size,
        device     = args.device,
        wandb_run  = run,
    )

    # ── Evaluate calibration ──
    logger.info("\nEvaluating post-QAT calibration ...")
    eval_samples = load_dataset_mcq(args.eval_dataset, n_samples=args.n_eval_samples)

    qat_result = evaluate_calibration(
        model      = model,
        tokenizer  = tokenizer,
        samples    = eval_samples,
        model_name = args.model_name,
        precision  = "QAT_NF4_LoRA",
        dataset    = args.eval_dataset,
        device     = args.device,
    )
    qat_result.save(
        output_dir / f"{args.model_name}_QAT_NF4_LoRA_{args.eval_dataset}_calibration.json"
    )
    log_calibration(run, qat_result)

    # ── Comparison ──
    metrics_cfg = {"ece": False, "mce": False, "brier": False, "accuracy": True}
    logger.info("\n=== QAT vs PTQ vs FP16 Comparison ===")
    for m, higher_better in metrics_cfg.items():
        ptq_rec = compute_kd_recovery(
            fp16_val=getattr(fp16_result, m),
            pre_kd_val=getattr(ptq_result, m),
            post_kd_val=getattr(qat_result, m),
            metric_name=m,
            higher_is_better=higher_better,
        )
        logger.info(
            f"  {m:12s}: FP16={getattr(fp16_result,m):.4f}  "
            f"PTQ={getattr(ptq_result,m):.4f}  "
            f"QAT={getattr(qat_result,m):.4f}  "
            f"(QAT recovery vs PTQ: {ptq_rec*100:.1f}%)"
        )

    # Save comparison
    comparison = {
        m: {
            "fp16": getattr(fp16_result, m),
            "ptq" : getattr(ptq_result,  m),
            "qat" : getattr(qat_result,  m),
        }
        for m in ["ece", "mce", "brier", "accuracy"]
    }
    with open(output_dir / "qat_vs_ptq_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    plot_qat_vs_ptq(
        fp16_result, ptq_result, qat_result,
        save_path=output_dir / "plots" / "qat_vs_ptq.png",
    )

    if run:
        run.log({f"comparison/{m}/qat": v["qat"] for m, v in comparison.items()})
        run.finish()

    logger.info(f"\nQAT outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
