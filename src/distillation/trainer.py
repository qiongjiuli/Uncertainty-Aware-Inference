"""
Knowledge Distillation for LLMs.

Approach:
  - Teacher: FP16 model (frozen)
  - Student: INT4 quantized model (lm_head + LoRA adapters trained)
  - Loss: alpha * KL(student || teacher) * T^2 + (1-alpha) * CE(student, labels)
  - Data: WikiText-2 (generic text, next-token prediction)

This targets calibration recovery: by forcing the student's output
distribution to match the teacher's soft distribution, we hope to
restore the calibrated confidence of the FP16 model.

Reference:
  Hinton et al. (2015) - Distilling the Knowledge in a Neural Network
  Kim et al. (2025)    - The Role of Teacher Calibration in KD
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KDConfig:
    temperature  : float = 4.0    # soften distributions (Hinton uses 4-20)
    alpha        : float = 0.7    # weight for KD loss vs CE loss
    lr           : float = 2e-5
    n_epochs     : int   = 3
    batch_size   : int   = 4
    max_length   : int   = 256
    warmup_ratio : float = 0.1
    grad_clip    : float = 1.0
    eval_every   : int   = 100    # steps


# ---------------------------------------------------------------------------
# KD Loss
# ---------------------------------------------------------------------------

class KDLoss(nn.Module):
    """
    Knowledge Distillation loss combining KL divergence (soft) and
    cross-entropy (hard). Operates on next-token logits.

    T^2 rescaling (Hinton 2015): ensures the gradient magnitude of
    the KD term is comparable to the CE term.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,   # (B, seq, vocab) or (B, vocab)
        teacher_logits: torch.Tensor,   # same shape
        labels: torch.Tensor,           # (B, seq) with -100 for ignored
    ) -> tuple[torch.Tensor, dict]:

        # Flatten to (B*seq, vocab) if sequence dim present
        if student_logits.dim() == 3:
            B, S, V = student_logits.shape
            # Only compute loss on non-ignored positions
            mask = labels.reshape(-1) != -100      # (B*S,)
            s_log = student_logits.reshape(-1, V)[mask]
            t_log = teacher_logits.reshape(-1, V)[mask]
            lbl   = labels.reshape(-1)[mask]
        else:
            s_log = student_logits
            t_log = teacher_logits
            lbl   = labels

        if lbl.numel() == 0:
            zero = student_logits.new_tensor(0.0, requires_grad=True)
            return zero, {"kd_loss": 0.0, "ce_loss": 0.0, "total": 0.0}

        # Soft KD loss
        t_soft  = F.softmax(t_log / self.T, dim=-1)
        s_lsoft = F.log_softmax(s_log / self.T, dim=-1)
        kd_loss = F.kl_div(s_lsoft, t_soft, reduction="batchmean") * (self.T ** 2)

        # Hard CE loss
        ce_loss = F.cross_entropy(s_log, lbl)

        total = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

        return total, {
            "kd_loss": kd_loss.item(),
            "ce_loss": ce_loss.item(),
            "total"  : total.item(),
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Plain text dataset for next-token prediction (KD training).
    Tokenizes and chunks into fixed-length windows.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 256,
    ):
        self.max_length = max_length
        self.examples   = []

        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)
            # Chunk into max_length windows with stride max_length//2
            stride = max_length // 2
            for start in range(0, max(1, len(ids) - max_length), stride):
                chunk = ids[start : start + max_length]
                if len(chunk) < 16:    # skip very short chunks
                    continue
                # Pad to max_length
                pad_len = max_length - len(chunk)
                chunk   = chunk + [tokenizer.pad_token_id] * pad_len
                self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ids = self.examples[idx]
        # Labels = shifted ids; pad positions get -100
        labels = ids.clone()
        labels[labels == 0] = -100    # 0 is pad_token_id for LLaMA
        return {"input_ids": ids, "labels": labels}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class KDTrainResult:
    history       : list[dict]
    final_kd_loss : float
    final_ce_loss : float
    n_steps       : int

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def train_kd(
    teacher_model,
    student_model,
    texts: list[str],
    tokenizer,
    config: KDConfig,
    output_dir: str | Path,
    device: str = "cuda",
    wandb_run=None,
) -> KDTrainResult:
    """
    Run KD training.

    Strategy:
      - Teacher is fully frozen.
      - Student: we unfreeze the lm_head (output projection) only.
        This is minimal but directly shapes the output probability
        distribution, which is what calibration measures.
      - If peft is available, add LoRA to q_proj / v_proj for richer
        adaptation without blowing GPU memory.

    Args:
        teacher_model : FP16 model, frozen
        student_model : quantized model to fine-tune
        texts         : list of plain text strings (KD corpus)
        tokenizer     : shared tokenizer
        config        : KDConfig
        output_dir    : where to save the trained student + logs
        device        : "cuda" or "cpu"
        wandb_run     : optional active W&B run for logging
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Freeze teacher ──
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # ── Optionally add LoRA to student ──
    student_model = _maybe_add_lora(student_model)

    # ── Unfreeze lm_head ──
    for name, param in student_model.named_parameters():
        if "lm_head" in name or "lora_" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {n_trainable:,}")

    # ── Dataset ──
    dataset    = TextDataset(texts, tokenizer, config.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    logger.info(f"KD dataset: {len(dataset)} chunks from {len(texts)} texts")

    # ── Optimizer + scheduler ──
    optimizer   = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)
    total_steps = len(dataloader) * config.n_epochs
    warmup_steps= int(total_steps * config.warmup_ratio)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    loss_fn = KDLoss(config.temperature, config.alpha)

    history      = []
    global_step  = 0
    running_loss = {"kd_loss": [], "ce_loss": [], "total": []}

    for epoch in range(config.n_epochs):
        student_model.train()
        logger.info(f"Epoch {epoch+1}/{config.n_epochs}")

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            # Teacher forward (no grad, use autocast for speed)
            with torch.no_grad(), torch.cuda.amp.autocast():
                t_out = teacher_model(input_ids=input_ids)
                t_logits = t_out.logits.detach()   # (B, S, vocab)

            # Student forward
            s_out    = student_model(input_ids=input_ids)
            s_logits = s_out.logits                # (B, S, vocab)

            loss, loss_dict = loss_fn(s_logits, t_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
            optimizer.step()
            scheduler.step()

            for k, v in loss_dict.items():
                running_loss[k].append(v)

            global_step += 1

            if global_step % config.eval_every == 0:
                avgs = {k: np.mean(v[-config.eval_every:])
                        for k, v in running_loss.items()}
                logger.info(
                    f"  Step {global_step:5d} | "
                    + "  ".join(f"{k}={v:.4f}" for k, v in avgs.items())
                )
                step_log = {"step": global_step, **avgs}
                history.append(step_log)

                if wandb_run:
                    wandb_run.log({"train/" + k: v for k, v in avgs.items()},
                                  step=global_step)

    # ── Save student ──
    save_path = output_dir / "student_kd"
    try:
        student_model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        logger.info(f"Student saved to {save_path}")
    except Exception as e:
        logger.warning(f"Could not save student checkpoint: {e}")

    final = {k: float(np.mean(v[-50:])) for k, v in running_loss.items()}
    result = KDTrainResult(
        history       = history,
        final_kd_loss = final.get("kd_loss", 0.0),
        final_ce_loss = final.get("ce_loss", 0.0),
        n_steps       = global_step,
    )
    result.save(output_dir / "kd_train_log.json")
    return result


def _maybe_add_lora(model) -> nn.Module:
    """
    Optionally attach LoRA adapters if peft is available.
    Falls back to lm_head-only if not.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"LoRA added — {n_lora:,} trainable params")
    except ImportError:
        logger.info("peft not installed — training lm_head only")
    return model


# ---------------------------------------------------------------------------
# Recovery analysis
# ---------------------------------------------------------------------------

def compute_kd_recovery(
    fp16_val: float,
    pre_kd_val: float,
    post_kd_val: float,
    metric_name: str,
    higher_is_better: bool = False,
) -> float:
    """
    Compute fraction of calibration quality recovered by KD.

    For error metrics (ECE, Brier): lower is better.
      degradation = pre_kd - fp16    (positive → quant hurt)
      improvement = pre_kd - post_kd (positive → KD helped)
      recovery = improvement / degradation

    For accuracy: higher is better.
      degradation = fp16 - pre_kd
      improvement = post_kd - pre_kd
      recovery = improvement / degradation

    Returns value in [0, 1] where 1 = full recovery, 0 = no improvement.
    Negative means KD made things worse.
    """
    if higher_is_better:
        degradation = fp16_val - pre_kd_val
        improvement = post_kd_val - pre_kd_val
    else:
        degradation = pre_kd_val - fp16_val
        improvement = pre_kd_val - post_kd_val

    if abs(degradation) < 1e-8:
        logger.debug(f"{metric_name}: no degradation, recovery undefined → 1.0")
        return 1.0

    return float(improvement / degradation)
