"""
Calibration metrics for LLMs.

KEY DESIGN: For multiple-choice tasks, we score each answer option
via log-likelihood under the model, then softmax over choices.
This gives a proper probability distribution over a small label set,
making ECE / Brier / reliability diagrams well-defined.

References:
  Guo et al. (ICML 2017) - ECE definition
  Brown et al. (2020)    - LLM log-likelihood scoring for MCQ
  Brier (1950)           - Brier score decomposition
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    model_name : str
    precision  : str
    dataset    : str
    n_samples  : int

    # Calibration
    ece        : float = 0.0
    mce        : float = 0.0
    brier      : float = 0.0
    brier_reliability : float = 0.0
    brier_resolution  : float = 0.0
    brier_uncertainty : float = 0.0

    # Accuracy
    accuracy          : float = 0.0
    mean_confidence   : float = 0.0

    # Entropy
    entropy_correct   : float = 0.0
    entropy_incorrect : float = 0.0
    mean_entropy      : float = 0.0

    # Bin data (for reliability diagram)
    bin_confidences : list[float] = field(default_factory=list)
    bin_accuracies  : list[float] = field(default_factory=list)
    bin_counts      : list[int]   = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        with open(path) as f:
            return cls(**json.load(f))

    def summary(self) -> str:
        return (
            f"{self.model_name} [{self.precision}] on {self.dataset} "
            f"(n={self.n_samples})\n"
            f"  ECE={self.ece:.4f}  MCE={self.mce:.4f}  "
            f"Brier={self.brier:.4f}  Acc={self.accuracy:.4f}"
        )


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected Calibration Error with equal-width bins.

    Args:
        confidences : max predicted probability per sample  (N,)
        correct     : 1.0 if prediction correct, 0.0 else  (N,)
        n_bins      : number of equal-width bins

    Returns:
        ece, bin_confidences, bin_accuracies, bin_counts
    """
    assert len(confidences) == len(correct), "Shape mismatch"
    assert np.all((confidences >= 0) & (confidences <= 1)), "Confidences out of [0,1]"

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf  = np.zeros(n_bins)
    bin_acc   = np.zeros(n_bins)
    bin_cnt   = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include right edge in the last bin
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() > 0:
            bin_conf[i] = confidences[mask].mean()
            bin_acc[i]  = correct[mask].mean()
            bin_cnt[i]  = mask.sum()

    N = len(confidences)
    ece = float(
        (np.abs(bin_acc - bin_conf) * bin_cnt / N).sum()
    )
    return ece, bin_conf, bin_acc, bin_cnt


def compute_mce(
    bin_confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_counts: np.ndarray,
) -> float:
    """Maximum Calibration Error — worst-case bin gap."""
    nonempty = bin_counts > 0
    if nonempty.sum() == 0:
        return 0.0
    return float(
        np.abs(bin_accuracies[nonempty] - bin_confidences[nonempty]).max()
    )


def compute_brier(
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Brier score and Murphy (1973) decomposition.

    Args:
        probs  : predicted probability vectors  (N, C)
        labels : integer class indices          (N,)

    Returns:
        (brier, reliability, resolution, uncertainty)
        brier = reliability - resolution + uncertainty
    """
    N, C = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0

    brier = float(((probs - one_hot) ** 2).sum(axis=1).mean())

    # Climatology
    p_bar = one_hot.mean(axis=0)                          # (C,)
    uncertainty = float((p_bar * (1.0 - p_bar)).sum())

    # Group by predicted class for reliability / resolution
    pred_classes = probs.argmax(axis=1)
    reliability = resolution = 0.0
    for c in range(C):
        mask = pred_classes == c
        if mask.sum() == 0:
            continue
        n_k  = mask.sum()
        p_k  = probs[mask].mean(axis=0)
        o_k  = one_hot[mask].mean(axis=0)
        reliability += n_k * ((p_k - o_k) ** 2).sum()
        resolution  += n_k * ((o_k - p_bar) ** 2).sum()

    reliability /= N
    resolution  /= N
    return brier, float(reliability), float(resolution), float(uncertainty)


def compute_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-sample Shannon entropy (nats)."""
    return -(probs * np.log(probs + eps)).sum(axis=1)


# ---------------------------------------------------------------------------
# LLM multiple-choice log-likelihood scorer
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_choices_lm(
    model,
    tokenizer,
    question: str,
    choices: list[str],
    device: str = "cuda",
    context_prefix: str = "Question: {q}\nAnswer: ",
) -> np.ndarray:
    """
    Score each answer choice by its average token log-likelihood
    under the language model, conditioned on the question.

    This is the standard approach used in GPT-3, LLaMA eval harnesses.

    Returns:
        probs : softmax over choice log-likelihoods  (n_choices,)
    """
    prefix = context_prefix.format(q=question)
    scores = []

    for choice in choices:
        full_text     = prefix + choice
        prefix_ids    = tokenizer.encode(prefix,    add_special_tokens=True)
        full_ids      = tokenizer.encode(full_text, add_special_tokens=True)

        # Tokens belonging to the choice (exclude prefix)
        choice_start  = len(prefix_ids)
        if choice_start >= len(full_ids):
            scores.append(-1e9)
            continue

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        outputs   = model(input_ids=input_ids)
        logits    = outputs.logits[0]                 # (seq_len, vocab)

        # Log-probs for each token predicted at position t: logits[t-1] → token[t]
        log_probs = F.log_softmax(logits, dim=-1)

        # Sum log-prob over choice tokens only
        choice_ids    = full_ids[choice_start:]
        choice_logps  = [
            log_probs[choice_start - 1 + i, tok].item()
            for i, tok in enumerate(choice_ids)
        ]
        # Average (not sum) to avoid length bias
        scores.append(np.mean(choice_logps))

    scores_arr = np.array(scores, dtype=np.float32)
    # Softmax over choice scores → probability distribution
    scores_arr -= scores_arr.max()    # numerical stability
    probs = np.exp(scores_arr)
    probs /= probs.sum()
    return probs


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_calibration(
    model,
    tokenizer,
    samples: list[dict],
    model_name: str,
    precision: str,
    dataset: str,
    device: str = "cuda",
    n_bins: int = 15,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Run full calibration evaluation over a list of MCQ samples.

    Each sample dict must have:
        "question" : str
        "choices"  : list[str]
        "answer"   : int   (index of correct choice)

    Args:
        model      : HuggingFace causal LM (already loaded and on device)
        tokenizer  : corresponding tokenizer
        samples    : list of MCQ sample dicts
        model_name : display name (e.g. "Llama-2-7B")
        precision  : config label  (e.g. "GPTQ_INT4")
        dataset    : dataset name  (e.g. "arc_challenge")
        device     : "cuda" or "cpu"
        n_bins     : ECE bins

    Returns:
        CalibrationResult with all metrics populated.
    """
    model.eval()

    all_probs   = []    # (N, n_choices)
    all_labels  = []    # (N,)

    for i, sample in enumerate(samples):
        if verbose and i % 100 == 0:
            print(f"  [{precision}] {i}/{len(samples)}")

        probs = score_choices_lm(
            model, tokenizer,
            sample["question"],
            sample["choices"],
            device=device,
        )
        all_probs.append(probs)
        all_labels.append(sample["answer"])

    # Pad to same n_choices (in case datasets have variable options)
    max_choices = max(p.shape[0] for p in all_probs)
    padded = np.zeros((len(all_probs), max_choices), dtype=np.float32)
    for i, p in enumerate(all_probs):
        padded[i, :len(p)] = p
        if len(p) < max_choices:
            # Renormalize if needed
            padded[i] /= padded[i].sum()

    all_probs  = padded
    all_labels = np.array(all_labels, dtype=int)

    confidences = all_probs.max(axis=1)             # (N,)
    predictions = all_probs.argmax(axis=1)          # (N,)
    correct     = (predictions == all_labels).astype(float)  # (N,)

    ece, bin_conf, bin_acc, bin_cnt = compute_ece(confidences, correct, n_bins)
    mce = compute_mce(bin_conf, bin_acc, bin_cnt)
    brier, reliability, resolution, uncertainty = compute_brier(all_probs, all_labels)

    entropies = compute_entropy(all_probs)
    ent_correct   = float(entropies[correct == 1].mean()) if correct.sum()   > 0 else 0.0
    ent_incorrect = float(entropies[correct == 0].mean()) if (1-correct).sum() > 0 else 0.0

    result = CalibrationResult(
        model_name        = model_name,
        precision         = precision,
        dataset           = dataset,
        n_samples         = len(samples),
        ece               = ece,
        mce               = mce,
        brier             = brier,
        brier_reliability = reliability,
        brier_resolution  = resolution,
        brier_uncertainty = uncertainty,
        accuracy          = float(correct.mean()),
        mean_confidence   = float(confidences.mean()),
        entropy_correct   = ent_correct,
        entropy_incorrect = ent_incorrect,
        mean_entropy      = float(entropies.mean()),
        bin_confidences   = bin_conf.tolist(),
        bin_accuracies    = bin_acc.tolist(),
        bin_counts        = bin_cnt.tolist(),
    )

    if verbose:
        print(f"\n{result.summary()}\n")

    return result
