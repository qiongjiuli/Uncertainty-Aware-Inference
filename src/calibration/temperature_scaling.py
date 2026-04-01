"""
src/calibration/temperature_scaling.py

Post-hoc calibration via temperature scaling (Guo et al. ICML 2017).

Temperature scaling fits a single scalar T on a held-out validation set
such that the scaled logits z/T minimize NLL. This is the simplest and
often most effective post-hoc recalibration method.

Used as a baseline: if a quantized model's calibration can be fully
recovered by temperature scaling alone (no KD needed), that is an
important null result.

Reference:
  Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
  https://arxiv.org/abs/1706.04599
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temperature scaling model
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    Wraps a set of logits and scales them by a learnable scalar T > 0.
    T > 1 → softer (lower confidence, better calibrated if overconfident)
    T < 1 → sharper (higher confidence, better calibrated if underconfident)
    """

    def __init__(self):
        super().__init__()
        # Initialize T = 1.5 (slightly > 1 since neural nets are often overconfident)
        self.temperature = nn.Parameter(torch.tensor([1.5]))

    @property
    def T(self) -> float:
        return float(self.temperature.item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature.clamp(min=0.01)

    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Find optimal temperature by minimising NLL on held-out logits.

        Args:
            logits : raw (unscaled) logits  (N, C)  — e.g. choice log-scores
            labels : integer class indices  (N,)
            lr     : learning rate for LBFGS
            max_iter: max LBFGS iterations

        Returns:
            Optimal temperature T*
        """
        self.train()
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer     = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled = self.forward(logits_t)
            loss   = nll_criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        self.eval()

        logger.info(f"Temperature scaling: T* = {self.T:.4f}")
        return self.T

    def scale_probs(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply calibrated temperature and return softmax probabilities.
        """
        self.eval()
        with torch.no_grad():
            t    = torch.tensor(logits, dtype=torch.float32)
            sc   = self.forward(t)
            prob = torch.softmax(sc, dim=-1)
        return prob.numpy()


# ---------------------------------------------------------------------------
# Fit and evaluate
# ---------------------------------------------------------------------------

@dataclass
class TemperatureScalingResult:
    precision        : str
    temperature      : float
    ece_before       : float
    ece_after        : float
    mce_before       : float
    mce_after        : float
    brier_before     : float
    brier_after      : float
    ece_improvement  : float   # ece_before - ece_after (positive = better)


def fit_temperature_scaling(
    logits        : np.ndarray,   # (N, C) raw log-scores before softmax
    labels        : np.ndarray,   # (N,)
    precision     : str,
    val_fraction  : float = 0.5,
    seed          : int   = 42,
) -> tuple[TemperatureScaler, TemperatureScalingResult]:
    """
    Split logits into fit / eval halves.
    Fit temperature on first half, report calibration improvement on second half.

    Returns (fitted scaler, result dict).
    """
    from .metrics import compute_ece, compute_mce, compute_brier

    N   = len(logits)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    split = N // 2
    fit_idx  = idx[:split]
    eval_idx = idx[split:]

    scaler = TemperatureScaler()

    # Fit on first half
    scaler.calibrate(logits[fit_idx], labels[fit_idx])

    # Evaluate on second half
    probs_before = _softmax(logits[eval_idx])
    probs_after  = scaler.scale_probs(logits[eval_idx])
    lbl          = labels[eval_idx]

    conf_b  = probs_before.max(axis=1)
    conf_a  = probs_after.max(axis=1)
    pred_b  = probs_before.argmax(axis=1)
    correct = (pred_b == lbl).astype(float)

    ece_b, bc_b, ba_b, bn_b = compute_ece(conf_b, correct)
    mce_b = compute_mce(bc_b, ba_b, bn_b)

    ece_a, bc_a, ba_a, bn_a = compute_ece(conf_a, correct)
    mce_a = compute_mce(bc_a, ba_a, bn_a)

    brier_b, *_ = compute_brier(probs_before, lbl)
    brier_a, *_ = compute_brier(probs_after,  lbl)

    result = TemperatureScalingResult(
        precision       = precision,
        temperature     = scaler.T,
        ece_before      = ece_b,
        ece_after       = ece_a,
        mce_before      = mce_b,
        mce_after       = mce_a,
        brier_before    = brier_b,
        brier_after     = brier_a,
        ece_improvement = ece_b - ece_a,
    )

    logger.info(
        f"[TS] {precision}: T={scaler.T:.3f} | "
        f"ECE {ece_b:.4f} → {ece_a:.4f} "
        f"({'↓ improved' if result.ece_improvement > 0 else '↑ worse'})"
    )
    return scaler, result


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
