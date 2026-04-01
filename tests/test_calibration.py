"""
tests/test_calibration.py

Unit tests for calibration metrics.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.metrics import (
    compute_brier,
    compute_ece,
    compute_entropy,
    compute_mce,
)
from src.calibration.temperature_scaling import TemperatureScaler, fit_temperature_scaling


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration(self):
        """A model that is 70% confident and 70% accurate should have ECE ≈ 0."""
        N = 1000
        rng = np.random.default_rng(0)
        confidences = np.full(N, 0.7)
        correct     = (rng.uniform(size=N) < 0.7).astype(float)
        ece, *_ = compute_ece(confidences, correct, n_bins=10)
        assert ece < 0.05, f"Expected ECE < 0.05 for well-calibrated model, got {ece:.4f}"

    def test_overconfident(self):
        """A model that is always 100% confident but only 50% accurate should have ECE ≈ 0.5."""
        N = 1000
        rng = np.random.default_rng(1)
        confidences = np.ones(N)
        correct     = (rng.uniform(size=N) < 0.5).astype(float)
        ece, *_ = compute_ece(confidences, correct, n_bins=10)
        assert 0.4 < ece < 0.6, f"Expected ECE ≈ 0.5 for overconfident model, got {ece:.4f}"

    def test_bin_counts_sum_to_n(self):
        N = 500
        rng = np.random.default_rng(2)
        confidences = rng.uniform(0, 1, N)
        correct     = (rng.uniform(size=N) < confidences).astype(float)
        _, _, _, bin_counts = compute_ece(confidences, correct, n_bins=15)
        assert bin_counts.sum() == N

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_ece(np.array([0.5, 0.6]), np.array([1.0]), n_bins=5)

    def test_output_range(self):
        N = 200
        rng = np.random.default_rng(3)
        confidences = rng.uniform(0, 1, N)
        correct     = rng.integers(0, 2, N).astype(float)
        ece, bin_conf, bin_acc, _ = compute_ece(confidences, correct, n_bins=15)
        assert 0.0 <= ece <= 1.0
        assert np.all(bin_conf >= 0) and np.all(bin_conf <= 1)
        assert np.all(bin_acc  >= 0) and np.all(bin_acc  <= 1)


# ---------------------------------------------------------------------------
# MCE
# ---------------------------------------------------------------------------

class TestMCE:
    def test_mce_geq_ece(self):
        """MCE must be >= ECE (it's the maximum, ECE is the average)."""
        N = 500
        rng = np.random.default_rng(4)
        confidences = rng.uniform(0, 1, N)
        correct     = (rng.uniform(size=N) < confidences).astype(float)
        ece, bc, ba, bn = compute_ece(confidences, correct)
        mce = compute_mce(bc, ba, bn)
        assert mce >= ece - 1e-8

    def test_empty_bins(self):
        """Should handle bins with zero count."""
        bin_conf  = np.array([0.1, 0.0, 0.9])
        bin_acc   = np.array([0.1, 0.0, 0.7])
        bin_cnt   = np.array([100,  0, 100])
        mce = compute_mce(bin_conf, bin_acc, bin_cnt)
        assert mce == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

class TestBrier:
    def test_perfect_model(self):
        """Perfect predictions should give Brier = 0."""
        N = 100
        labels = np.arange(N) % 4           # 4-class problem
        probs  = np.zeros((N, 4))
        probs[np.arange(N), labels] = 1.0   # one-hot
        brier, *_ = compute_brier(probs, labels)
        assert brier < 1e-8

    def test_random_model(self):
        """Uniform random model on 4 classes should have Brier ≈ 0.75."""
        N = 10000
        rng    = np.random.default_rng(5)
        labels = rng.integers(0, 4, N)
        probs  = np.ones((N, 4)) / 4.0
        brier, *_ = compute_brier(probs, labels)
        assert abs(brier - 0.75) < 0.05

    def test_decomposition_identity(self):
        """Brier = reliability - resolution + uncertainty."""
        N = 500
        rng    = np.random.default_rng(6)
        labels = rng.integers(0, 3, N)
        probs  = rng.dirichlet(np.ones(3), size=N)
        brier, rel, res, unc = compute_brier(probs, labels)
        reconstructed = rel - res + unc
        assert abs(brier - reconstructed) < 1e-5


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_entropy(self):
        """Uniform distribution over C classes should have entropy = log(C)."""
        C     = 4
        probs = np.full((10, C), 1.0 / C)
        ent   = compute_entropy(probs)
        expected = np.log(C)
        assert np.allclose(ent, expected, atol=1e-5)

    def test_deterministic_entropy(self):
        """One-hot distribution should have entropy ≈ 0."""
        probs = np.zeros((10, 4))
        probs[:, 0] = 1.0
        ent = compute_entropy(probs)
        assert np.allclose(ent, 0.0, atol=1e-4)

    def test_entropy_nonnegative(self):
        rng   = np.random.default_rng(7)
        probs = rng.dirichlet(np.ones(5), size=100)
        ent   = compute_entropy(probs)
        assert np.all(ent >= -1e-8)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

class TestTemperatureScaling:
    def _make_overconfident_logits(self, N=500, C=4, seed=42):
        rng    = np.random.default_rng(seed)
        labels = rng.integers(0, C, N)
        # Overconfident: large logit for predicted class
        logits = rng.normal(0, 0.5, (N, C))
        logits[np.arange(N), labels] += 5.0   # push confidence up
        return logits, labels

    def test_temperature_reduces_ece_overconfident(self):
        """Fitting T on overconfident logits should reduce ECE."""
        logits, labels = self._make_overconfident_logits()
        scaler, result = fit_temperature_scaling(logits, labels, precision="test")
        assert result.ece_improvement > 0, (
            f"Expected ECE improvement, got {result.ece_improvement:.4f}"
        )

    def test_temperature_gt_1_for_overconfident(self):
        """Overconfident model should require T > 1 to soften probabilities."""
        logits, labels = self._make_overconfident_logits()
        scaler, result = fit_temperature_scaling(logits, labels, precision="test")
        assert result.temperature > 1.0, (
            f"Expected T > 1 for overconfident model, got T={result.temperature:.3f}"
        )

    def test_scale_probs_sums_to_one(self):
        logits, _ = self._make_overconfident_logits(N=20)
        scaler = TemperatureScaler()
        probs  = scaler.scale_probs(logits)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
