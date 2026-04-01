"""
tests/test_analysis.py

Unit tests for Pareto dominance and routing simulation.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.pareto import (
    ExperimentPoint,
    find_optimal_threshold,
    is_dominated,
    pareto_front,
    simulate_routing,
)


def _make_point(
    model="M", precision="FP16",
    ece=0.05, brier=0.2, accuracy=0.75,
    tokens_per_sec=100.0, gpu_mem_gb=14.0, latency_ms=10.0,
) -> ExperimentPoint:
    return ExperimentPoint(
        model_name=model, precision=precision, dataset="arc",
        ece=ece, mce=ece, brier=brier, accuracy=accuracy,
        mean_confidence=0.8,
        tokens_per_sec=tokens_per_sec, gpu_mem_gb=gpu_mem_gb, latency_ms=latency_ms,
    )


class TestPareto:
    def test_dominated_point_identified(self):
        """Point strictly worse on all objectives should be dominated."""
        good = _make_point("M", "FP16",    ece=0.03, accuracy=0.80, tokens_per_sec=200, gpu_mem_gb=10)
        bad  = _make_point("M", "INT4",    ece=0.07, accuracy=0.70, tokens_per_sec=100, gpu_mem_gb=20)
        assert is_dominated(bad, [good, bad])

    def test_pareto_optimal_not_dominated(self):
        """A point on the Pareto front should not be dominated."""
        p1 = _make_point("M", "A", ece=0.03, accuracy=0.80, tokens_per_sec=100, gpu_mem_gb=20)
        p2 = _make_point("M", "B", ece=0.07, accuracy=0.85, tokens_per_sec=200, gpu_mem_gb=10)
        # p1 is better on ECE and GPU memory; p2 is better on accuracy and speed
        # Neither dominates the other
        assert not is_dominated(p1, [p1, p2])
        assert not is_dominated(p2, [p1, p2])

    def test_pareto_front_size(self):
        """Pareto front should be a subset of all points."""
        points = [
            _make_point("M", "A", ece=0.03, accuracy=0.80),
            _make_point("M", "B", ece=0.07, accuracy=0.70),   # dominated by A
            _make_point("M", "C", ece=0.06, accuracy=0.85),   # better accuracy than A
        ]
        pf = pareto_front(points)
        assert len(pf) <= len(points)
        # B should be dominated
        assert points[1] not in pf

    def test_single_point_is_pareto(self):
        p = _make_point()
        assert pareto_front([p]) == [p]

    def test_all_identical_are_all_pareto(self):
        """If all points are identical, none dominates the others."""
        points = [_make_point() for _ in range(5)]
        pf = pareto_front(points)
        assert len(pf) == 5


class TestRoutingSimulation:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.N = 1000
        self.confidences   = rng.beta(4, 2, self.N)       # mostly high conf
        self.correct_quant = (rng.uniform(size=self.N) < 0.70).astype(float)
        self.correct_fp16  = (rng.uniform(size=self.N) < 0.75).astype(float)

    def test_output_length(self):
        routing = simulate_routing(
            self.confidences, self.correct_quant, self.correct_fp16,
            ece_quant=0.06, ece_fp16=0.03,
            thresholds=np.linspace(0.1, 0.9, 20),
        )
        assert len(routing) == 20

    def test_frac_cheap_decreases_with_threshold(self):
        """Higher threshold → fewer queries routed to cheap model."""
        routing = simulate_routing(
            self.confidences, self.correct_quant, self.correct_fp16,
            ece_quant=0.06, ece_fp16=0.03,
        )
        fracs = [r.frac_cheap for r in routing]
        # Should be monotonically non-increasing
        assert all(fracs[i] >= fracs[i+1] - 1e-8 for i in range(len(fracs)-1))

    def test_zero_threshold_all_cheap(self):
        routing = simulate_routing(
            self.confidences, self.correct_quant, self.correct_fp16,
            ece_quant=0.06, ece_fp16=0.03,
            thresholds=np.array([0.0]),
        )
        assert routing[0].frac_cheap == pytest.approx(1.0, abs=1e-6)

    def test_threshold_one_all_expensive(self):
        routing = simulate_routing(
            self.confidences, self.correct_quant, self.correct_fp16,
            ece_quant=0.06, ece_fp16=0.03,
            thresholds=np.array([1.0]),
        )
        assert routing[0].frac_cheap == pytest.approx(0.0, abs=0.01)

    def test_cost_saving_positive_for_low_threshold(self):
        routing = simulate_routing(
            self.confidences, self.correct_quant, self.correct_fp16,
            ece_quant=0.06, ece_fp16=0.03,
            thresholds=np.array([0.3]),
        )
        assert routing[0].cost_saving_pct > 0

    def test_find_optimal_threshold_respects_tolerances(self):
        routing = simulate_routing(
            self.confidences, self.correct_quant, self.correct_fp16,
            ece_quant=0.06, ece_fp16=0.03,
        )
        fp16_acc = float(self.correct_fp16.mean())
        fp16_ece = 0.03
        optimal  = find_optimal_threshold(
            routing, fp16_acc=fp16_acc, fp16_ece=fp16_ece,
            acc_tolerance=0.02, ece_tolerance=0.01,
        )
        if optimal:
            assert (fp16_acc - optimal.effective_acc) <= 0.02 + 1e-6
            assert (optimal.effective_ece - fp16_ece)  <= 0.01 + 1e-6
