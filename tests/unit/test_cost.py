"""Tests for cost calculator and tracker."""

from src.cost.calculator import CostCalculator
from src.cost.tracker import CostTracker


class TestCostCalculator:
    def setup_method(self):
        self.calc = CostCalculator(
            small_model_cost_per_1k=0.002,
            large_model_cost_per_1k=0.06,
            small_model_name="phi-2",
            large_model_name="llama-2-70b",
        )

    def test_small_model_cost(self):
        est = self.calc.calculate("phi-2", prompt_tokens=100, completion_tokens=100)
        assert est.cost_usd == pytest.approx(0.0004)  # 200 tokens * 0.002/1000
        assert est.total_tokens == 200

    def test_large_model_cost(self):
        est = self.calc.calculate("llama-2-70b", prompt_tokens=100, completion_tokens=100)
        assert est.cost_usd == pytest.approx(0.012)  # 200 tokens * 0.06/1000

    def test_savings_calculation(self):
        est = self.calc.calculate("phi-2", prompt_tokens=500, completion_tokens=500)
        assert est.savings_usd > 0
        assert est.savings_pct > 90  # Small model should save >90% vs large

    def test_large_model_no_savings(self):
        est = self.calc.calculate("llama-2-70b", prompt_tokens=100, completion_tokens=100)
        assert est.savings_usd == 0.0
        assert est.savings_pct == 0.0


class TestCostTracker:
    def setup_method(self):
        self.tracker = CostTracker()
        self.calc = CostCalculator()

    def test_record_and_summary(self):
        est = self.calc.calculate("phi-2", 100, 100)
        self.tracker.record(est, complexity_score=0.3)
        summary = self.tracker.get_summary()
        assert summary.total_requests == 1
        assert summary.total_cost_usd > 0
        assert summary.total_tokens == 200

    def test_multiple_records(self):
        for _ in range(10):
            est = self.calc.calculate("phi-2", 100, 100)
            self.tracker.record(est)
        summary = self.tracker.get_summary()
        assert summary.total_requests == 10
        assert "phi-2" in summary.requests_by_model

    def test_mixed_model_tracking(self):
        est_small = self.calc.calculate("phi-2", 100, 100)
        est_large = self.calc.calculate("llama-2-70b", 100, 100)
        self.tracker.record(est_small)
        self.tracker.record(est_large)
        summary = self.tracker.get_summary()
        assert summary.total_requests == 2
        assert len(summary.requests_by_model) == 2
        assert summary.savings_pct > 0  # Mixed usage saves vs all-large

    def test_reset(self):
        est = self.calc.calculate("phi-2", 100, 100)
        self.tracker.record(est)
        self.tracker.reset()
        summary = self.tracker.get_summary()
        assert summary.total_requests == 0
        assert summary.total_cost_usd == 0.0

    def test_recent_events(self):
        for i in range(5):
            est = self.calc.calculate("phi-2", 100 * (i + 1), 100)
            self.tracker.record(est)
        events = self.tracker.get_recent_events(3)
        assert len(events) == 3


import pytest
