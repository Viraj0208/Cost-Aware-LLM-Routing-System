"""Tests for router components: feature extractor, routing engine, threshold manager."""

import pytest

from src.config.settings import load_settings
from src.config.model_registry import ModelRegistry
from src.router.feature_extractor import FeatureExtractor
from src.router.routing_engine import RoutingEngine
from src.router.threshold_manager import ThresholdManager


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_simple_prompt_low_score(self):
        features = self.extractor.extract("What is the capital of France?")
        assert features.rule_based_score < 0.5
        assert features.token_count < 30

    def test_complex_prompt_high_score(self):
        prompt = (
            "Write a Python function that implements a binary search tree with "
            "insertion, deletion, and search operations. Explain the time complexity "
            "of each operation step by step and compare it with a hash table. "
            "Also implement unit tests for edge cases."
        )
        features = self.extractor.extract(prompt)
        assert features.rule_based_score > 0.5
        assert features.has_code_markers or features.has_reasoning_markers

    def test_code_detection(self):
        features = self.extractor.extract("def hello_world(): print('hello')")
        assert features.has_code_markers

    def test_math_detection(self):
        features = self.extractor.extract("Calculate the integral of x^2 from 0 to 5")
        assert features.has_math_markers

    def test_reasoning_detection(self):
        features = self.extractor.extract("Explain why the sky is blue step by step")
        assert features.has_reasoning_markers

    def test_multi_part_question(self):
        prompt = "1. What is AI?\n2. How does it work?\n3. What are its applications?"
        features = self.extractor.extract(prompt)
        assert features.question_depth >= 3

    def test_score_bounded(self):
        for prompt in ["hi", "a" * 1000, "explain everything about quantum physics in detail"]:
            features = self.extractor.extract(prompt)
            assert 0.0 <= features.rule_based_score <= 1.0


class TestThresholdManager:
    def test_default_threshold(self):
        tm = ThresholdManager(0.5)
        assert tm.threshold == 0.5

    def test_routing_decision(self):
        tm = ThresholdManager(0.5)
        assert tm.should_route_to_large(0.7)
        assert not tm.should_route_to_large(0.3)
        assert tm.should_route_to_large(0.5)  # Equal to threshold = large

    def test_mode_switching(self):
        tm = ThresholdManager(0.5)
        tm.set_mode("conservative")
        assert tm.threshold == 0.3
        tm.set_mode("aggressive")
        assert tm.threshold == 0.7

    def test_threshold_clamped(self):
        tm = ThresholdManager(0.5)
        tm.threshold = 1.5
        assert tm.threshold == 1.0
        tm.threshold = -0.5
        assert tm.threshold == 0.0


class TestRoutingEngine:
    def setup_method(self):
        self.settings = load_settings(mode="simulation")
        self.registry = ModelRegistry()
        self.registry.load_from_yaml()
        self.engine = RoutingEngine(self.settings, self.registry)

    def test_simple_query_routes_to_small(self):
        decision = self.engine.route("What is 2+2?")
        assert decision.target_tier == "small"
        assert decision.complexity_score < 0.5

    def test_complex_query_routes_to_large(self):
        prompt = (
            "Write a detailed implementation of a distributed consensus algorithm "
            "like Raft in Python. Explain each step of the leader election process "
            "and implement log replication with proper error handling."
        )
        decision = self.engine.route(prompt)
        assert decision.target_tier == "large"
        assert decision.complexity_score >= 0.5

    def test_forced_routing(self):
        decision = self.engine.route("Hello", force_model="llama-2-70b")
        assert decision.target_model == "llama-2-70b"
        assert "forced" in decision.reasoning[0].lower()

    def test_invalid_forced_model_rejected(self):
        with pytest.raises(ValueError, match="Unknown force_model"):
            self.engine.route("Hello", force_model="not-a-model")

    def test_decision_has_cost_estimates(self):
        decision = self.engine.route("What is Python?")
        assert decision.estimated_cost >= 0
        assert decision.cost_if_large >= 0
        assert decision.potential_savings_pct >= 0

    def test_routing_produces_explanations(self):
        decision = self.engine.route("Explain quantum computing step by step")
        assert len(decision.reasoning) >= 1

    def test_routing_latency_reasonable(self):
        decision = self.engine.route("What is the meaning of life?")
        assert decision.routing_latency_ms < 100  # Should be well under 100ms
