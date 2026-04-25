"""Integration tests for the full inference pipeline."""

import asyncio
import json
from pathlib import Path

import pytest

from src.config.settings import load_settings
from src.config.model_registry import ModelRegistry
from src.cost.tracker import CostTracker
from src.inference.pipeline import InferencePipeline
from src.models.mock_backend import MockBackend


@pytest.fixture
def pipeline():
    settings = load_settings(mode="simulation")
    registry = ModelRegistry()
    registry.load_from_yaml()

    backends = {
        settings.models.small.name: MockBackend(
            model_name=settings.models.small.name,
            tier="small",
            avg_latency_ms=10,  # Fast for tests
        ),
        settings.models.large.name: MockBackend(
            model_name=settings.models.large.name,
            tier="large",
            avg_latency_ms=20,
        ),
    }

    return InferencePipeline(
        settings=settings,
        model_registry=registry,
        backends=backends,
    )


def test_simple_prompt_uses_small_model(pipeline):
    result = asyncio.run(pipeline.run("What is 2+2?"))
    assert result.model_tier == "small"
    assert result.text  # Non-empty response
    assert result.cost.cost_usd > 0
    assert result.total_latency_ms > 0


def test_complex_prompt_uses_large_model(pipeline):
    prompt = (
        "Write a Python function that implements a binary search tree with "
        "insertion, deletion, and search operations. Explain the time "
        "complexity step by step."
    )
    result = asyncio.run(pipeline.run(prompt))
    assert result.model_tier == "large"
    assert result.text
    assert len(result.text) > 50  # Large model gives detailed response


def test_forced_model_override(pipeline):
    result = asyncio.run(pipeline.run("Hello", force_model="llama-2-70b"))
    assert result.model_used == "llama-2-70b"


def test_invalid_forced_model_rejected(pipeline):
    with pytest.raises(ValueError, match="Unknown force_model"):
        asyncio.run(pipeline.run("Hello", force_model="not-a-model"))


def test_cost_tracking_accumulates(pipeline):
    for prompt in ["What is Python?", "Define AI.", "Who is Einstein?"]:
        asyncio.run(pipeline.run(prompt))

    summary = pipeline.cost_tracker.get_summary()
    assert summary.total_requests == 3
    assert summary.total_cost_usd > 0


def test_small_model_saves_cost(pipeline):
    result = asyncio.run(pipeline.run("What is the capital of Japan?"))
    assert result.cost.savings_pct > 0  # Should save vs always using large


def test_pipeline_health_check(pipeline):
    health = asyncio.run(pipeline.health_check())
    assert all(status is True for status in health.values())


def test_pipeline_with_sample_prompts(pipeline):
    """Run a batch of sample prompts and verify routing makes sense."""
    sample_path = Path(__file__).parent.parent.parent / "demo" / "sample_prompts.json"
    with open(sample_path) as f:
        data = json.load(f)

    correct = 0
    total = len(data["prompts"])

    for item in data["prompts"]:
        result = asyncio.run(pipeline.run(item["text"]))
        if result.model_tier == item["expected_tier"]:
            correct += 1

    accuracy = correct / total
    # We expect at least 60% accuracy with rule-based routing
    assert accuracy >= 0.6, f"Routing accuracy {accuracy:.0%} is below 60% threshold"
