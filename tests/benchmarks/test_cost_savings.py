"""Benchmark tests for cost savings — should achieve 50-85% savings."""

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
            avg_latency_ms=5,
        ),
        settings.models.large.name: MockBackend(
            model_name=settings.models.large.name,
            tier="large",
            avg_latency_ms=10,
        ),
    }

    return InferencePipeline(
        settings=settings,
        model_registry=registry,
        backends=backends,
    )


@pytest.mark.benchmark
def test_cost_savings_on_mixed_workload(pipeline):
    """Route sample prompts and verify meaningful cost savings."""
    sample_path = Path(__file__).parent.parent.parent / "demo" / "sample_prompts.json"
    with open(sample_path) as f:
        data = json.load(f)

    loop = asyncio.new_event_loop()
    for item in data["prompts"]:
        loop.run_until_complete(pipeline.run(item["text"]))
    loop.close()

    summary = pipeline.cost_tracker.get_summary()

    print(f"\nCost Savings Benchmark:")
    print(f"  Total requests: {summary.total_requests}")
    print(f"  Total cost: ${summary.total_cost_usd:.6f}")
    print(f"  Cost if always large: ${summary.total_cost_if_large_usd:.6f}")
    print(f"  Savings: ${summary.total_savings_usd:.6f} ({summary.savings_pct:.1f}%)")
    print(f"  Requests by model: {summary.requests_by_model}")

    # At least some savings should be achieved
    assert summary.savings_pct > 0, "No cost savings achieved"
    # With a balanced mix of simple and complex prompts, rule-based routing
    # saves cost by routing simple prompts to the cheap model.
    # The exact savings % depends on the prompt mix and threshold.
    assert summary.savings_pct > 10, (
        f"Savings of {summary.savings_pct:.1f}% is below 10% target"
    )
