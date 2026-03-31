"""Quick demo runner — demonstrates the routing system without starting a server.

Usage:
    python -m scripts.run_demo
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_settings
from src.config.model_registry import ModelRegistry
from src.cost.tracker import CostTracker
from src.inference.pipeline import InferencePipeline
from src.models.mock_backend import MockBackend


DEMO_PROMPTS = [
    ("Simple", "What is the capital of France?"),
    ("Simple", "Define photosynthesis."),
    ("Simple", "Hello, how are you?"),
    ("Complex", "Write a Python implementation of Dijkstra's shortest path algorithm with type hints and explain the time complexity step by step."),
    ("Complex", "Compare and contrast microservices and monolithic architecture. Analyze trade-offs for a high-traffic e-commerce platform."),
    ("Simple", "What does HTTP stand for?"),
    ("Complex", "Implement a thread-safe producer-consumer queue in Python using asyncio with proper shutdown handling."),
    ("Simple", "Name three planets."),
    ("Complex", "Derive the gradient of cross-entropy loss and explain each step of backpropagation in a neural network."),
    ("Simple", "Is Python a programming language? Yes or no."),
]


async def main():
    print("=" * 70)
    print("  Cost-Aware LLM Routing System — Demo")
    print("=" * 70)

    settings = load_settings(mode="simulation")
    registry = ModelRegistry()
    registry.load_from_yaml()

    backends = {
        settings.models.small.name: MockBackend(
            model_name=settings.models.small.name,
            tier="small",
            avg_latency_ms=30,
        ),
        settings.models.large.name: MockBackend(
            model_name=settings.models.large.name,
            tier="large",
            avg_latency_ms=100,
        ),
    }

    pipeline = InferencePipeline(
        settings=settings,
        model_registry=registry,
        backends=backends,
    )

    print(f"\nThreshold: {settings.router.threshold}")
    print(f"Small model: {settings.models.small.name} (${settings.models.small.cost_per_1k_tokens}/1K tokens)")
    print(f"Large model: {settings.models.large.name} (${settings.models.large.cost_per_1k_tokens}/1K tokens)")
    print()

    for expected, prompt in DEMO_PROMPTS:
        result = await pipeline.run(prompt)

        routed_correctly = result.model_tier == expected.lower()
        icon = "+" if routed_correctly else "-"
        print(f"[{icon}] {expected:>7} | Score: {result.routing_decision.complexity_score:.2f} | "
              f"Model: {result.model_used:<15} | "
              f"Cost: ${result.cost.cost_usd:.6f} | "
              f"Savings: {result.cost.savings_pct:.0f}% | "
              f"Latency: {result.total_latency_ms:.0f}ms")
        print(f"         Prompt: {prompt[:70]}...")
        print()

    # Summary
    summary = pipeline.cost_tracker.get_summary()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total requests:       {summary.total_requests}")
    print(f"  Total cost:           ${summary.total_cost_usd:.6f}")
    print(f"  Cost if always large: ${summary.total_cost_if_large_usd:.6f}")
    print(f"  Total savings:        ${summary.total_savings_usd:.6f} ({summary.savings_pct:.1f}%)")
    print(f"  Avg cost/request:     ${summary.avg_cost_per_request:.6f}")
    print(f"  Requests by model:    {dict(summary.requests_by_model)}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
