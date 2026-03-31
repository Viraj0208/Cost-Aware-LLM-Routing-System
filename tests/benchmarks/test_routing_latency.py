"""Benchmark tests for routing latency — must be under 10ms."""

import time

import pytest

from src.config.settings import load_settings
from src.config.model_registry import ModelRegistry
from src.router.routing_engine import RoutingEngine


@pytest.fixture
def engine():
    settings = load_settings(mode="simulation")
    registry = ModelRegistry()
    registry.load_from_yaml()
    return RoutingEngine(settings, registry)


SAMPLE_PROMPTS = [
    "What is Python?",
    "Define machine learning.",
    "Write a binary search tree implementation in Python with error handling.",
    "Explain the theory of relativity step by step.",
    "Hello!",
    "Compare and contrast SQL and NoSQL databases for a high-traffic application.",
    "What is 2+2?",
    "Implement a distributed hash table with consistent hashing.",
    "Who is Albert Einstein?",
    "Analyze the computational complexity of quicksort in all cases.",
]


@pytest.mark.benchmark
def test_routing_latency_under_10ms(engine):
    """All routing decisions must complete in under 10ms."""
    for prompt in SAMPLE_PROMPTS:
        start = time.perf_counter()
        decision = engine.route(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10, (
            f"Routing took {elapsed_ms:.2f}ms for prompt: {prompt[:50]}..."
        )


@pytest.mark.benchmark
def test_routing_latency_p99(engine):
    """P99 routing latency must be under 10ms over 100 iterations."""
    latencies = []
    for prompt in SAMPLE_PROMPTS * 10:  # 100 total
        start = time.perf_counter()
        engine.route(prompt)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = sum(latencies) / len(latencies)

    print(f"\nRouting Latency (n={len(latencies)}):")
    print(f"  Mean: {mean:.3f}ms")
    print(f"  P50:  {latencies[50]:.3f}ms")
    print(f"  P99:  {p99:.3f}ms")
    print(f"  Max:  {max(latencies):.3f}ms")

    assert p99 < 10, f"P99 routing latency {p99:.2f}ms exceeds 10ms target"
