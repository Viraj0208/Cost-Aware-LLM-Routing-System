"""Tests for simulated model backends."""

import asyncio

from src.models.base import GenerationParams
from src.models.mock_backend import MockBackend


def run(coro):
    return asyncio.run(coro)


def test_mock_backend_reports_name_and_tier():
    backend = MockBackend("phi-2", "small")

    assert backend.model_name() == "phi-2"
    assert backend.model_tier() == "small"


def test_mock_backend_health_check_is_true():
    backend = MockBackend("phi-2", "small")

    assert run(backend.health_check()) is True


def test_greeting_prompt_uses_greeting_category():
    backend = MockBackend("phi-2", "small", avg_latency_ms=0, latency_jitter=0)

    result = run(backend.generate("Hello there"))

    assert result.metadata["category"] == "greeting"
    assert result.metadata["tier"] == "small"
    assert result.metadata["simulated"] is True


def test_code_prompt_uses_code_category():
    backend = MockBackend("llama-2-70b", "large", avg_latency_ms=0, latency_jitter=0)

    result = run(backend.generate("Write a Python function to sort numbers"))

    assert result.metadata["category"] == "code"
    assert result.model_name == "llama-2-70b"


def test_math_prompt_uses_math_category():
    backend = MockBackend("llama-2-70b", "large", avg_latency_ms=0, latency_jitter=0)

    result = run(backend.generate("Calculate the integral of x squared"))

    assert result.metadata["category"] == "math"


def test_reasoning_prompt_uses_reasoning_category():
    backend = MockBackend("llama-2-70b", "large", avg_latency_ms=0, latency_jitter=0)

    result = run(backend.generate("Explain why caching improves API performance"))

    assert result.metadata["category"] == "reasoning"


def test_generation_respects_max_tokens():
    backend = MockBackend("llama-2-70b", "large", avg_latency_ms=0, latency_jitter=0)

    result = run(
        backend.generate(
            "Explain why distributed systems are difficult",
            GenerationParams(max_tokens=8),
        )
    )

    assert len(result.text.split()) <= 8


def test_generation_reports_token_usage_and_latency():
    backend = MockBackend("phi-2", "small", avg_latency_ms=0, latency_jitter=0)

    result = run(backend.generate("What is Python?"))

    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens
    assert result.latency_ms >= 0
