"""Dependency injection for the FastAPI application."""

from __future__ import annotations

from functools import lru_cache

from src.config.settings import get_settings, Settings
from src.config.model_registry import ModelRegistry
from src.cost.tracker import CostTracker
from src.inference.pipeline import InferencePipeline
from src.models.mock_backend import MockBackend
from src.router.triton_classifier import TritonFeatureRouter


# Global singletons
_pipeline: InferencePipeline | None = None
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def get_pipeline() -> InferencePipeline:
    """Get or create the inference pipeline singleton."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    settings = get_settings()
    registry = ModelRegistry()
    registry.load_from_yaml()

    backends = _create_backends(settings)
    complexity_predictor = _create_complexity_predictor(settings)

    _pipeline = InferencePipeline(
        settings=settings,
        model_registry=registry,
        backends=backends,
        cost_tracker=get_cost_tracker(),
        complexity_predictor=complexity_predictor,
    )
    return _pipeline


def _create_complexity_predictor(settings: Settings):
    """Create the optional Triton complexity predictor."""
    if settings.mode != "production":
        return None

    return TritonFeatureRouter(
        triton_url=settings.triton.url,
        protocol=settings.triton.protocol,
        model_name="router",
        timeout_ms=settings.triton.request_timeout_ms,
        max_tokens=settings.router.max_input_tokens,
    )


def _create_backends(settings: Settings) -> dict:
    """Create model backends based on application mode."""
    backends = {}

    if settings.mode in {"simulation", "production"}:
        backends[settings.models.small.name] = MockBackend(
            model_name=settings.models.small.name,
            tier="small",
            avg_latency_ms=settings.models.small.avg_latency_ms,
        )
        backends[settings.models.large.name] = MockBackend(
            model_name=settings.models.large.name,
            tier="large",
            avg_latency_ms=settings.models.large.avg_latency_ms,
        )
    else:
        # Local mode with HuggingFace — fall back to mock if not available
        backends[settings.models.small.name] = MockBackend(
            model_name=settings.models.small.name,
            tier="small",
            avg_latency_ms=settings.models.small.avg_latency_ms,
        )
        backends[settings.models.large.name] = MockBackend(
            model_name=settings.models.large.name,
            tier="large",
            avg_latency_ms=settings.models.large.avg_latency_ms,
        )

    return backends


def reset_singletons() -> None:
    """Reset all singletons (for testing)."""
    global _pipeline, _cost_tracker
    _pipeline = None
    _cost_tracker = None
