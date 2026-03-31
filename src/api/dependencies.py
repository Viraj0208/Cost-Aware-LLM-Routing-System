"""Dependency injection for the FastAPI application."""

from __future__ import annotations

from functools import lru_cache

from src.config.settings import get_settings, Settings
from src.config.model_registry import ModelRegistry
from src.cost.tracker import CostTracker
from src.inference.pipeline import InferencePipeline
from src.models.mock_backend import MockBackend


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

    # Create backends based on mode
    backends = _create_backends(settings)

    _pipeline = InferencePipeline(
        settings=settings,
        model_registry=registry,
        backends=backends,
        cost_tracker=get_cost_tracker(),
    )
    return _pipeline


def _create_backends(settings: Settings) -> dict:
    """Create model backends based on application mode."""
    backends = {}

    if settings.mode == "simulation":
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
    elif settings.mode == "production":
        # Triton backends would be initialized here
        raise NotImplementedError("Production mode requires Triton backend setup")
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
