"""Health check API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline
from src.api.schemas import HealthResponse, ModelInfo, ModelsResponse
from src.config.settings import get_settings
from src.inference.pipeline import InferencePipeline

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> HealthResponse:
    """Check system health and model availability."""
    settings = get_settings()
    model_health = await pipeline.health_check()

    overall = "healthy" if all(model_health.values()) else "degraded"
    if not model_health:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        mode=settings.mode,
        models=model_health,
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> ModelsResponse:
    """List all available models and their status."""
    settings = get_settings()
    model_health = await pipeline.health_check()

    models = []
    for config in [settings.models.small, settings.models.large]:
        models.append(ModelInfo(
            name=config.name,
            tier=config.tier,
            cost_per_1k_tokens=config.cost_per_1k_tokens,
            avg_latency_ms=config.avg_latency_ms,
            quality_score=config.quality_score,
            backend=config.backend,
            healthy=model_health.get(config.name, False),
        ))

    return ModelsResponse(models=models)
