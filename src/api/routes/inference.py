"""Inference and routing API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline
from src.api.schemas import (
    CompletionRequest,
    CompletionResponse,
    CostSchema,
    RouteOnlyRequest,
    RouteOnlyResponse,
    RoutingDecisionSchema,
    UsageSchema,
)
from src.inference.pipeline import InferencePipeline
from src.models.base import GenerationParams
from src.monitoring import prometheus_metrics as pm

router = APIRouter()


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> CompletionResponse:
    """Generate a completion with intelligent cost-aware routing."""
    params = GenerationParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    result = await pipeline.run(
        prompt=request.prompt,
        params=params,
        force_model=request.force_model,
    )

    # Record Prometheus metrics
    pm.REQUESTS_TOTAL.labels(model=result.model_used, tier=result.model_tier).inc()
    pm.TOKENS_TOTAL.labels(model=result.model_used, direction="input").inc(result.prompt_tokens)
    pm.TOKENS_TOTAL.labels(model=result.model_used, direction="output").inc(result.completion_tokens)
    pm.ROUTING_LATENCY.observe(result.routing_latency_ms / 1000)
    pm.INFERENCE_LATENCY.labels(model=result.model_used).observe(result.inference_latency_ms / 1000)
    pm.TOTAL_LATENCY.observe(result.total_latency_ms / 1000)
    pm.COST_PER_REQUEST.labels(model=result.model_used).observe(result.cost.cost_usd)

    # Update rolling savings gauge
    summary = pipeline.cost_tracker.get_summary()
    pm.COST_SAVINGS_PCT.set(summary.savings_pct)

    return CompletionResponse(
        request_id=result.request_id,
        text=result.text,
        model_used=result.model_used,
        model_tier=result.model_tier,
        routing=RoutingDecisionSchema(
            target_model=result.routing_decision.target_model,
            target_tier=result.routing_decision.target_tier,
            complexity_score=result.routing_decision.complexity_score,
            routing_latency_ms=result.routing_decision.routing_latency_ms,
            reasoning=result.routing_decision.reasoning,
            potential_savings_pct=result.routing_decision.potential_savings_pct,
        ),
        usage=UsageSchema(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        ),
        cost=CostSchema(
            cost_usd=result.cost.cost_usd,
            cost_if_large_model_usd=result.cost.cost_if_large_model_usd,
            savings_usd=result.cost.savings_usd,
            savings_pct=result.cost.savings_pct,
        ),
        latency_ms=result.total_latency_ms,
    )


@router.post("/v1/route", response_model=RouteOnlyResponse)
async def route_only(
    request: RouteOnlyRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> RouteOnlyResponse:
    """Get routing decision without generating a response."""
    decision = pipeline.routing_engine.route(request.prompt)

    return RouteOnlyResponse(
        target_model=decision.target_model,
        target_tier=decision.target_tier,
        complexity_score=decision.complexity_score,
        routing_latency_ms=decision.routing_latency_ms,
        reasoning=decision.reasoning,
        estimated_cost=decision.estimated_cost,
        cost_if_large=decision.cost_if_large,
        potential_savings_pct=decision.potential_savings_pct,
    )
