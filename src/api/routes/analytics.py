"""Cost and routing analytics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline, get_cost_tracker
from src.api.schemas import (
    CostSummaryResponse,
    RecentDecision,
    RoutingAnalyticsResponse,
)
from src.cost.tracker import CostTracker
from src.inference.pipeline import InferencePipeline

router = APIRouter()


@router.get("/v1/analytics/costs", response_model=CostSummaryResponse)
async def cost_analytics(
    tracker: CostTracker = Depends(get_cost_tracker),
) -> CostSummaryResponse:
    """Get cost analytics summary."""
    summary = tracker.get_summary()
    return CostSummaryResponse(
        total_cost_usd=summary.total_cost_usd,
        total_cost_if_large_usd=summary.total_cost_if_large_usd,
        total_savings_usd=summary.total_savings_usd,
        savings_pct=summary.savings_pct,
        total_requests=summary.total_requests,
        requests_by_model=summary.requests_by_model,
        cost_by_model=summary.cost_by_model,
        avg_cost_per_request=summary.avg_cost_per_request,
        total_tokens=summary.total_tokens,
    )


@router.get("/v1/analytics/routing", response_model=RoutingAnalyticsResponse)
async def routing_analytics(
    tracker: CostTracker = Depends(get_cost_tracker),
) -> RoutingAnalyticsResponse:
    """Get routing decision analytics."""
    summary = tracker.get_summary()
    events = tracker.get_recent_events(20)

    # Calculate routing distribution percentages
    distribution = {}
    if summary.total_requests > 0:
        for model, count in summary.requests_by_model.items():
            distribution[model] = (count / summary.total_requests) * 100

    # Calculate average complexity score
    avg_score = 0.0
    if events:
        avg_score = sum(e.complexity_score for e in events) / len(events)

    recent = [
        RecentDecision(
            timestamp=e.timestamp.isoformat(),
            model_name=e.model_name,
            complexity_score=e.complexity_score,
            cost_usd=e.cost_usd,
        )
        for e in reversed(events)  # Most recent first
    ]

    return RoutingAnalyticsResponse(
        total_requests=summary.total_requests,
        requests_by_model=summary.requests_by_model,
        routing_distribution=distribution,
        avg_complexity_score=avg_score,
        recent_decisions=recent,
    )


@router.post("/v1/analytics/reset")
async def reset_analytics(
    tracker: CostTracker = Depends(get_cost_tracker),
) -> dict:
    """Reset all analytics data."""
    tracker.reset()
    return {"status": "ok", "message": "Analytics data reset successfully"}
