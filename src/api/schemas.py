"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """Request body for the completions endpoint."""
    prompt: str = Field(..., min_length=1, description="The input prompt")
    max_tokens: int = Field(256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    force_model: str | None = Field(None, description="Force routing to a specific model")


class RoutingDecisionSchema(BaseModel):
    """Routing decision details."""
    target_model: str
    target_tier: str
    complexity_score: float
    routing_latency_ms: float
    reasoning: list[str]
    potential_savings_pct: float


class UsageSchema(BaseModel):
    """Token usage details."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CostSchema(BaseModel):
    """Cost details for a single request."""
    cost_usd: float
    cost_if_large_model_usd: float
    savings_usd: float
    savings_pct: float


class CompletionResponse(BaseModel):
    """Response from the completions endpoint."""
    request_id: str
    text: str
    model_used: str
    model_tier: str
    routing: RoutingDecisionSchema
    usage: UsageSchema
    cost: CostSchema
    latency_ms: float


class RouteOnlyRequest(BaseModel):
    """Request body for route-only endpoint (no generation)."""
    prompt: str = Field(..., min_length=1)


class RouteOnlyResponse(BaseModel):
    """Response from route-only endpoint."""
    target_model: str
    target_tier: str
    complexity_score: float
    routing_latency_ms: float
    reasoning: list[str]
    estimated_cost: float
    cost_if_large: float
    potential_savings_pct: float


class HealthResponse(BaseModel):
    """System health status."""
    status: str
    mode: str
    models: dict[str, bool]


class CostSummaryResponse(BaseModel):
    """Cost analytics summary."""
    total_cost_usd: float
    total_cost_if_large_usd: float
    total_savings_usd: float
    savings_pct: float
    total_requests: int
    requests_by_model: dict[str, int]
    cost_by_model: dict[str, float]
    avg_cost_per_request: float
    total_tokens: int


class RoutingAnalyticsResponse(BaseModel):
    """Routing decision analytics."""
    total_requests: int
    requests_by_model: dict[str, int]
    routing_distribution: dict[str, float]  # Percentage per model
    avg_complexity_score: float
    recent_decisions: list[RecentDecision]


class RecentDecision(BaseModel):
    """A recent routing decision for analytics."""
    timestamp: str
    model_name: str
    complexity_score: float
    cost_usd: float


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str
    tier: str
    cost_per_1k_tokens: float
    avg_latency_ms: float
    quality_score: float
    backend: str
    healthy: bool


class ModelsResponse(BaseModel):
    """List of available models."""
    models: list[ModelInfo]
