"""Cost tracking and accumulation across requests."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from src.cost.calculator import CostEstimate


@dataclass
class CostSummary:
    """Summary of accumulated costs."""
    total_cost_usd: float
    total_cost_if_large_usd: float
    total_savings_usd: float
    savings_pct: float
    total_requests: int
    requests_by_model: dict[str, int]
    cost_by_model: dict[str, float]
    avg_cost_per_request: float
    total_tokens: int


@dataclass
class CostEvent:
    """Single cost tracking event."""
    timestamp: datetime
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_if_large_usd: float
    complexity_score: float


class CostTracker:
    """Thread-safe cost tracking across all requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[CostEvent] = []
        self._total_cost = 0.0
        self._total_cost_if_large = 0.0
        self._total_tokens = 0
        self._request_count = 0
        self._requests_by_model: dict[str, int] = defaultdict(int)
        self._cost_by_model: dict[str, float] = defaultdict(float)

    def record(self, estimate: CostEstimate, complexity_score: float = 0.0) -> None:
        """Record a cost event from an inference request."""
        event = CostEvent(
            timestamp=datetime.now(),
            model_name=estimate.model_name,
            prompt_tokens=estimate.prompt_tokens,
            completion_tokens=estimate.completion_tokens,
            cost_usd=estimate.cost_usd,
            cost_if_large_usd=estimate.cost_if_large_model_usd,
            complexity_score=complexity_score,
        )

        with self._lock:
            self._events.append(event)
            self._total_cost += estimate.cost_usd
            self._total_cost_if_large += estimate.cost_if_large_model_usd
            self._total_tokens += estimate.total_tokens
            self._request_count += 1
            self._requests_by_model[estimate.model_name] += 1
            self._cost_by_model[estimate.model_name] += estimate.cost_usd

    def get_summary(self) -> CostSummary:
        """Get a summary of all tracked costs."""
        with self._lock:
            total_savings = self._total_cost_if_large - self._total_cost
            savings_pct = 0.0
            if self._total_cost_if_large > 0:
                savings_pct = (total_savings / self._total_cost_if_large) * 100

            return CostSummary(
                total_cost_usd=self._total_cost,
                total_cost_if_large_usd=self._total_cost_if_large,
                total_savings_usd=max(0.0, total_savings),
                savings_pct=max(0.0, savings_pct),
                total_requests=self._request_count,
                requests_by_model=dict(self._requests_by_model),
                cost_by_model=dict(self._cost_by_model),
                avg_cost_per_request=(
                    self._total_cost / self._request_count
                    if self._request_count > 0
                    else 0.0
                ),
                total_tokens=self._total_tokens,
            )

    def get_recent_events(self, n: int = 50) -> list[CostEvent]:
        """Get the most recent N cost events."""
        with self._lock:
            return list(self._events[-n:])

    def reset(self) -> None:
        """Reset all tracked cost data."""
        with self._lock:
            self._events.clear()
            self._total_cost = 0.0
            self._total_cost_if_large = 0.0
            self._total_tokens = 0
            self._request_count = 0
            self._requests_by_model.clear()
            self._cost_by_model.clear()
