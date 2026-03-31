"""Cost calculation for LLM inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostEstimate:
    """Cost breakdown for a single inference request."""
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_if_large_model_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def savings_usd(self) -> float:
        return max(0.0, self.cost_if_large_model_usd - self.cost_usd)

    @property
    def savings_pct(self) -> float:
        if self.cost_if_large_model_usd <= 0:
            return 0.0
        return (self.savings_usd / self.cost_if_large_model_usd) * 100


class CostCalculator:
    """Calculates inference costs based on token counts and model pricing."""

    def __init__(
        self,
        small_model_cost_per_1k: float = 0.002,
        large_model_cost_per_1k: float = 0.06,
        small_model_name: str = "phi-2",
        large_model_name: str = "llama-2-70b",
    ) -> None:
        self._costs = {
            small_model_name: small_model_cost_per_1k,
            large_model_name: large_model_cost_per_1k,
        }
        self._large_model_name = large_model_name

    def calculate(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> CostEstimate:
        """Calculate cost for a single inference request."""
        total_tokens = prompt_tokens + completion_tokens
        cost_per_1k = self._costs.get(model_name, 0.01)
        cost = (total_tokens / 1000) * cost_per_1k

        # What would it cost with the large model?
        large_cost_per_1k = self._costs.get(self._large_model_name, 0.06)
        cost_if_large = (total_tokens / 1000) * large_cost_per_1k

        return CostEstimate(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            cost_if_large_model_usd=cost_if_large,
        )

    def register_model_cost(self, model_name: str, cost_per_1k: float) -> None:
        """Register or update cost for a model."""
        self._costs[model_name] = cost_per_1k
