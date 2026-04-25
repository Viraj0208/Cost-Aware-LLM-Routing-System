"""End-to-end inference pipeline orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import Settings
from src.config.model_registry import ModelRegistry
from src.cost.calculator import CostCalculator, CostEstimate
from src.cost.tracker import CostTracker
from src.inference.preprocessor import Preprocessor
from src.models.base import ModelBackend, GenerationParams, GenerationResult
from src.router.routing_engine import RoutingEngine, RoutingDecision
from src.utils.timing import timer


@dataclass
class PipelineResult:
    """Complete result from the inference pipeline."""
    request_id: str
    text: str
    model_used: str
    model_tier: str
    routing_decision: RoutingDecision
    cost: CostEstimate
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    routing_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InferencePipeline:
    """Orchestrates the full inference flow: preprocess → route → generate → track."""

    def __init__(
        self,
        settings: Settings,
        model_registry: ModelRegistry,
        backends: dict[str, ModelBackend],
        cost_tracker: CostTracker | None = None,
        complexity_predictor: Any | None = None,
    ) -> None:
        self._settings = settings
        self._registry = model_registry
        self._backends = backends  # model_name -> backend
        self._routing_engine = RoutingEngine(settings, model_registry, complexity_predictor)
        self._preprocessor = Preprocessor(max_tokens=settings.router.max_input_tokens)
        self._cost_calculator = CostCalculator(
            small_model_cost_per_1k=settings.models.small.cost_per_1k_tokens,
            large_model_cost_per_1k=settings.models.large.cost_per_1k_tokens,
            small_model_name=settings.models.small.name,
            large_model_name=settings.models.large.name,
        )
        self._cost_tracker = cost_tracker or CostTracker()

    @property
    def routing_engine(self) -> RoutingEngine:
        return self._routing_engine

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    async def run(
        self,
        prompt: str,
        params: GenerationParams | None = None,
        force_model: str | None = None,
    ) -> PipelineResult:
        """Execute the full inference pipeline.

        Args:
            prompt: The input prompt.
            params: Generation parameters.
            force_model: Optional model name to override routing.

        Returns:
            Complete PipelineResult with text, costs, and routing info.
        """
        request_id = str(uuid.uuid4())[:8]
        params = params or GenerationParams()

        with timer("total_pipeline") as total_t:
            # 1. Preprocess
            preprocessed = self._preprocessor.process(prompt)

            # 2. Route
            routing_decision = self._routing_engine.route(
                preprocessed.cleaned_prompt,
                force_model=force_model,
            )

            # 3. Get the backend for the target model
            backend = self._get_backend(routing_decision.target_model)

            # 4. Generate
            with timer("inference") as inf_t:
                gen_result = await backend.generate(preprocessed.cleaned_prompt, params)

            # 5. Calculate cost
            cost = self._cost_calculator.calculate(
                model_name=routing_decision.target_model,
                prompt_tokens=gen_result.prompt_tokens,
                completion_tokens=gen_result.completion_tokens,
            )

            # 6. Track cost
            self._cost_tracker.record(cost, routing_decision.complexity_score)

        return PipelineResult(
            request_id=request_id,
            text=gen_result.text,
            model_used=routing_decision.target_model,
            model_tier=routing_decision.target_tier,
            routing_decision=routing_decision,
            cost=cost,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            total_tokens=gen_result.total_tokens,
            routing_latency_ms=routing_decision.routing_latency_ms,
            inference_latency_ms=inf_t.elapsed_ms,
            total_latency_ms=total_t.elapsed_ms,
            metadata=gen_result.metadata,
        )

    def _get_backend(self, model_name: str) -> ModelBackend:
        """Get the backend for a given model."""
        if model_name in self._backends:
            return self._backends[model_name]

        raise RuntimeError(f"No backend available for model '{model_name}'")

    async def health_check(self) -> dict[str, bool]:
        """Check health of all backends."""
        results = {}
        predictor = getattr(self._routing_engine, "_ml_classifier", None)
        if predictor is not None and hasattr(predictor, "health_check"):
            results["triton_router"] = predictor.health_check()
        for name, backend in self._backends.items():
            try:
                results[name] = await backend.health_check()
            except Exception:
                results[name] = False
        return results
