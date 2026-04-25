"""Core routing engine that decides which model to use for each query."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.model_registry import ModelProfile, ModelRegistry
from src.config.settings import Settings
from src.router.feature_extractor import FeatureExtractor, PromptFeatures
from src.router.threshold_manager import ThresholdManager
from src.utils.timing import timer


@dataclass
class RoutingDecision:
    """Complete routing decision with explanation."""
    target_model: str
    target_tier: str
    complexity_score: float
    routing_latency_ms: float
    reasoning: list[str]
    features: PromptFeatures
    estimated_cost: float = 0.0
    cost_if_large: float = 0.0
    potential_savings_pct: float = 0.0


class RoutingEngine:
    """Intelligent routing engine that selects the optimal model for each query.

    Combines rule-based feature extraction with optional ML-based classification
    to determine prompt complexity and route to the appropriate model tier.
    """

    def __init__(
        self,
        settings: Settings,
        model_registry: ModelRegistry,
        complexity_predictor: Any | None = None,
    ) -> None:
        self._settings = settings
        self._registry = model_registry
        self._feature_extractor = FeatureExtractor()
        self._threshold_manager = ThresholdManager(settings.router.threshold)
        self._ml_classifier = complexity_predictor

    @property
    def threshold_manager(self) -> ThresholdManager:
        return self._threshold_manager

    def route(self, prompt: str, force_model: str | None = None) -> RoutingDecision:
        """Route a prompt to the optimal model.

        Args:
            prompt: The input prompt text.
            force_model: Optional model name to force routing to.

        Returns:
            RoutingDecision with target model and explanation.
        """
        with timer("routing") as t:
            features = self._feature_extractor.extract(prompt)

            # If model is forced, skip routing logic
            if force_model:
                valid_models = {
                    self._settings.models.small.name,
                    self._settings.models.large.name,
                }
                if force_model not in valid_models:
                    available = ", ".join(sorted(valid_models))
                    raise ValueError(
                        f"Unknown force_model '{force_model}'. Available models: {available}."
                    )
                return self._build_decision(
                    target_model_name=force_model,
                    complexity_score=features.rule_based_score,
                    features=features,
                    routing_latency_ms=t.elapsed_ms,
                    reasoning=["Model explicitly forced by user"],
                )

            # Get complexity score
            complexity_score = self._get_complexity_score(features)

            # Make routing decision
            use_large = self._threshold_manager.should_route_to_large(complexity_score)
            reasoning = self._explain_decision(features, complexity_score, use_large)

            if use_large:
                target = self._settings.models.large.name
            else:
                target = self._settings.models.small.name

        return self._build_decision(
            target_model_name=target,
            complexity_score=complexity_score,
            features=features,
            routing_latency_ms=t.elapsed_ms,
            reasoning=reasoning,
        )

    def _get_complexity_score(self, features: PromptFeatures) -> float:
        """Compute final complexity score from features and optional ML classifier."""
        rule_score = features.rule_based_score

        if self._ml_classifier is not None:
            # Weighted combination of ML and rule-based scores
            ml_weight = self._settings.router.feature_weights.get("ml_score", 0.7)
            rule_weight = self._settings.router.feature_weights.get("rule_score", 0.3)
            ml_score = self._ml_classifier.predict_score(features)
            return ml_score * ml_weight + rule_score * rule_weight

        return rule_score

    def _build_decision(
        self,
        target_model_name: str,
        complexity_score: float,
        features: PromptFeatures,
        routing_latency_ms: float,
        reasoning: list[str],
    ) -> RoutingDecision:
        """Build a complete routing decision with cost estimates."""
        small_config = self._settings.models.small
        large_config = self._settings.models.large

        # Determine tier
        is_large = target_model_name == large_config.name
        target_tier = "large" if is_large else "small"

        # Estimate costs
        est_tokens = features.token_count * 2  # Rough: output ~ input tokens
        if is_large:
            estimated_cost = (est_tokens / 1000) * large_config.cost_per_1k_tokens
        else:
            estimated_cost = (est_tokens / 1000) * small_config.cost_per_1k_tokens

        cost_if_large = (est_tokens / 1000) * large_config.cost_per_1k_tokens

        savings_pct = 0.0
        if cost_if_large > 0:
            savings_pct = max(0.0, (1 - estimated_cost / cost_if_large) * 100)

        return RoutingDecision(
            target_model=target_model_name,
            target_tier=target_tier,
            complexity_score=complexity_score,
            routing_latency_ms=routing_latency_ms,
            reasoning=reasoning,
            features=features,
            estimated_cost=estimated_cost,
            cost_if_large=cost_if_large,
            potential_savings_pct=savings_pct,
        )

    def _explain_decision(
        self,
        features: PromptFeatures,
        score: float,
        use_large: bool,
    ) -> list[str]:
        """Generate human-readable explanation for the routing decision."""
        reasons = []
        target = "large model" if use_large else "small model"
        reasons.append(
            f"Complexity score {score:.2f} {'≥' if use_large else '<'} "
            f"threshold {self._threshold_manager.threshold:.2f} → {target}"
        )

        if features.has_code_markers:
            reasons.append("Code patterns detected — higher complexity")
        if features.has_math_markers:
            reasons.append("Mathematical content detected — higher complexity")
        if features.has_reasoning_markers:
            reasons.append("Reasoning/analysis patterns detected — higher complexity")
        if features.token_count >= 200:
            reasons.append(f"Long prompt ({features.token_count} tokens) — higher complexity")
        elif features.token_count <= 30:
            reasons.append(f"Short prompt ({features.token_count} tokens) — lower complexity")
        if features.high_complexity_keyword_count > 0:
            reasons.append(f"{features.high_complexity_keyword_count} high-complexity keyword(s) found")
        if features.low_complexity_keyword_count > 0:
            reasons.append(f"{features.low_complexity_keyword_count} low-complexity keyword(s) found")

        return reasons
