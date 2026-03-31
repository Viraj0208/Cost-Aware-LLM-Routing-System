"""Model registry for managing available LLM profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from src.config.settings import CONFIG_DIR


@dataclass
class QualityScores:
    """Quality scores across different task categories."""
    general: float = 0.7
    code: float = 0.6
    math: float = 0.5
    reasoning: float = 0.6


@dataclass
class ModelProfile:
    """Complete profile of an LLM model including cost and capability info."""
    name: str
    display_name: str
    tier: Literal["small", "large"]
    parameters: float  # Number of parameters
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    avg_latency_ms: float
    max_context_length: int
    quality_scores: QualityScores = field(default_factory=QualityScores)

    @property
    def avg_cost_per_1k_tokens(self) -> float:
        """Average cost per 1K tokens (input + output) / 2."""
        return (self.cost_per_1k_input_tokens + self.cost_per_1k_output_tokens) / 2

    @property
    def cost_ratio(self) -> float:
        """Cost ratio relative to cheapest possible ($0.001/1K tokens)."""
        return self.avg_cost_per_1k_tokens / 0.001


@dataclass
class OptimizationProfile:
    """TensorRT optimization profile."""
    speedup_factor: float
    quality_retention: float
    requires_calibration: bool = False


class ModelRegistry:
    """Central registry of available models and their profiles."""

    def __init__(self) -> None:
        self._models: dict[str, ModelProfile] = {}
        self._optimization_profiles: dict[str, OptimizationProfile] = {}

    def load_from_yaml(self, path: Path | None = None) -> None:
        """Load model profiles from YAML configuration."""
        path = path or CONFIG_DIR / "models.yaml"
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        for model_name, model_data in data.get("models", {}).items():
            quality = model_data.pop("quality_scores", {})
            self._models[model_name] = ModelProfile(
                name=model_name,
                quality_scores=QualityScores(**quality),
                **model_data,
            )

        for opt_name, opt_data in data.get("optimization", {}).items():
            self._optimization_profiles[opt_name] = OptimizationProfile(**opt_data)

    def get_model(self, name: str) -> ModelProfile:
        """Get a model profile by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry. Available: {list(self._models.keys())}")
        return self._models[name]

    def get_models_by_tier(self, tier: Literal["small", "large"]) -> list[ModelProfile]:
        """Get all models of a specific tier."""
        return [m for m in self._models.values() if m.tier == tier]

    def get_default_small(self) -> ModelProfile:
        """Get the default small model (cheapest in small tier)."""
        small_models = self.get_models_by_tier("small")
        if not small_models:
            raise ValueError("No small models registered")
        return min(small_models, key=lambda m: m.avg_cost_per_1k_tokens)

    def get_default_large(self) -> ModelProfile:
        """Get the default large model (highest quality in large tier)."""
        large_models = self.get_models_by_tier("large")
        if not large_models:
            raise ValueError("No large models registered")
        return max(large_models, key=lambda m: m.quality_scores.general)

    def get_optimization_profile(self, name: str) -> OptimizationProfile:
        """Get an optimization profile (fp16, int8, etc.)."""
        return self._optimization_profiles[name]

    def list_models(self) -> list[ModelProfile]:
        """List all registered models."""
        return list(self._models.values())

    def register_model(self, profile: ModelProfile) -> None:
        """Register a new model profile."""
        self._models[profile.name] = profile
