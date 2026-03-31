"""Pydantic Settings with YAML configuration layering."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])


class RouterConfig(BaseModel):
    threshold: float = 0.5
    use_ml_classifier: bool = False
    ml_model_path: str = "models/router/distilbert-complexity"
    onnx_model_path: str = "models/router/router.onnx"
    feature_weights: dict[str, float] = Field(
        default_factory=lambda: {"ml_score": 0.7, "rule_score": 0.3}
    )
    max_input_tokens: int = 512


class ModelConfig(BaseModel):
    name: str
    tier: Literal["small", "large"]
    cost_per_1k_tokens: float
    avg_latency_ms: float
    max_context_length: int = 4096
    quality_score: float = 0.7
    backend: Literal["mock", "huggingface", "triton"] = "mock"


class ModelsConfig(BaseModel):
    small: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="phi-2",
            tier="small",
            cost_per_1k_tokens=0.002,
            avg_latency_ms=50,
            quality_score=0.65,
        )
    )
    large: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="llama-2-70b",
            tier="large",
            cost_per_1k_tokens=0.06,
            avg_latency_ms=200,
            quality_score=0.92,
        )
    )


class CostConfig(BaseModel):
    currency: str = "USD"
    budget_alert_threshold: float = 100.0
    track_savings: bool = True


class MonitoringConfig(BaseModel):
    prometheus_enabled: bool = True
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"


class TritonConfig(BaseModel):
    url: str = "localhost:8001"
    protocol: Literal["grpc", "http"] = "grpc"
    model_repository: str = "src/triton/model_repository"
    request_timeout_ms: int = 30000


class Settings(BaseModel):
    """Application settings loaded from YAML configs with env var overrides."""

    mode: Literal["simulation", "local", "production"] = "simulation"
    api: APIConfig = Field(default_factory=APIConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    triton: TritonConfig = Field(default_factory=TritonConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(mode: str | None = None) -> Settings:
    """Load settings from YAML files with environment overlay.

    Loading order: default.yaml -> {mode}.yaml -> environment variables.
    """
    # Load base config
    default_path = CONFIG_DIR / "default.yaml"
    config = {}
    if default_path.exists():
        with open(default_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Determine mode
    mode = mode or os.getenv("APP_MODE", config.get("mode", "simulation"))
    config["mode"] = mode

    # Load mode-specific overlay
    mode_path = CONFIG_DIR / f"{mode}.yaml"
    if mode_path.exists():
        with open(mode_path, "r") as f:
            mode_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, mode_config)

    # Apply environment variable overrides
    env_overrides = {
        "api.host": os.getenv("API_HOST"),
        "api.port": os.getenv("API_PORT"),
        "router.threshold": os.getenv("ROUTING_THRESHOLD"),
        "monitoring.log_level": os.getenv("LOG_LEVEL"),
        "triton.url": os.getenv("TRITON_URL"),
    }

    for dotted_key, value in env_overrides.items():
        if value is not None:
            keys = dotted_key.split(".")
            target = config
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            # Auto-convert numeric strings
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            target[keys[-1]] = value

    return Settings(**config)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return load_settings()
