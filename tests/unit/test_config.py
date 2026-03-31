"""Tests for configuration and model registry."""

from src.config.settings import load_settings
from src.config.model_registry import ModelRegistry, ModelProfile, QualityScores


def test_load_default_settings():
    settings = load_settings(mode="simulation")
    assert settings.mode == "simulation"
    assert settings.api.port == 8000
    assert settings.router.threshold == 0.5
    assert settings.models.small.tier == "small"
    assert settings.models.large.tier == "large"


def test_model_config_costs():
    settings = load_settings(mode="simulation")
    assert settings.models.small.cost_per_1k_tokens < settings.models.large.cost_per_1k_tokens


def test_model_registry_load():
    registry = ModelRegistry()
    registry.load_from_yaml()
    models = registry.list_models()
    assert len(models) >= 2


def test_model_registry_tiers():
    registry = ModelRegistry()
    registry.load_from_yaml()
    small = registry.get_models_by_tier("small")
    large = registry.get_models_by_tier("large")
    assert len(small) >= 1
    assert len(large) >= 1
    for m in small:
        assert m.tier == "small"
    for m in large:
        assert m.tier == "large"


def test_model_registry_defaults():
    registry = ModelRegistry()
    registry.load_from_yaml()
    small = registry.get_default_small()
    large = registry.get_default_large()
    assert small.tier == "small"
    assert large.tier == "large"
    assert small.avg_cost_per_1k_tokens < large.avg_cost_per_1k_tokens


def test_model_registry_register():
    registry = ModelRegistry()
    profile = ModelProfile(
        name="test-model",
        display_name="Test Model",
        tier="small",
        parameters=1e9,
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.001,
        avg_latency_ms=30,
        max_context_length=2048,
    )
    registry.register_model(profile)
    assert registry.get_model("test-model") == profile
