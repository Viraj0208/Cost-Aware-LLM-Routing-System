"""Shared test fixtures."""

import pytest

from src.config.settings import load_settings, Settings
from src.config.model_registry import ModelRegistry


@pytest.fixture
def settings() -> Settings:
    """Load test settings in simulation mode."""
    return load_settings(mode="simulation")


@pytest.fixture
def model_registry() -> ModelRegistry:
    """Create a model registry loaded from config."""
    registry = ModelRegistry()
    registry.load_from_yaml()
    return registry
