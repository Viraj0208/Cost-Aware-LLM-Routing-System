"""Dynamic routing threshold management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ThresholdConfig:
    """Configuration for routing thresholds."""
    default: float = 0.5
    conservative: float = 0.3  # Routes more to large model
    aggressive: float = 0.7    # Routes more to small model


class ThresholdManager:
    """Manages the routing threshold with support for dynamic adjustment."""

    def __init__(self, initial_threshold: float = 0.5) -> None:
        self._threshold = initial_threshold
        self._config = ThresholdConfig()
        self._mode: Literal["default", "conservative", "aggressive", "custom"] = "default"

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))
        self._mode = "custom"

    def set_mode(self, mode: Literal["default", "conservative", "aggressive"]) -> None:
        """Set threshold using a predefined mode."""
        self._mode = mode
        if mode == "default":
            self._threshold = self._config.default
        elif mode == "conservative":
            self._threshold = self._config.conservative
        elif mode == "aggressive":
            self._threshold = self._config.aggressive

    @property
    def mode(self) -> str:
        return self._mode

    def should_route_to_large(self, complexity_score: float) -> bool:
        """Determine if a prompt should be routed to the large model."""
        return complexity_score >= self._threshold
