"""Abstract base class for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result of text generation from a model backend."""
    text: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelBackend(ABC):
    """Abstract base class for LLM model backends.

    All model backends (mock, HuggingFace, Triton) implement this interface.
    """

    @abstractmethod
    async def generate(self, prompt: str, params: GenerationParams | None = None) -> GenerationResult:
        """Generate text from the model.

        Args:
            prompt: The input prompt text.
            params: Generation parameters (temperature, max_tokens, etc.).

        Returns:
            GenerationResult with generated text and metadata.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model backend is healthy and ready to serve."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model this backend serves."""
        ...

    @abstractmethod
    def model_tier(self) -> str:
        """Return the tier of the model (small or large)."""
        ...
