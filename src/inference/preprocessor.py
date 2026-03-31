"""Input preprocessing for the inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.tokenizer import estimate_token_count


@dataclass
class PreprocessedInput:
    """Preprocessed input ready for inference."""
    original_prompt: str
    cleaned_prompt: str
    token_count: int


class Preprocessor:
    """Preprocesses input prompts before routing and inference."""

    def __init__(self, max_tokens: int = 512) -> None:
        self._max_tokens = max_tokens

    def process(self, prompt: str) -> PreprocessedInput:
        """Clean and prepare a prompt for inference."""
        cleaned = prompt.strip()

        # Normalize whitespace
        cleaned = " ".join(cleaned.split())

        token_count = estimate_token_count(cleaned)

        return PreprocessedInput(
            original_prompt=prompt,
            cleaned_prompt=cleaned,
            token_count=token_count,
        )
