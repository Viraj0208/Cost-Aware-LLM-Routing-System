"""Tests for prompt preprocessing."""

import pytest

from src.inference.preprocessor import Preprocessor


def test_preprocessor_normalizes_whitespace():
    result = Preprocessor(max_tokens=20).process("  Hello\n\n   world  ")

    assert result.cleaned_prompt == "Hello world"
    assert result.original_prompt == "  Hello\n\n   world  "
    assert result.token_count > 0


def test_preprocessor_rejects_oversized_prompt():
    prompt = "word " * 100

    with pytest.raises(ValueError, match="Prompt is too long"):
        Preprocessor(max_tokens=20).process(prompt)
