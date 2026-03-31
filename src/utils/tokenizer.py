"""Token counting utilities for cost estimation."""

from __future__ import annotations

import re


def estimate_token_count(text: str) -> int:
    """Estimate token count using a simple word-piece heuristic.

    This provides a reasonable approximation without requiring a real tokenizer.
    Rule of thumb: ~1.3 tokens per word for English text.
    """
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences in text using punctuation heuristics."""
    if not text.strip():
        return 0
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])


def average_word_length(text: str) -> float:
    """Calculate average word length."""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def unique_token_ratio(text: str) -> float:
    """Calculate ratio of unique tokens to total tokens (vocabulary diversity)."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)
