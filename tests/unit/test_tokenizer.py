"""Tests for lightweight tokenizer utility functions."""

import pytest

from src.utils.tokenizer import (
    average_word_length,
    count_sentences,
    count_words,
    estimate_token_count,
    unique_token_ratio,
)


def test_estimate_token_count_empty_text_is_zero():
    assert estimate_token_count("") == 0


def test_estimate_token_count_never_returns_zero_for_non_empty_text():
    assert estimate_token_count("hello") == 1


def test_estimate_token_count_uses_word_piece_multiplier():
    assert estimate_token_count("one two three four five") == 6


def test_count_words_splits_on_any_whitespace():
    assert count_words("one\ttwo\nthree   four") == 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world.", 1),
        ("Hello! How are you? Fine.", 3),
        ("No punctuation but still one sentence", 1),
        ("   ", 0),
    ],
)
def test_count_sentences_handles_common_punctuation(text, expected):
    assert count_sentences(text) == expected


def test_average_word_length_returns_zero_for_empty_text():
    assert average_word_length("") == 0.0


def test_average_word_length_uses_all_words():
    assert average_word_length("AI routing system") == pytest.approx((2 + 7 + 6) / 3)


def test_unique_token_ratio_is_case_insensitive():
    assert unique_token_ratio("AI ai Router") == pytest.approx(2 / 3)


def test_unique_token_ratio_returns_zero_for_empty_text():
    assert unique_token_ratio("") == 0.0
