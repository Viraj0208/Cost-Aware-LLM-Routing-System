"""Rule-based feature extraction from prompts for complexity estimation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config.settings import CONFIG_DIR
from src.utils.tokenizer import (
    estimate_token_count,
    count_sentences,
    average_word_length,
    unique_token_ratio,
)


@dataclass
class PromptFeatures:
    """Extracted features from a prompt for routing decisions."""
    token_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    unique_token_ratio: float
    has_code_markers: bool
    has_math_markers: bool
    has_reasoning_markers: bool
    high_complexity_keyword_count: int
    low_complexity_keyword_count: int
    question_depth: int  # Nested questions / multi-part questions
    rule_based_score: float  # Aggregated complexity score [0, 1]


class FeatureExtractor:
    """Extracts complexity features from prompts using rule-based heuristics."""

    # Patterns for detecting code
    CODE_PATTERNS = [
        r'```',                    # Markdown code blocks
        r'def\s+\w+',             # Python function definitions
        r'function\s+\w+',        # JS function definitions
        r'class\s+\w+',           # Class definitions
        r'import\s+\w+',          # Import statements
        r'\w+\.\w+\(',            # Method calls
        r'if\s*\(.+\)',           # Conditional expressions
        r'for\s*\(.+\)',          # Loop expressions
        r'\bimplement\b',         # Implementation requests
        r'\bwrite\s+a\s+\w*\s*(function|class|program|code|script|module)\b',
        r'\bdebug\b',             # Debugging requests
    ]

    # Patterns for detecting math
    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Arithmetic
        r'equation|formula|calculate|compute|solve',
        r'integral|derivative|matrix|vector|probability',
        r'∑|∏|∫|√|π|θ|Σ',            # Math symbols
    ]

    # Patterns for detecting reasoning
    REASONING_PATTERNS = [
        r'explain\s+why',
        r'what\s+would\s+happen\s+if',
        r'compare\s+and\s+contrast',
        r'pros?\s+and\s+cons?',
        r'step\s+by\s+step',
        r'reason(ing)?|analy[sz]e|evaluat',
        r'in\s+what\s+way',
        r'how\s+does\s+.+\s+affect',
    ]

    def __init__(self) -> None:
        self._high_keywords: list[str] = []
        self._low_keywords: list[str] = []
        self._token_count_high = 200
        self._token_count_low = 30
        self._load_routing_rules()

    def _load_routing_rules(self) -> None:
        """Load complexity keywords from routing_rules.yaml."""
        rules_path = CONFIG_DIR / "routing_rules.yaml"
        if rules_path.exists():
            with open(rules_path, "r") as f:
                rules = yaml.safe_load(f) or {}
            indicators = rules.get("complexity_indicators", {})
            self._high_keywords = indicators.get("high_complexity_keywords", [])
            self._low_keywords = indicators.get("low_complexity_keywords", [])
            self._token_count_high = indicators.get("token_count_high", 200)
            self._token_count_low = indicators.get("token_count_low", 30)

    def extract(self, prompt: str) -> PromptFeatures:
        """Extract all features from a prompt and compute a rule-based score."""
        prompt_lower = prompt.lower()

        token_count = estimate_token_count(prompt)
        word_count = len(prompt.split())
        sentence_count = count_sentences(prompt)
        avg_wl = average_word_length(prompt)
        utr = unique_token_ratio(prompt)

        has_code = any(re.search(p, prompt, re.IGNORECASE) for p in self.CODE_PATTERNS)
        has_math = any(re.search(p, prompt, re.IGNORECASE) for p in self.MATH_PATTERNS)
        has_reasoning = any(re.search(p, prompt, re.IGNORECASE) for p in self.REASONING_PATTERNS)

        high_count = sum(1 for kw in self._high_keywords if kw.lower() in prompt_lower)
        low_count = sum(1 for kw in self._low_keywords if kw.lower() in prompt_lower)

        # Detect multi-part questions
        question_marks = prompt.count("?")
        numbered_parts = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', prompt))
        question_depth = max(question_marks, numbered_parts)

        # Compute aggregated rule-based score
        score = self._compute_score(
            token_count=token_count,
            utr=utr,
            has_code=has_code,
            has_math=has_math,
            has_reasoning=has_reasoning,
            high_count=high_count,
            low_count=low_count,
            question_depth=question_depth,
            sentence_count=sentence_count,
        )

        return PromptFeatures(
            token_count=token_count,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_wl,
            unique_token_ratio=utr,
            has_code_markers=has_code,
            has_math_markers=has_math,
            has_reasoning_markers=has_reasoning,
            high_complexity_keyword_count=high_count,
            low_complexity_keyword_count=low_count,
            question_depth=question_depth,
            rule_based_score=score,
        )

    def _compute_score(
        self,
        token_count: int,
        utr: float,
        has_code: bool,
        has_math: bool,
        has_reasoning: bool,
        high_count: int,
        low_count: int,
        question_depth: int,
        sentence_count: int,
    ) -> float:
        """Compute an aggregated complexity score in [0, 1]."""
        score = 0.5  # Start neutral

        # Length factor: longer prompts tend to be more complex
        if token_count >= self._token_count_high:
            score += 0.15
        elif token_count >= 100:
            score += 0.08  # Moderately long prompts
        elif token_count <= self._token_count_low:
            score -= 0.15

        # Vocabulary diversity
        if utr > 0.8:
            score += 0.05
        elif utr < 0.4:
            score -= 0.05

        # Domain markers (strong signals)
        if has_code:
            score += 0.15
        if has_math:
            score += 0.15
        if has_reasoning:
            score += 0.10

        # Keyword matches
        keyword_signal = (high_count - low_count) * 0.08
        score += max(-0.2, min(0.2, keyword_signal))  # Clamp

        # Multi-part questions
        if question_depth >= 3:
            score += 0.10
        elif question_depth >= 2:
            score += 0.05

        # Sentence complexity
        if sentence_count >= 5:
            score += 0.05

        return max(0.0, min(1.0, score))
