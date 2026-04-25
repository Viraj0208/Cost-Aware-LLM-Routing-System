"""Feature vector conversion for the TensorRT/Triton router model."""

from __future__ import annotations

import numpy as np

from src.router.feature_extractor import PromptFeatures


FEATURE_INPUT_NAME = "FEATURES"
FEATURE_OUTPUT_NAME = "LOGITS"
FEATURE_COUNT = 10


def features_to_vector(features: PromptFeatures, max_tokens: int = 512) -> np.ndarray:
    """Convert extracted prompt features into the fixed router model input."""
    token_norm = min(features.token_count / max_tokens, 1.0)
    word_norm = min(features.word_count / 400, 1.0)
    sentence_norm = min(features.sentence_count / 20, 1.0)
    avg_word_norm = min(features.avg_word_length / 12, 1.0)
    keyword_signal = max(
        0.0,
        min(1.0, (features.high_complexity_keyword_count - features.low_complexity_keyword_count + 5) / 10),
    )
    question_depth_norm = min(features.question_depth / 5, 1.0)

    return np.array(
        [[
            token_norm,
            word_norm,
            sentence_norm,
            avg_word_norm,
            features.unique_token_ratio,
            float(features.has_code_markers),
            float(features.has_math_markers),
            float(features.has_reasoning_markers),
            keyword_signal,
            question_depth_norm,
        ]],
        dtype=np.float32,
    )
