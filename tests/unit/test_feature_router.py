"""Tests for the feature-based Triton router helpers."""

import numpy as np
import pytest

from src.router.feature_extractor import FeatureExtractor
from src.router.feature_vector import FEATURE_COUNT, features_to_vector
from src.router.triton_classifier import _complexity_probability


def test_features_to_vector_shape_and_dtype():
    features = FeatureExtractor().extract("Implement a binary search tree in Python.")

    vector = features_to_vector(features)

    assert vector.shape == (1, FEATURE_COUNT)
    assert vector.dtype == np.float32
    assert np.all(vector >= 0)
    assert np.all(vector <= 1)


def test_complexity_probability_from_two_logits():
    score = _complexity_probability(np.array([[0.0, 2.0]], dtype=np.float32))

    assert score == pytest.approx(0.880797, rel=1e-5)


def test_complexity_probability_from_single_logit():
    score = _complexity_probability(np.array([[0.0]], dtype=np.float32))

    assert score == pytest.approx(0.5)
