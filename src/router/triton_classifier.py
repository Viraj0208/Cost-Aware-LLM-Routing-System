"""Triton-backed prompt complexity classifier."""

from __future__ import annotations

import numpy as np

from src.router.feature_extractor import PromptFeatures
from src.router.feature_vector import (
    FEATURE_COUNT,
    FEATURE_INPUT_NAME,
    FEATURE_OUTPUT_NAME,
    features_to_vector,
)


class TritonFeatureRouter:
    """Calls a TensorRT router model served by Triton Inference Server."""

    def __init__(
        self,
        triton_url: str,
        protocol: str = "grpc",
        model_name: str = "router",
        timeout_ms: int = 30000,
        max_tokens: int = 512,
    ) -> None:
        self._triton_url = triton_url
        self._protocol = protocol
        self._model_name = model_name
        self._timeout_s = timeout_ms / 1000
        self._max_tokens = max_tokens
        self._client = self._init_client()

    def _init_client(self):
        if self._protocol == "grpc":
            import tritonclient.grpc as grpcclient

            return grpcclient.InferenceServerClient(url=self._triton_url)

        import tritonclient.http as httpclient

        return httpclient.InferenceServerClient(url=self._triton_url)

    def predict_score(self, features: PromptFeatures) -> float:
        """Return probability that the prompt is complex."""
        vector = features_to_vector(features, max_tokens=self._max_tokens)

        if self._protocol == "grpc":
            import tritonclient.grpc as client_mod

            infer_input = client_mod.InferInput(FEATURE_INPUT_NAME, [1, FEATURE_COUNT], "FP32")
            output = client_mod.InferRequestedOutput(FEATURE_OUTPUT_NAME)
            kwargs = {"client_timeout": self._timeout_s}
        else:
            import tritonclient.http as client_mod

            infer_input = client_mod.InferInput(FEATURE_INPUT_NAME, [1, FEATURE_COUNT], "FP32")
            output = client_mod.InferRequestedOutput(FEATURE_OUTPUT_NAME)
            kwargs = {}

        infer_input.set_data_from_numpy(vector)

        try:
            result = self._client.infer(
                model_name=self._model_name,
                inputs=[infer_input],
                outputs=[output],
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Triton router inference failed: {exc}") from exc

        logits = result.as_numpy(FEATURE_OUTPUT_NAME)
        if logits is None:
            raise RuntimeError(f"Triton router did not return {FEATURE_OUTPUT_NAME}")

        return _complexity_probability(logits)

    def health_check(self) -> bool:
        """Return whether the Triton router model is ready."""
        try:
            return bool(self._client.is_model_ready(self._model_name))
        except Exception:
            return False


def _complexity_probability(logits: np.ndarray) -> float:
    """Convert router logits to probability of class 1, complex."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    if logits.shape[-1] == 1:
        return float(1 / (1 + np.exp(-logits[0][0])))

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    return float(probs[0][1])
