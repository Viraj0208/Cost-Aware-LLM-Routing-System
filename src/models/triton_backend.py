"""Triton Inference Server client backend."""

from __future__ import annotations

import time
from typing import Any

from src.models.base import ModelBackend, GenerationParams, GenerationResult
from src.utils.tokenizer import estimate_token_count


class TritonBackend(ModelBackend):
    """Model backend that communicates with Triton Inference Server.

    Uses either gRPC or HTTP protocol to send inference requests.
    Requires: pip install tritonclient[all]
    """

    def __init__(
        self,
        model_name: str,
        tier: str,
        triton_url: str = "localhost:8001",
        protocol: str = "grpc",
        timeout_ms: int = 30000,
    ) -> None:
        self._model_name = model_name
        self._tier = tier
        self._triton_url = triton_url
        self._protocol = protocol
        self._timeout_ms = timeout_ms
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Triton client."""
        try:
            if self._protocol == "grpc":
                import tritonclient.grpc as grpcclient
                self._client = grpcclient.InferenceServerClient(url=self._triton_url)
            else:
                import tritonclient.http as httpclient
                self._client = httpclient.InferenceServerClient(url=self._triton_url)
        except ImportError:
            raise ImportError(
                "Triton client not installed. Install with: pip install tritonclient[all]"
            )

    def model_name(self) -> str:
        return self._model_name

    def model_tier(self) -> str:
        return self._tier

    async def generate(self, prompt: str, params: GenerationParams | None = None) -> GenerationResult:
        """Send inference request to Triton server."""
        import numpy as np

        params = params or GenerationParams()
        start = time.perf_counter()

        # Prepare input
        if self._protocol == "grpc":
            import tritonclient.grpc as grpcclient
            input_data = grpcclient.InferInput("INPUT_TEXT", [1, 1], "BYTES")
            input_data.set_data_from_numpy(np.array([[prompt.encode("utf-8")]], dtype=np.object_))

            output = grpcclient.InferRequestedOutput("OUTPUT_TEXT")
            result = self._client.infer(
                model_name=self._model_name,
                inputs=[input_data],
                outputs=[output],
                client_timeout=self._timeout_ms / 1000,
            )
            output_text = result.as_numpy("OUTPUT_TEXT")[0][0].decode("utf-8")
        else:
            import tritonclient.http as httpclient
            input_data = httpclient.InferInput("INPUT_TEXT", [1, 1], "BYTES")
            input_data.set_data_from_numpy(np.array([[prompt.encode("utf-8")]], dtype=np.object_))

            output = httpclient.InferRequestedOutput("OUTPUT_TEXT")
            result = self._client.infer(
                model_name=self._model_name,
                inputs=[input_data],
                outputs=[output],
            )
            output_text = result.as_numpy("OUTPUT_TEXT")[0][0].decode("utf-8")

        elapsed_ms = (time.perf_counter() - start) * 1000

        return GenerationResult(
            text=output_text,
            model_name=self._model_name,
            prompt_tokens=estimate_token_count(prompt),
            completion_tokens=estimate_token_count(output_text),
            latency_ms=elapsed_ms,
            metadata={"backend": "triton", "protocol": self._protocol},
        )

    async def health_check(self) -> bool:
        """Check if the model is ready on Triton."""
        try:
            return self._client.is_model_ready(self._model_name)
        except Exception:
            return False
