"""DistilBERT-based complexity classifier for prompt routing.

Supports two inference modes:
- PyTorch: for development and fine-tuning
- ONNX Runtime: for production inference (<10ms)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger("llm_router")


class ComplexityClassifier:
    """Classifies prompt complexity using a fine-tuned DistilBERT model.

    Returns a score in [0, 1] where 0 = simple, 1 = complex.
    """

    def __init__(
        self,
        model_path: str | None = None,
        onnx_path: str | None = None,
        max_length: int = 128,
        backend: Literal["pytorch", "onnx", "auto"] = "auto",
    ) -> None:
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        self._onnx_session = None
        self._backend = backend
        self._loaded = False

        if backend == "auto":
            if onnx_path and Path(onnx_path).exists():
                self._init_onnx(onnx_path)
            elif model_path and Path(model_path).exists():
                self._init_pytorch(model_path)
            else:
                logger.warning("No classifier model found — using rule-based scoring only")
        elif backend == "onnx" and onnx_path:
            self._init_onnx(onnx_path)
        elif backend == "pytorch" and model_path:
            self._init_pytorch(model_path)

    def _init_pytorch(self, model_path: str) -> None:
        """Initialize PyTorch backend."""
        try:
            import torch
            from transformers import (
                DistilBertTokenizerFast,
                DistilBertForSequenceClassification,
            )
            self._tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self._model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self._model.eval()
            self._backend = "pytorch"
            self._loaded = True
            logger.info(f"Loaded PyTorch classifier from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load PyTorch classifier: {e}")

    def _init_onnx(self, onnx_path: str) -> None:
        """Initialize ONNX Runtime backend."""
        try:
            import onnxruntime as ort
            from transformers import DistilBertTokenizerFast

            self._onnx_session = ort.InferenceSession(onnx_path)
            # Load tokenizer from parent directory of ONNX file
            tokenizer_path = str(Path(onnx_path).parent)
            self._tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
            self._backend = "onnx"
            self._loaded = True
            logger.info(f"Loaded ONNX classifier from {onnx_path}")
        except Exception as e:
            logger.warning(f"Failed to load ONNX classifier: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict_score(self, prompt: str) -> float:
        """Predict complexity score for a prompt.

        Returns:
            Float in [0, 1] where higher = more complex.
        """
        if not self._loaded:
            return 0.5  # Neutral fallback

        if self._backend == "onnx":
            return self._predict_onnx(prompt)
        else:
            return self._predict_pytorch(prompt)

    def _predict_pytorch(self, prompt: str) -> float:
        """Run inference with PyTorch model."""
        import torch

        inputs = self._tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Return probability of class 1 (complex)
            return probs[0][1].item()

    def _predict_onnx(self, prompt: str) -> float:
        """Run inference with ONNX Runtime."""
        import numpy as np

        inputs = self._tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="np",
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        logits = self._onnx_session.run(None, onnx_inputs)[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return float(probs[0][1])
