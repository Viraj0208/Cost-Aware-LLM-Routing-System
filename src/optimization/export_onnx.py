"""Export models to ONNX format for TensorRT conversion.

Usage:
    python -m src.optimization.export_onnx --model distilbert-base-uncased --output models/router/router.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export_classifier_to_onnx(
    model_path: str,
    output_path: str,
    max_length: int = 128,
    opset_version: int = 14,
) -> None:
    """Export a HuggingFace sequence classifier to ONNX format."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        print(f"Requires: pip install torch transformers  ({e})")
        return

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    dummy = tokenizer(
        "Sample input text",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(output_file),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
    )
    print(f"ONNX export complete: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF model name")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    export_classifier_to_onnx(args.model, args.output, args.max_length)


if __name__ == "__main__":
    main()
