"""Export the trained DistilBERT classifier to ONNX format.

Usage:
    python -m training.export_model --model models/router/distilbert-complexity --output models/router/router.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export_to_onnx(model_path: str, output_path: str, max_length: int = 128):
    """Export the DistilBERT classifier to ONNX."""
    try:
        import torch
        import numpy as np
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    except ImportError as e:
        print(f"ML dependencies required: {e}")
        return

    print(f"Loading model from {model_path}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Create dummy input
    dummy_text = "This is a sample prompt for tracing."
    inputs = tokenizer(
        dummy_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Export
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(output_file),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    # Validate
    try:
        import onnxruntime as ort
        print("Validating ONNX model...")
        session = ort.InferenceSession(str(output_file))
        onnx_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }
        onnx_outputs = session.run(None, onnx_inputs)

        # Compare with PyTorch outputs
        with torch.no_grad():
            pt_outputs = model(**inputs)
        pt_logits = pt_outputs.logits.numpy()
        onnx_logits = onnx_outputs[0]

        diff = np.abs(pt_logits - onnx_logits).max()
        print(f"Max difference between PyTorch and ONNX: {diff:.6f}")
        if diff < 1e-4:
            print("Validation PASSED — outputs match within tolerance.")
        else:
            print("WARNING: Outputs differ by more than 1e-4.")
    except ImportError:
        print("onnxruntime not installed — skipping validation.")

    print(f"ONNX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export router to ONNX")
    parser.add_argument("--model", type=str, default="models/router/distilbert-complexity")
    parser.add_argument("--output", type=str, default="models/router/router.onnx")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    export_to_onnx(args.model, args.output, args.max_length)


if __name__ == "__main__":
    main()
