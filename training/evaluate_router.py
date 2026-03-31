"""Evaluate the trained router classifier.

Usage:
    python -m training.evaluate_router --model models/router/distilbert-complexity --data training/data/router_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def evaluate(model_path: str, data_path: str, max_length: int = 128):
    """Evaluate the trained classifier on a dataset."""
    try:
        import torch
        import numpy as np
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as e:
        print(f"ML dependencies required: {e}")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Load data
    prompts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["prompt"])
            labels.append(int(row["label"]))

    # Run predictions with timing
    predictions = []
    latencies = []

    print(f"Running inference on {len(prompts)} samples...")
    with torch.no_grad():
        for prompt in prompts:
            start = time.perf_counter()
            inputs = tokenizer(prompt, truncation=True, padding=True,
                               max_length=max_length, return_tensors="pt")
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            elapsed_ms = (time.perf_counter() - start) * 1000

            predictions.append(pred)
            latencies.append(elapsed_ms)

    # Report
    print("\n=== Classification Report ===")
    print(classification_report(labels, predictions, target_names=["simple", "complex"]))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(labels, predictions)
    print(f"  Predicted:  simple  complex")
    print(f"  simple:     {cm[0][0]:>6}  {cm[0][1]:>7}")
    print(f"  complex:    {cm[1][0]:>6}  {cm[1][1]:>7}")

    print(f"\n=== Latency Statistics ===")
    print(f"  Mean:   {np.mean(latencies):.2f}ms")
    print(f"  Median: {np.median(latencies):.2f}ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f}ms")
    print(f"  Min:    {np.min(latencies):.2f}ms")
    print(f"  Max:    {np.max(latencies):.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Evaluate router classifier")
    parser.add_argument("--model", type=str, default="models/router/distilbert-complexity")
    parser.add_argument("--data", type=str, default="training/data/router_dataset.csv")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    evaluate(args.model, args.data, args.max_length)


if __name__ == "__main__":
    main()
