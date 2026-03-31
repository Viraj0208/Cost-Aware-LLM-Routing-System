"""Benchmark inference latency and throughput across model formats.

Usage:
    python -m src.optimization.benchmark --format onnx --model models/router/router.onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def benchmark_onnx(model_path: str, num_runs: int = 100, max_length: int = 128):
    """Benchmark ONNX Runtime inference latency."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Requires: pip install onnxruntime")
        return

    session = ort.InferenceSession(model_path)

    # Create dummy inputs
    input_ids = np.random.randint(0, 30000, size=(1, max_length)).astype(np.int64)
    attention_mask = np.ones((1, max_length), dtype=np.int64)

    # Warmup
    for _ in range(10):
        session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        latencies.append((time.perf_counter() - start) * 1000)

    _print_stats("ONNX Runtime", latencies, num_runs)


def benchmark_pytorch(model_path: str, num_runs: int = 100, max_length: int = 128):
    """Benchmark PyTorch inference latency."""
    try:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    except ImportError:
        print("Requires: pip install torch transformers")
        return

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    dummy_text = "This is a sample input for benchmarking the model inference speed."
    inputs = tokenizer(dummy_text, truncation=True, padding="max_length",
                       max_length=max_length, return_tensors="pt")

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(**inputs)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)

    _print_stats("PyTorch", latencies, num_runs)


def _print_stats(name: str, latencies: list[float], num_runs: int):
    """Print benchmark statistics."""
    arr = np.array(latencies)
    print(f"\n{'='*50}")
    print(f"  {name} Benchmark Results ({num_runs} runs)")
    print(f"{'='*50}")
    print(f"  Mean:       {arr.mean():.2f} ms")
    print(f"  Median:     {np.median(arr):.2f} ms")
    print(f"  Std Dev:    {arr.std():.2f} ms")
    print(f"  P50:        {np.percentile(arr, 50):.2f} ms")
    print(f"  P90:        {np.percentile(arr, 90):.2f} ms")
    print(f"  P95:        {np.percentile(arr, 95):.2f} ms")
    print(f"  P99:        {np.percentile(arr, 99):.2f} ms")
    print(f"  Min:        {arr.min():.2f} ms")
    print(f"  Max:        {arr.max():.2f} ms")
    print(f"  Throughput: {1000 / arr.mean():.0f} inferences/sec")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference")
    parser.add_argument("--format", choices=["onnx", "pytorch"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if args.format == "onnx":
        benchmark_onnx(args.model, args.runs, args.max_length)
    else:
        benchmark_pytorch(args.model, args.runs, args.max_length)


if __name__ == "__main__":
    main()
