"""Check that the Triton TensorRT router is reachable and returns scores."""

from __future__ import annotations

import argparse

from src.router.feature_extractor import FeatureExtractor
from src.router.triton_classifier import TritonFeatureRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Triton router inference")
    parser.add_argument("--triton-url", default="localhost:8001")
    parser.add_argument("--protocol", choices=["grpc", "http"], default="grpc")
    parser.add_argument("--prompt", default="Implement a binary search tree and explain the time complexity.")
    args = parser.parse_args()

    extractor = FeatureExtractor()
    classifier = TritonFeatureRouter(
        triton_url=args.triton_url,
        protocol=args.protocol,
    )
    features = extractor.extract(args.prompt)
    score = classifier.predict_score(features)

    print(f"Prompt: {args.prompt}")
    print(f"Complexity score: {score:.4f}")
    print("Triton router check passed.")


if __name__ == "__main__":
    main()
