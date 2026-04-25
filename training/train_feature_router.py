"""Train and export the feature-based router for Triton/TensorRT.

This model is intentionally small: it consumes the 10 numeric features from
``src.router.feature_vector`` and outputs two logits: simple and complex.

Usage:
    python -m training.train_feature_router
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.router.feature_extractor import FeatureExtractor
from src.router.feature_vector import FEATURE_COUNT, FEATURE_INPUT_NAME, FEATURE_OUTPUT_NAME, features_to_vector
from training.prepare_dataset import generate_dataset


class FeatureRouterMLP:
    """Factory wrapper so torch is imported only when training/exporting."""

    @staticmethod
    def create():
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(FEATURE_COUNT, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )


def build_training_arrays(n_simple: int, n_complex: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic prompts and convert them to feature vectors."""
    dataset = generate_dataset(n_simple=n_simple, n_complex=n_complex, seed=seed)
    extractor = FeatureExtractor()

    x_rows = []
    y_rows = []
    for item in dataset:
        features = extractor.extract(str(item["prompt"]))
        x_rows.append(features_to_vector(features).reshape(FEATURE_COUNT))
        y_rows.append(int(item["label"]))

    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64)


def train_router(
    output_dir: Path,
    n_simple: int = 2500,
    n_complex: int = 2500,
    epochs: int = 80,
    lr: float = 0.01,
    seed: int = 42,
) -> None:
    """Train the feature router and export PyTorch + ONNX artifacts."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install ML dependencies first: pip install -r requirements-ml.txt") from exc

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_np, y_np = build_training_arrays(n_simple=n_simple, n_complex=n_complex, seed=seed)
    split_idx = int(len(x_np) * 0.8)
    x_train, x_val = x_np[:split_idx], x_np[split_idx:]
    y_train, y_val = y_np[:split_idx], y_np[split_idx:]

    model = FeatureRouterMLP.create()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val_t)
                val_pred = val_logits.argmax(dim=1)
                accuracy = (val_pred == y_val_t).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} val_accuracy={accuracy:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / "feature_router.pt"
    onnx_path = output_dir / "router.onnx"
    metadata_path = output_dir / "router_metadata.json"

    torch.save(model.state_dict(), pt_path)

    model.eval()
    dummy = torch.zeros((1, FEATURE_COUNT), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=[FEATURE_INPUT_NAME],
        output_names=[FEATURE_OUTPUT_NAME],
        dynamic_axes={
            FEATURE_INPUT_NAME: {0: "batch_size"},
            FEATURE_OUTPUT_NAME: {0: "batch_size"},
        },
        opset_version=14,
    )

    metadata = {
        "input_name": FEATURE_INPUT_NAME,
        "output_name": FEATURE_OUTPUT_NAME,
        "input_shape": [1, FEATURE_COUNT],
        "classes": ["simple", "complex"],
        "trtexec_fp16": (
            f"trtexec --onnx=router.onnx --saveEngine=model.plan --fp16 "
            f"--minShapes={FEATURE_INPUT_NAME}:1x{FEATURE_COUNT} "
            f"--optShapes={FEATURE_INPUT_NAME}:1x{FEATURE_COUNT} "
            f"--maxShapes={FEATURE_INPUT_NAME}:1x{FEATURE_COUNT}"
        ),
        "trtexec_fp32": (
            f"trtexec --onnx=router.onnx --saveEngine=model.plan "
            f"--minShapes={FEATURE_INPUT_NAME}:1x{FEATURE_COUNT} "
            f"--optShapes={FEATURE_INPUT_NAME}:1x{FEATURE_COUNT} "
            f"--maxShapes={FEATURE_INPUT_NAME}:1x{FEATURE_COUNT}"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved PyTorch weights: {pt_path}")
    print(f"Saved ONNX model: {onnx_path}")
    print(f"Saved metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the feature router")
    parser.add_argument("--output-dir", type=Path, default=Path("models/router"))
    parser.add_argument("--n-simple", type=int, default=2500)
    parser.add_argument("--n-complex", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_router(
        output_dir=args.output_dir,
        n_simple=args.n_simple,
        n_complex=args.n_complex,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
