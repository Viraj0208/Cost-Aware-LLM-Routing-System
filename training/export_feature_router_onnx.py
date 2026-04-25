"""Export a deterministic feature router to ONNX.

This is the simplest path for the course project: it creates a small linear
classifier with the same 10 prompt features used by the Python router. The ONNX
file can be converted to TensorRT with trtexec and served by Triton.

Usage:
    python -m training.export_feature_router_onnx --output models/router/router.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.router.feature_vector import FEATURE_COUNT, FEATURE_INPUT_NAME, FEATURE_OUTPUT_NAME


def export_router_onnx(output_path: Path) -> None:
    """Create the ONNX router model."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise SystemExit("Install ONNX first: pip install onnx") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    complex_weights = np.array(
        [
            0.8,  # token count
            0.3,  # word count
            0.5,  # sentence count
            0.2,  # average word length
            0.1,  # vocabulary diversity
            2.0,  # code marker
            2.0,  # math marker
            1.5,  # reasoning marker
            2.0,  # keyword signal
            1.0,  # question depth
        ],
        dtype=np.float32,
    )
    weight = np.stack([-complex_weights, complex_weights], axis=1)
    bias = np.array([2.4, -2.4], dtype=np.float32)

    features = helper.make_tensor_value_info(
        FEATURE_INPUT_NAME,
        TensorProto.FLOAT,
        ["batch_size", FEATURE_COUNT],
    )
    logits = helper.make_tensor_value_info(
        FEATURE_OUTPUT_NAME,
        TensorProto.FLOAT,
        ["batch_size", 2],
    )

    nodes = [
        helper.make_node("MatMul", [FEATURE_INPUT_NAME, "WEIGHT"], ["MATMUL_OUT"]),
        helper.make_node("Add", ["MATMUL_OUT", "BIAS"], [FEATURE_OUTPUT_NAME]),
    ]
    initializers = [
        numpy_helper.from_array(weight, name="WEIGHT"),
        numpy_helper.from_array(bias, name="BIAS"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="feature_router",
        inputs=[features],
        outputs=[logits],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="cost-aware-llm-router",
        opset_imports=[helper.make_operatorsetid("", 14)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    metadata_path = output_path.with_name("router_metadata.json")
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

    print(f"Saved ONNX router: {output_path}")
    print(f"Saved metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic feature router to ONNX")
    parser.add_argument("--output", type=Path, default=Path("models/router/router.onnx"))
    args = parser.parse_args()

    export_router_onnx(args.output)


if __name__ == "__main__":
    main()
