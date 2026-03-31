"""Convert ONNX models to TensorRT engines.

NOTE: This script requires NVIDIA TensorRT and only runs on Linux/Docker.

Usage (direct):
    python -m src.optimization.convert_tensorrt --onnx models/router/router.onnx --output models/router/router.plan --precision fp16

Usage (Docker):
    python -m src.optimization.convert_tensorrt --onnx models/router/router.onnx --output models/router/router.plan --docker
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def convert_with_trtexec(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    workspace_mb: int = 4096,
    min_shapes: str = "input_ids:1x1,attention_mask:1x1",
    opt_shapes: str = "input_ids:8x128,attention_mask:8x128",
    max_shapes: str = "input_ids:32x512,attention_mask:32x512",
) -> None:
    """Convert ONNX to TensorRT engine using trtexec."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace_mb}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.extend(["--int8", "--fp16"])  # INT8 with FP16 fallback

    print(f"Running TensorRT conversion: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        engine_path = Path(output_path)
        print(f"TensorRT engine saved: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Conversion failed:\n{result.stderr}")


def convert_with_docker(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
) -> None:
    """Convert using NVIDIA TensorRT Docker container."""
    onnx_abs = os.path.abspath(onnx_path)
    output_abs = os.path.abspath(output_path)
    work_dir = os.path.dirname(onnx_abs)

    onnx_name = os.path.basename(onnx_path)
    output_name = os.path.basename(output_path)

    precision_flag = "--fp16" if precision == "fp16" else "--int8 --fp16"

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{work_dir}:/workspace",
        "nvcr.io/nvidia/tensorrt:24.01-py3",
        "trtexec",
        f"--onnx=/workspace/{onnx_name}",
        f"--saveEngine=/workspace/{output_name}",
        precision_flag,
        "--workspace=4096",
    ]

    print(f"Running TensorRT conversion in Docker...")
    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd)


def convert_with_python_api(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
) -> None:
    """Convert using TensorRT Python API (requires tensorrt package)."""
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT Python package not installed.")
        print("Install with: pip install tensorrt (Linux only)")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            return

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 1), (8, 128), (max_batch_size, 512))
    profile.set_shape("attention_mask", (1, 1), (8, 128), (max_batch_size, 512))
    config.add_optimization_profile(profile)

    # Build engine
    print("Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build TensorRT engine")
        return

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, required=True, help="Output TensorRT engine path")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--docker", action="store_true", help="Use Docker for conversion")
    parser.add_argument("--python-api", action="store_true", help="Use TensorRT Python API")
    args = parser.parse_args()

    if args.docker:
        convert_with_docker(args.onnx, args.output, args.precision)
    elif args.python_api:
        convert_with_python_api(args.onnx, args.output, args.precision)
    else:
        convert_with_trtexec(args.onnx, args.output, args.precision)


if __name__ == "__main__":
    main()
