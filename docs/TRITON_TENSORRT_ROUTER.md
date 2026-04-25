# Triton + TensorRT Router Setup

This project uses Triton + TensorRT for the prompt complexity router. The final
LLM responses are still mock backends, which keeps the demo practical while
still exercising Triton/TensorRT in the routing path.

## 1. Export ONNX Router

On the development machine:

```bash
pip install onnx
python -m training.export_feature_router_onnx --output models/router/router.onnx
```

This creates:

```text
models/router/router.onnx
models/router/router_metadata.json
```

The model contract is:

```text
Input:  FEATURES, FP32, shape 1x10
Output: LOGITS,   FP32, shape 1x2
```

## 2. Convert ONNX to TensorRT

Send `models/router/router.onnx` to the NVIDIA GPU machine.

On that machine, run:

```bash
trtexec --onnx=router.onnx --saveEngine=model.plan --fp16 --minShapes=FEATURES:1x10 --optShapes=FEATURES:1x10 --maxShapes=FEATURES:1x10
```

If FP16 fails, use FP32:

```bash
trtexec --onnx=router.onnx --saveEngine=model.plan --minShapes=FEATURES:1x10 --optShapes=FEATURES:1x10 --maxShapes=FEATURES:1x10
```

Send back:

```text
model.plan
```

## 3. Place TensorRT Engine

Put the returned file here:

```text
src/triton/model_repository/router/1/model.plan
```

## 4. Run Triton-Backed Demo

From the `docker` directory:

```bash
docker compose up --build api demo prometheus
```

The API will run in `production` mode and call Triton for router inference:

```text
http://localhost:8000
```

Check the router directly:

```bash
python -m scripts.check_triton_router --triton-url localhost:8001
```

## Notes

- The TensorRT `.plan` file is generated hardware/version-specifically. For the
  cleanest demo, run Triton on the same GPU laptop that generated it.
- If `model.plan` is missing, Triton will not load the router model.
- The API still uses mock small/large model backends after routing. This is
  intentional for a course demo and avoids serving full LLMs on GPU.
