Place the TensorRT engine for the feature router in this directory as:

```text
model.plan
```

Build it from `models/router/router.onnx` using `trtexec`. The expected model
input is `FEATURES` with shape `1x10`, and the output is `LOGITS` with shape
`1x2`.
