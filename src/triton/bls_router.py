"""Business Logic Scripting (BLS) model for conditional routing within Triton.

This script runs as a Triton Python backend model. It receives the router's
classification output and conditionally dispatches to either model_small or
model_large using Triton's BLS API.

Triton's ensemble DAG doesn't support conditional branching natively, so BLS
handles the routing logic server-side, eliminating network round-trips.
"""

import json
import numpy as np

try:
    import triton_python_backend_utils as pb_utils

    class TritonPythonModel:
        """BLS model that routes requests to small or large model based on router output."""

        def initialize(self, args):
            self.model_config = json.loads(args["model_config"])
            self.threshold = 0.5  # Complexity threshold

        def execute(self, requests):
            responses = []

            for request in requests:
                # Get router logits
                router_logits = pb_utils.get_input_tensor_by_name(
                    request, "ROUTER_LOGITS"
                ).as_numpy()

                # Get preprocessed inputs to forward to the selected model
                input_ids = pb_utils.get_input_tensor_by_name(request, "INPUT_IDS")
                attention_mask = pb_utils.get_input_tensor_by_name(request, "ATTENTION_MASK")

                # Compute complexity probability (softmax of logits)
                exp_logits = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
                complexity_score = float(probs[0][1])  # P(complex)

                # Route based on threshold
                if complexity_score >= self.threshold:
                    target_model = "model_large"
                    target_tier = "large"
                else:
                    target_model = "model_small"
                    target_tier = "small"

                # Create inference request to the target model using BLS
                infer_request = pb_utils.InferenceRequest(
                    model_name=target_model,
                    requested_output_names=["logits"],
                    inputs=[input_ids, attention_mask],
                )

                # Execute inference on the selected model
                infer_response = infer_request.exec()

                if infer_response.has_error():
                    error = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"Failed to run {target_model}: {infer_response.error().message()}"
                        )
                    )
                    responses.append(error)
                    continue

                # Get model output
                output_logits = pb_utils.get_output_tensor_by_name(infer_response, "logits")

                # Create routing info JSON
                routing_info = json.dumps({
                    "target_model": target_model,
                    "target_tier": target_tier,
                    "complexity_score": complexity_score,
                    "threshold": self.threshold,
                }).encode("utf-8")

                routing_tensor = pb_utils.Tensor(
                    "ROUTING_INFO",
                    np.array([[routing_info]], dtype=np.object_),
                )

                response = pb_utils.InferenceResponse(
                    output_tensors=[output_logits, routing_tensor]
                )
                responses.append(response)

            return responses

        def finalize(self):
            pass

except ImportError:
    # Not running inside Triton
    pass
