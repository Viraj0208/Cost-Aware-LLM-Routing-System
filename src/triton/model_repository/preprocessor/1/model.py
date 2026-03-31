"""Triton Python backend model for text preprocessing (tokenization)."""

import json
import numpy as np

try:
    import triton_python_backend_utils as pb_utils
    from transformers import DistilBertTokenizerFast

    class TritonPythonModel:
        """Tokenizes input text for downstream models."""

        def initialize(self, args):
            model_config = json.loads(args["model_config"])
            self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            self.max_length = 128

        def execute(self, requests):
            responses = []
            for request in requests:
                input_text = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
                text = input_text.as_numpy()[0][0].decode("utf-8")

                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="np",
                )

                input_ids = pb_utils.Tensor("INPUT_IDS", encoded["input_ids"].astype(np.int64))
                attention_mask = pb_utils.Tensor("ATTENTION_MASK", encoded["attention_mask"].astype(np.int64))

                response = pb_utils.InferenceResponse(output_tensors=[input_ids, attention_mask])
                responses.append(response)

            return responses

        def finalize(self):
            pass

except ImportError:
    # Not running inside Triton — this file is for reference
    pass
