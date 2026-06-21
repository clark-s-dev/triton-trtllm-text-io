"""guardrail — co-located safety classifier, called by the BLS gateway as a step.

One small, ungated, easy-access classifier per direction (swap for Llama Guard 3 /
IBM Granite Guardian / NVIDIA Aegis in production — same TEXT/MODE -> BLOCKED contract):

    MODE="input"   -> prompt-injection / jailbreak  (protectai/deberta-v3-base-prompt-injection-v2)
    MODE="output"  -> toxicity / unsafe content      (unitary/toxic-bert)

Returns BLOCKED (bool), CATEGORY (the triggering label), SCORE (confidence).
A ~110-184M classifier co-locates next to a 1.5B LLM well inside the L4's 24 GB,
so input moderation can short-circuit the LLM and output moderation can gate the
stream — the B9 cascade + B12 co-location patterns in one model.
"""

from __future__ import annotations

import json

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import pipeline


def _str(request, name, default=""):
    t = pb_utils.get_input_tensor_by_name(request, name)
    if t is None:
        return default
    arr = t.as_numpy().reshape(-1)
    if arr.size == 0:
        return default
    v = arr[0]
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


class TritonPythonModel:
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        p = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}

        self.threshold = float(p.get("BLOCK_THRESHOLD", "0.5"))
        device = 0 if p.get("DEVICE", "cuda") == "cuda" else -1

        input_model = p.get("INPUT_MODEL", "protectai/deberta-v3-base-prompt-injection-v2")
        output_model = p.get("OUTPUT_MODEL", "unitary/toxic-bert")
        self.input_clf = pipeline("text-classification", model=input_model,
                                  device=device, truncation=True, max_length=512)
        self.output_clf = pipeline("text-classification", model=output_model,
                                   device=device, truncation=True, max_length=512, top_k=None)

        self.input_block = set(p.get("INPUT_BLOCK_LABELS", "INJECTION").split(","))
        self.output_block = set(p.get(
            "OUTPUT_BLOCK_LABELS",
            "toxic,severe_toxic,obscene,threat,insult,identity_hate",
        ).split(","))

    def execute(self, requests):
        responses = []
        for request in requests:
            blocked, category, score = self._classify(_str(request, "TEXT"),
                                                       _str(request, "MODE", "output"))
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("BLOCKED", np.array([blocked], dtype=bool)),
                pb_utils.Tensor("CATEGORY", np.array([category.encode("utf-8")], dtype=np.object_)),
                pb_utils.Tensor("SCORE", np.array([score], dtype=np.float32)),
            ]))
        return responses

    def _classify(self, text, mode):
        if not text.strip():
            return False, "", 0.0
        if mode == "input":
            top = self.input_clf(text)[0]
            label, score = top["label"], float(top["score"])
            blocked = label in self.input_block and score >= self.threshold
            return blocked, (label if blocked else ""), score
        # output: toxic-bert with top_k=None returns all label scores
        scores = self.output_clf(text)
        scores = scores[0] if scores and isinstance(scores[0], list) else scores
        worst = max(scores, key=lambda d: d["score"]) if scores else {"label": "", "score": 0.0}
        blocked = worst["label"] in self.output_block and float(worst["score"]) >= self.threshold
        return blocked, (worst["label"] if blocked else ""), float(worst["score"])
