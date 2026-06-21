"""text_pipeline_bls — the production gateway for triton-trtllm-text-io.

A single Triton **BLS** (Business Logic Scripting) model that turns a raw
TensorRT-LLM engine into a streaming `messages -> text` endpoint AND adds the
Part II production features, all by calling other Triton models as pipeline steps:

    MESSAGES ─► (1) route ─► (2) input guard ─► (3) chat-template + tokenize
             ─► (4) tensorrt_llm_{small|large}  (decoupled streaming, KV reuse)
             ─► (5) incremental detokenize + cross-token stop
             ─► (6) chunked output guard ─► stream TEXT deltas out

Why BLS (not an ensemble): steps (5)/(6) need per-request *state* carried across
the token stream (the detok buffer, the moderation window) and the ability to
*stop early*. That control flow lives naturally in imperative Python here.

This file is the orchestration only. The detok/stop math is imported from the
project's single source of truth (`src/text_io`, Part I) so the streaming path
and the laptop unit tests use identical logic.

NOTE: tensor names for the `tensorrt_llm` model (input_ids, input_lengths,
request_output_len, streaming, temperature, runtime_top_p, output_ids,
sequence_length) follow the tensorrtllm_backend template — verify against your
pinned TRT-LLM version. Validate this model on the L4 box; it cannot run on a
CPU-only laptop.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import triton_python_backend_utils as pb_utils

# --- single source of truth: Part I's pre/post math --------------------------
# Triton puts this model dir on sys.path; we also append the repo's src/ so the
# gateway reuses the SAME incremental detokenizer + stop matcher the unit tests
# cover. Set TEXT_IO_SRC in the container if your layout differs.
_SRC = os.environ.get("TEXT_IO_SRC", "/workspace/src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from text_io.detokenize_incremental import IncrementalDetokenizer
    from text_io.stop import StopSequenceMatcher
except Exception as exc:  # pragma: no cover - surfaced at load time on the server
    raise ImportError(
        f"Could not import text_io from {_SRC}. Scaffold Part I's src/text_io "
        f"(detokenize_incremental.py, stop.py) or set TEXT_IO_SRC. Original: {exc}"
    )

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# small input helpers (Triton bytes-tensors -> python scalars)
# ---------------------------------------------------------------------------
def _get_str(request, name, default=None):
    t = pb_utils.get_input_tensor_by_name(request, name)
    if t is None:
        return default
    arr = t.as_numpy().reshape(-1)
    if arr.size == 0:
        return default
    v = arr[0]
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


def _get_scalar(request, name, dtype, default):
    t = pb_utils.get_input_tensor_by_name(request, name)
    if t is None:
        return default
    arr = t.as_numpy().reshape(-1)
    return dtype(arr[0]) if arr.size else default


def _get_str_list(request, name):
    t = pb_utils.get_input_tensor_by_name(request, name)
    if t is None:
        return []
    out = []
    for v in t.as_numpy().reshape(-1):
        out.append(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v))
    return [s for s in out if s]


def _last_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return messages[-1].get("content", "") if messages else ""


class TritonPythonModel:
    # -- lifecycle -----------------------------------------------------------
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        p = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}

        tok_dir = p.get("TOKENIZER_DIR", "/models/Qwen2.5-1.5B-Instruct")
        # The tokenizer is the server-side source of the chat template AND the
        # vocabulary the incremental detokenizer decodes against.
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)

        self.small_model = p.get("SMALL_MODEL", "tensorrt_llm_small")
        self.large_model = p.get("LARGE_MODEL", "tensorrt_llm_large")
        self.out_guard_window = int(p.get("OUTPUT_GUARD_WINDOW_CHARS", "120"))
        self.enable_guard = p.get("ENABLE_GUARDRAILS", "true").lower() == "true"
        self.guard_model = p.get("GUARD_MODEL", "guardrail")

        # II.2 prefix-affinity bookkeeping: hash(shared prefix) -> last engine.
        # In a multi-instance deploy this would be a shared/consistent map; here
        # it documents the intent and keeps same-prefix traffic on one engine.
        self._affinity = {}

    def finalize(self):
        self._affinity.clear()

    # -- main entrypoint (decoupled: we stream via the response sender) -------
    def execute(self, requests):
        for request in requests:
            sender = request.get_response_sender()
            try:
                self._handle(request, sender)
            except Exception as exc:  # never leave a stream un-finalized
                resp = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"text_pipeline_bls: {exc}")
                )
                sender.send(resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        return None  # decoupled models return None

    # -- the pipeline --------------------------------------------------------
    def _handle(self, request, sender):
        messages = json.loads(_get_str(request, "MESSAGES", "[]"))
        max_tokens = _get_scalar(request, "MAX_TOKENS", int, 256)
        temperature = _get_scalar(request, "TEMPERATURE", float, 0.7)
        top_p = _get_scalar(request, "TOP_P", float, 0.95)
        stop_strings = _get_str_list(request, "STOP")
        requested_model = _get_str(request, "MODEL", "")

        # (2) INPUT GUARDRAIL — short-circuit *before* spending the LLM.
        if self.enable_guard:
            verdict = self._guard(_last_user_text(messages), mode="input")
            if verdict["blocked"]:
                self._send(sender, f"[blocked: {verdict['category']}]",
                           finish="content_filter", final=True)
                return

        # (1) ROUTE — explicit `model` field, else complexity heuristic.
        target = self._route(requested_model, messages, max_tokens)

        # (3) PREPROCESS — server-side chat template + tokenize. Shared prefix is
        #     placed first by the template so KV-cache block hashes align (II.2).
        input_ids = np.asarray(
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
            dtype=np.int32,
        )

        # (4)+(5)+(6) stream from the engine, detok incrementally, moderate, emit.
        detok = IncrementalDetokenizer(self.tokenizer)
        stops = StopSequenceMatcher(stop_strings)
        pending = ""          # buffer awaiting an output-guard decision
        finish = "length"     # default if we hit max_tokens

        for step_ids in self._stream_engine(
            target, input_ids, max_tokens, temperature, top_p
        ):
            delta = detok.add(step_ids)               # UTF-8 / SentencePiece safe
            if not delta:
                continue
            emit, stop_hit = stops.feed(delta)        # cross-token-boundary stops
            pending += emit

            if stop_hit or len(pending) >= self.out_guard_window:
                if pending and not self._flush(sender, pending):
                    return                            # blocked + redacted -> done
                pending = ""
            if stop_hit:
                finish = "stop"
                break

        # drain held-back buffers at end of stream (incomplete bytes / partial stop)
        pending += detok.flush()
        pending += stops.flush()

        # final flush + completion
        if pending and self.enable_guard and self._guard(pending, "output")["blocked"]:
            self._send(sender, "[redacted: unsafe content]",
                       finish="content_filter", final=True)
            return
        self._send(sender, pending, finish=finish, final=True)

    # -- (6) output guard on a buffered chunk; returns False if blocked -------
    def _flush(self, sender, text):
        if self.enable_guard and self._guard(text, mode="output")["blocked"]:
            self._send(sender, "[redacted: unsafe content]",
                       finish="content_filter", final=True)
            return False
        self._send(sender, text, final=False)
        return True

    # -- (1) routing ---------------------------------------------------------
    def _route(self, requested_model, messages, max_tokens):
        # Explicit override via the OpenAI-style `model` field.
        if requested_model:
            if requested_model.endswith("small") or "0.5" in requested_model:
                return self.small_model
            if requested_model.endswith("large") or "1.5" in requested_model:
                return self.large_model
        # Complexity heuristic (the cheap router). A tiny classifier model is the
        # upgrade — same call site, just swap this body for a guard()-style exec.
        approx_chars = sum(len(m.get("content", "")) for m in messages)
        if approx_chars > 800 or max_tokens > 256:
            return self.large_model
        return self.small_model

    # -- (2)/(6) call the co-located guard model -----------------------------
    def _guard(self, text, mode):
        if not text or not text.strip():
            return {"blocked": False, "category": "", "score": 0.0}
        req = pb_utils.InferenceRequest(
            model_name=self.guard_model,
            requested_output_names=["BLOCKED", "CATEGORY", "SCORE"],
            inputs=[
                pb_utils.Tensor("TEXT", np.array([text.encode("utf-8")], dtype=np.object_)),
                pb_utils.Tensor("MODE", np.array([mode.encode("utf-8")], dtype=np.object_)),
            ],
        )
        resp = req.exec()
        if resp.has_error():
            # Policy: fail-CLOSED on input (be safe), fail-OPEN on output (don't
            # nuke a good generation because the guard hiccuped).
            return {"blocked": mode == "input", "category": "guard_error", "score": 1.0}
        blocked = bool(pb_utils.get_output_tensor_by_name(resp, "BLOCKED").as_numpy().reshape(-1)[0])
        cat = pb_utils.get_output_tensor_by_name(resp, "CATEGORY").as_numpy().reshape(-1)[0]
        cat = cat.decode("utf-8") if isinstance(cat, (bytes, bytearray)) else str(cat)
        return {"blocked": blocked, "category": cat, "score": 0.0}

    # -- (4) decoupled streaming call into the chosen TRT-LLM engine ----------
    def _stream_engine(self, model_name, input_ids, max_tokens, temperature, top_p):
        inputs = [
            pb_utils.Tensor("input_ids", input_ids.reshape(1, -1)),
            pb_utils.Tensor("input_lengths", np.array([[input_ids.size]], dtype=np.int32)),
            pb_utils.Tensor("request_output_len", np.array([[max_tokens]], dtype=np.int32)),
            pb_utils.Tensor("temperature", np.array([[temperature]], dtype=np.float32)),
            pb_utils.Tensor("runtime_top_p", np.array([[top_p]], dtype=np.float32)),
            pb_utils.Tensor("streaming", np.array([[True]], dtype=bool)),
        ]
        req = pb_utils.InferenceRequest(
            model_name=model_name,
            requested_output_names=["output_ids", "sequence_length"],
            inputs=inputs,
        )
        # exec(decoupled=True) yields one response per decode step.
        emitted = 0
        for resp in req.exec(decoupled=True):
            if resp.has_error():
                raise RuntimeError(resp.error().message())
            seq = pb_utils.get_output_tensor_by_name(resp, "output_ids").as_numpy().reshape(-1)
            # Some backend versions stream the full running sequence, others just
            # the new token(s). Track how many we've seen and yield only the new.
            new = seq[emitted:]
            emitted = seq.shape[0]
            if new.size:
                yield new.astype(np.int64).tolist()

    # -- streamed response out ----------------------------------------------
    def _send(self, sender, text, finish=None, final=False):
        tensors = [pb_utils.Tensor("TEXT", np.array([text.encode("utf-8")], dtype=np.object_))]
        if finish is not None:
            tensors.append(
                pb_utils.Tensor("FINISH_REASON", np.array([finish.encode("utf-8")], dtype=np.object_))
            )
        resp = pb_utils.InferenceResponse(output_tensors=tensors)
        flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if final else 0
        sender.send(resp, flags=flags)
