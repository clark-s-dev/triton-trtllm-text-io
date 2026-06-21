# Root Cause Analysis — Local L4 bring-up (engine build, serving, observability)

> 🌐 **中文版 (Chinese):** [`RCA-CN.md`](./RCA-CN.md)

This document records every issue hit while bringing `triton-trtllm-text-io` up
end-to-end on a single **NVIDIA L4** with the **NGC `tritonserver:24.10-trtllm-python-py3`**
image (which bundles **TensorRT-LLM 0.14.0**), and how each was fixed. Each entry has
**Symptom**, **Root cause**, and **Fix**.

---

## 1. Engine build: TensorRT-LLM examples/library version mismatch

**Symptom**
- `scripts/build_engines.sh` cloned the TensorRT-LLM **examples at `v0.13.0`**, but the build
  container logged `[TensorRT-LLM] TensorRT-LLM version: 0.14.0`. The `convert_checkpoint.py`
  from one release was run against the library of another.

**Root cause**
- The script's defaults were mispaired: `TRITON_TAG=24.10-trtllm-python-py3` ships TRT-LLM
  **0.14.0**, but `TRTLLM_REF` defaulted to `v0.13.0`. The example scripts and the installed
  library must be the same version.

**Fix**
- Set `TRTLLM_REF=v0.14.0` in `build_engines.sh` to match the image.

---

## 2. Engine build: Qwen2.5 tied word embeddings → `None` lm_head

**Symptom**
- `convert_checkpoint.py` crashed in `tensorrt_llm/layers/linear.py:407` with
  `AttributeError: 'NoneType' object has no attribute 'to'` while loading weights. It happened
  with **both** v0.13 and v0.14 examples.

**Root cause**
- Qwen2.5-0.5B/1.5B set `tie_word_embeddings: true` — the checkpoint has **no separate
  `lm_head.weight`** (it is tied to the input embedding). The default converter path tried to
  load `lm_head.weight`, got `None`, and called `.to()` on it.

**Fix**
- Pass `--use_embedding_sharing` to `convert_checkpoint.py`. This sets
  `share_embedding_table=True`, so the loader remaps `lm_head` to the vocab-embedding weight
  instead of expecting a separate tensor.

---

## 3. Engine build: NGC 0.14.0 library bug in `check_share_embedding()`

**Symptom**
- After enabling embedding sharing, the converter failed earlier in `qwen/model.py:346`:
  `TypeError: ModelWeightsLoader.check_share_embedding() missing 1 required positional argument: 'config'`.

**Root cause**
- The `tensorrt_llm` **bundled in the NGC 24.10 image is internally inconsistent**:
  `models/qwen/model.py` calls `loader.check_share_embedding()` with no argument, but the
  installed `models/model_weights_loader.py` defines `check_share_embedding(self, config)`
  (the helper that actually remaps the tied lm_head). The image bundles **no** examples, so the
  GitHub example scripts must be used and cannot avoid this library call.

**Fix**
- Patch the call inside the build container before converting:
  `sed -i 's/loader.check_share_embedding()/loader.check_share_embedding(config)/' <path>/qwen/model.py`.
  The library path is **hardcoded** because deriving it via `python -c "import tensorrt_llm…"`
  prints a version banner to **stdout** that polluted the captured path. The build container runs
  as **root**, so stale `ckpt/` left by a failed run must be cleaned with a throwaway root
  container (`docker run --rm -v $PWD:/work … rm -rf /work/ckpt`), not a host `rm`.

---

## 4. Model load: "trimmed" `tensorrt_llm` configs declare no I/O tensors

**Symptom**
- The `tensorrt_llm_{small,large}/config.pbtxt` declared only `parameters` — no `input`/`output`
  blocks — which the `tensorrtllm` backend cannot load.

**Root cause**
- The repo configs were intentionally "trimmed to the Part II knobs," omitting the required
  tensor set (`input_ids`, `input_lengths`, `request_output_len`, … → `output_ids`,
  `sequence_length`) that the BLS calls and the backend requires.

**Fix**
- Merge the full `input[]` set verbatim from the tensorrtllm_backend **v0.14** template, keep
  `output_ids` + `sequence_length` as `output[]`, and re-attach the project's Part II parameters.

---

## 5. Serving: only the first token of text is returned

**Symptom**
- A 128-token request streamed back only `"GPU"` then `finish_reason: length`. The model
  generated the full response but the client saw one token of text.

**Root cause**
- The 24.10 TRT-LLM backend streams the **new token(s) per decoupled response** (a delta),
  not the full running sequence. `_stream_engine` assumed cumulative output and did
  `new = seq[emitted:]; emitted = seq.shape[0]`. With per-response deltas, `emitted` became the
  delta length (1) and every subsequent `seq[1:]` was empty → all tokens after the first were dropped.

**Fix**
- Yield each response's tokens directly (each response is already the delta).

---

## 6. Serving: generation never stops at end-of-turn

**Symptom**
- Every request ran to `max_tokens` (`finish_reason: length`), even short answers like
  "The capital of France is Paris."

**Root cause**
- `_stream_engine` did not pass an `end_id`, so the engine had no end-of-turn token to stop on.

**Fix**
- Pass `end_id = tokenizer.eos_token_id` as an engine input. (At this point the BLS still
  mislabeled an EOS stop as `finish_reason: length` — generation was correct, only the label;
  that labeling is fixed in §9 below.)

---

## 7. Observability: Prometheus & OTel collector crash — config permission denied

**Symptom**
- `prometheus` exited(2) and `otel-collector` exited(1) with `open …: permission denied`
  on their mounted config files; Grafana/Jaeger/DCGM stayed up.

**Root cause**
- The repo was cloned with mode `640` (`-rw-r-----`, umask 027). These containers run as
  **non-root** users, which fall under "other" and cannot read the mounted config files.

**Fix**
- `chmod -R o+rX observability/` to make configs world-readable. (Git tracks only the
  executable bit, so `640→644` does not appear as a repo change.)

---

## 8. Observability: OTel collector crash — duplicate spanmetrics dimension

**Symptom**
- After the permission fix, `otel-collector` still exited:
  `connectors::spanmetrics: failed validating dimensions: duplicate dimension name service.name`.
  No traces reached Jaeger and the `otel-spanmetrics` Prometheus target was `down`.

**Root cause**
- In the spanmetrics connector (contrib 0.110.0), `service.name`/`span.name`/`span.kind`/
  `status.code` are **implicit** dimensions. The config re-declared `service.name`/`span.name`,
  which fails validation.

**Fix**
- Remove the explicit `dimensions` block (rely on the implicit ones; add only extra dimensions).
  After this, traces flow Triton → collector → Jaeger and all Prometheus targets are `up`.

---

## 9. Serving: finish_reason mislabeled "length" on a natural (EOS) stop

**Symptom**
- A short answer the model finished on its own — e.g. "The capital of France is Paris." in
  ~8 tokens against a 256-token budget — still returned `finish_reason: length`. The generation
  was correct; only the reported reason was wrong.

**Root cause**
- In `text_pipeline_bls`, `finish` was initialized to `"length"` and only ever flipped to
  `"stop"` when a client STOP string matched. Nothing detected an end-of-turn (EOS) stop, so every
  EOS-terminated generation fell through to the `"length"` default. (Follow-up to §6: §6 made the
  engine stop at EOS via `end_id`; the BLS still mislabeled the reason.)

**Fix**
- Count the tokens the engine emits and classify the reason explicitly — a STOP-string match or
  an early halt (`generated < max_tokens`, i.e. EOS) → `"stop"`; only exhausting the budget →
  `"length"`, matching OpenAI semantics. The decision is a pure function `classify_finish_reason()`
  extracted to `src/text_io/finish.py` (the single source of truth), imported by the BLS and covered
  by a GPU-free unit test (`tests/test_finish.py`, wired into `make test`). Verified live on the L4:
  EOS→`stop`, tiny-budget→`length`, stop-string→`stop`, CJK+emoji→`stop`.

---

## 10. Serving/DevX: edits to the `src/` "single source of truth" silently don't take effect

**Symptom**
- After adding `src/text_io/finish.py` and importing it from the BLS, `docker restart triton-llm`
  came up with `text_pipeline_bls` **UNAVAILABLE** — `ImportError: No module named 'text_io.finish'` —
  and Triton's exit-on-error then made the whole container **Exit(1)**. The other three models
  (guardrail, both engines) loaded fine, which is what made it confusing.

**Root cause**
- `src/` was **only** baked into the image (`Dockerfile: COPY src /workspace/src`), while
  `model_repository/` was bind-mounted. So editing the BLS `model.py` took effect on restart
  (it lives under the mount), but a **new file** under `src/text_io/` stayed invisible until an
  image rebuild. The project documents an "edit and `docker restart`" workflow and calls
  `src/text_io` the single source of truth — yet that workflow silently did not apply to `src/`.
  The existing `detokenize_incremental`/`stop` imported only because they were present at the last
  image build.

**Fix**
- Bind-mount the host `src/` over `/workspace/src` in `start_server.sh`
  (`-v "$PWD/src:/workspace/src"`), so the single source of truth is live-on-restart exactly like
  `model_repository`. The `Dockerfile` still `COPY`s `src/` so the image stays self-contained for a
  fresh clone (the runtime mount shadows the baked copy). Rebuilt the image and recreated the
  `triton-llm` container with the mount; all four models then **READY**.

---

## Appendix: host environment notes (not code changes)

- System Python 3.12 has **no pip** → bootstrap a venv (`python3 -m venv .venv --without-pip`
  then `get-pip.py`). `huggingface_hub` is now **1.x** — the CLI is `hf` (`hf download …`); the
  `[cli]` extra and `huggingface-cli` name are gone. The base image numpy is **1.26.4** and
  `pip install` (no `--upgrade`) keeps it, so the NumPy-2.x BYTES-tensor bug does not bite here.
  Before launching, the co-located CV server (`triton-fused`) must be stopped — it shares ports
  8000–8002 and GPU memory.
