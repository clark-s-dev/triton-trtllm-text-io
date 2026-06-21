# Implementation Tutorial — how each feature is built

A code-level walkthrough of `triton-trtllm-text-io`, for someone who wants to understand
*how* the gateway works, not just run it. We follow a request from "client sends `messages`"
to "client receives streamed text," and at each stop explain **what** the feature does,
**where** it lives, the **key code**, and **why** it's built that way.

> Companion docs: [`REPORT.md`](./REPORT.md) (architecture + status), [`RCA.md`](./RCA-EN.md)
> (every bring-up bug, bilingual), [`GUARDRAILS.md`](./GUARDRAILS.md) (guardrail test report).

## The request, end to end

```
 MESSAGES ─► (1) route ─► (2) input guard ─► (2b) scope guard ─► (3) chat-template + tokenize
          ─► (4) tensorrt_llm_{small|large}  (decoupled streaming · KV-cache reuse)
          ─► (5) incremental detokenize + cross-token stop
          ─► (6) chunked output guard ─► (7) finish_reason ─► streamed TEXT deltas
```

Steps (1)–(7) all happen **inside one Triton model**, `text_pipeline_bls`. The LLM itself
(`tensorrt_llm_small` / `_large`) is a separate model this one *calls*.

### Map: feature → code

| # | Feature | Primary code |
|---|---|---|
| 0 | The BLS gateway (why, not an ensemble) | `model_repository/text_pipeline_bls/1/model.py` |
| 1 | Routing (small vs large) | `model.py → _route()` |
| 2 | Input / output / **scope** guardrails | `model_repository/guardrail/1/model.py`, `model.py → _guard()/_flush()` |
| 3 | Server-side chat-template + tokenize | `model.py → _handle()` |
| 4 | Decoupled streaming from TRT-LLM | `model.py → _stream_engine()` |
| 5 | Incremental detokenization (byte-exact) | `src/text_io/detokenize_incremental.py` |
| 6 | Cross-token stop sequences | `src/text_io/stop.py` |
| 7 | `finish_reason` classification | `src/text_io/finish.py` |
| 8 | KV-cache reuse | `model_repository/tensorrt_llm_*/config.pbtxt`, `scripts/build_engines.sh` |
| 9 | Engine build | `scripts/build_engines.sh` |
| 10 | Readiness gate | `scripts/check_env.py` |
| 11 | Observability | `observability/*` |
| 12 | The streaming client | `client/client_fused.py` |
| 13 | Test strategy (single source of truth) | `src/text_io/*` ↔ `tests/*` |

---

## 0. The gateway: a BLS model, not an ensemble

**What.** One Triton model orchestrates the whole pipeline by calling other models as steps.

**Where.** `model_repository/text_pipeline_bls/1/model.py` + its `config.pbtxt`.

**How.** The config declares the model **decoupled** (one request → many streamed responses)
and CPU-only (it's pure orchestration; the GPU work is in the models it calls):

```pbtxt
# text_pipeline_bls/config.pbtxt
backend: "python"
max_batch_size: 0
model_transaction_policy { decoupled: true }   # required for token streaming
instance_group [ { count: 2 kind: KIND_CPU } ]  # each Python instance is its own process
```

Because it's decoupled, `execute()` doesn't *return* responses — it grabs a **response sender**
per request and pushes responses as tokens arrive, marking the last one `FINAL`:

```python
# model.py → execute()
def execute(self, requests):
    for request in requests:
        sender = request.get_response_sender()
        try:
            self._handle(request, sender)
        except Exception as exc:                       # never leave a stream un-finalized
            resp = pb_utils.InferenceResponse(error=pb_utils.TritonError(f"text_pipeline_bls: {exc}"))
            sender.send(resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
    return None  # decoupled models return None
```

**Why a BLS and not a Triton ensemble.** An ensemble is a static DAG — it can't carry
**per-request state** across the token stream (the detok buffer, the moderation window) or
**stop early**. Steps 5/6 need exactly that imperative control flow, so it lives in Python.
See `_handle()` for the whole pipeline in ~50 lines.

---

## 1. Routing — pick the cheap model when you can

**What.** Send a request to the 0.5B engine or the 1.5B engine.

**Where.** `model.py → _route()`.

**How.** An explicit OpenAI-style `model` field wins; otherwise a cheap complexity heuristic:

```python
def _route(self, requested_model, messages, max_tokens):
    if requested_model:                                   # explicit override
        if requested_model.endswith("small") or "0.5" in requested_model:
            return self.small_model
        if requested_model.endswith("large") or "1.5" in requested_model:
            return self.large_model
    approx_chars = sum(len(m.get("content", "")) for m in messages)
    if approx_chars > 800 or max_tokens > 256:            # "looks expensive" → big model
        return self.large_model
    return self.small_model
```

**Why.** The heuristic is deliberately trivial and documented as the upgrade point: swap the
body for a tiny classifier model (called like the guard) and the call site doesn't change.

---

## 2. Guardrails — a co-located classifier, called as a step

**What.** Block prompt-injection on the way *in* (before spending the LLM) and toxic content on
the way *out* (gating the stream in chunks).

**Where.** The classifier is its own model: `model_repository/guardrail/1/model.py`. The gateway
calls it at `model.py → _guard()` and `_flush()`.

**How — the classifier.** One small HF `pipeline` per direction, behind a uniform
`TEXT/MODE → BLOCKED/CATEGORY/SCORE` contract:

```python
# guardrail/1/model.py → initialize()
self.input_clf  = pipeline("text-classification", model=input_model,  device=device, ...)  # deberta-v3 injection
self.output_clf = pipeline("text-classification", model=output_model, device=device, top_k=None)  # toxic-bert
self.threshold  = float(p.get("BLOCK_THRESHOLD", "0.5"))
```

**How — the cascade.** Input moderation runs first and *short-circuits the LLM entirely*:

```python
# model.py → _handle()
if self.enable_guard:
    verdict = self._guard(_last_user_text(messages), mode="input")
    if verdict["blocked"]:
        self._send(sender, f"[blocked: {verdict['category']}]", finish="content_filter", final=True)
        return                                            # the engine never runs
```

Output moderation runs on each buffered chunk (`_flush`) and on the final buffer; a hit replaces
the response. The gateway calls the guard model in-process via a Triton `InferenceRequest`:

```python
# model.py → _guard()
req = pb_utils.InferenceRequest(model_name=self.guard_model,
        requested_output_names=["BLOCKED", "CATEGORY", "SCORE"],
        inputs=[pb_utils.Tensor("TEXT", ...), pb_utils.Tensor("MODE", ...)])
resp = req.exec()
if resp.has_error():
    # Policy: fail-CLOSED on input (be safe), fail-OPEN on output (don't nuke a good answer).
    return {"blocked": mode == "input", "category": "guard_error", "score": 1.0}
```

**Scope gate (`MODE="topic"`).** A third classifier restricts the gateway to one business domain
(here, **NVIDIA GTC**) — zero-shot NLI, so it needs no training, just candidate labels:

```python
# guardrail/1/model.py → _classify(), mode == "topic"
res = self.topic_clf(text, self.topic_labels,             # ["NVIDIA GTC …", "an unrelated topic"]
                     hypothesis_template="This text is about {}.", multi_label=False)
on_topic = dict(zip(res["labels"], res["scores"]))[self.topic_label]
blocked = on_topic < self.topic_threshold                  # below the relevance bar → off-topic
```

The gateway runs it right after the injection check and returns a **static denial** before the LLM:

```python
# model.py → _handle(), step (2b)
if self.restrict_topic and self._guard(_last_user_text(messages), mode="topic")["blocked"]:
    self._send(sender, self.topic_deny_message, finish="content_filter", final=True)
    return
```

It runs on **CPU** (`TOPIC_DEVICE=cpu`) so it takes no VRAM from the engines, and everything —
labels, threshold, denial message, on/off — is a `config.pbtxt` knob. Measured separation
(on-topic ≥ 0.67, off-topic ≤ 0.43; threshold 0.5) and the static denials are in
[`GUARDRAILS.md`](./GUARDRAILS.md) §3–§4.

**Why.** A ~110–184M classifier co-located next to a 1.5B LLM fits the 24 GB L4 easily, so input
moderation can run *before* the expensive generation and output moderation can gate a live stream.
The `TEXT/MODE` contract means you can swap in Llama Guard 3 / Granite Guardian with **zero**
gateway changes. Real measured behavior is in [`GUARDRAILS.md`](./GUARDRAILS.md).

---

## 3. Server-side chat-template + tokenize

**What.** The client sends raw `messages`; the *server* turns them into token ids.

**Where.** `model.py → _handle()`.

```python
input_ids = np.asarray(
    self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True),
    dtype=np.int32,
)
```

**Why.** Two reasons. (a) Clients shouldn't need the tokenizer or the model's chat template —
that's the gateway's whole point. (b) Applying the template server-side puts the shared system/
prefix tokens *first*, so the KV-cache block hashes line up across requests and prefix reuse
(feature 8) actually fires.

---

## 4. Decoupled streaming from the TRT-LLM engine

**What.** Stream token deltas out of the `tensorrtllm` backend.

**Where.** `model.py → _stream_engine()`.

**How.** Build the engine inputs (tensor names follow the tensorrtllm_backend template), then
iterate `req.exec(decoupled=True)` — each response is one decode step:

```python
inputs = [
    pb_utils.Tensor("input_ids", input_ids.reshape(1, -1)),
    pb_utils.Tensor("input_lengths", np.array([[input_ids.size]], dtype=np.int32)),
    pb_utils.Tensor("request_output_len", np.array([[max_tokens]], dtype=np.int32)),
    pb_utils.Tensor("temperature", ...), pb_utils.Tensor("runtime_top_p", ...),
    pb_utils.Tensor("streaming", np.array([[True]], dtype=bool)),
]
if self.end_id is not None:                                # stop at end-of-turn, not max_tokens
    inputs.append(pb_utils.Tensor("end_id", np.array([[self.end_id]], dtype=np.int32)))
...
for resp in req.exec(decoupled=True):
    out = pb_utils.get_output_tensor_by_name(resp, "output_ids")
    seq = out.as_numpy().reshape(-1)
    if seq.size:
        yield seq.astype(np.int64).tolist()
```

**The non-obvious part (RCA §5).** The 24.10 backend streams the **new token(s) per response** (a
delta), *not* the cumulative running sequence. An earlier version assumed cumulative output and
sliced `seq[emitted:]`, which dropped every token after the first. Here we yield each response's
tokens directly. The `end_id` input (RCA §6) is what makes generation stop at end-of-turn.

---

## 5. Incremental detokenization — the centerpiece

**What.** Turn streaming token ids into **byte-exact** text, so a Chinese glyph or emoji split
across two tokens never renders as `�`.

**Where.** `src/text_io/detokenize_incremental.py` (lines 23–76).

**The problem.** Tokens ≠ characters. With byte-level BPE (Qwen/Llama 3), one CJK glyph (3 UTF-8
bytes) or emoji (4 bytes) is often split across tokens. `tokenizer.decode([one_token])` then
yields incomplete bytes → `�`.

**The algorithm** — a running token buffer plus a `(prefix_offset, read_offset)` window. On each
new token, decode the window and emit only the *newly completed* text; if it ends in `�`, the
trailing bytes are still in flight, so emit nothing and wait:

```python
# detokenize_incremental.py → _step()
self.tokens.extend(pieces)
prefix_text = self.tok.convert_tokens_to_string(self.tokens[self.prefix_offset:self.read_offset])
new_text    = self.tok.convert_tokens_to_string(self.tokens[self.prefix_offset:])

if len(new_text) > len(prefix_text) and not new_text.endswith(_REPLACEMENT_CHAR):
    self.prefix_offset = self.read_offset
    self.read_offset = len(self.tokens)
    return new_text[len(prefix_text):]          # the safe-to-emit delta
return ""                                        # incomplete multibyte char — wait
```

`flush()` drains whatever is held back at end-of-stream. Using `convert_tokens_to_string` (not
`decode`) means the same code is correct for SentencePiece too (it handles the `▁` space marker).

**Why it matters.** This is the hardest correctness claim in the project and the reason the whole
streaming-text-on-TRT-LLM thesis holds. It's proven on a laptop with no GPU — see feature 13.

---

## 6. Cross-token stop sequences

**What.** Honor client stop strings (e.g. `</tool_call>`) even when the string is split across
streamed text deltas.

**Where.** `src/text_io/stop.py`.

**How.** Detect the stop in *decoded text*, not token space. Hold back the longest suffix of the
buffer that *could* still grow into a stop string, so you never emit characters that should have
been truncated:

```python
# stop.py → feed()
self.buffer += text
hit_idx = None                                       # 1) is a COMPLETE stop present?
for s in self.stops:
    i = self.buffer.find(s)
    if i != -1 and (hit_idx is None or i < hit_idx):
        hit_idx = i
if hit_idx is not None:
    emit = self.buffer[:hit_idx]; self.buffer = ""
    return emit, True                                #    emit up to it, signal stop
hold = self._partial_suffix_len()                    # 2) longest suffix that is a stop PREFIX
if hold:
    emit, self.buffer = self.buffer[:-hold], self.buffer[-hold:]   # hold it back for next delta
else:
    emit, self.buffer = self.buffer, ""
return emit, False
```

**Why.** A naive "is the stop string in this delta?" check misses `</too` + `l_call>` arriving in
two pieces. The held-back suffix is the crux, and it's unit-tested directly (feature 13).

---

## 7. `finish_reason` classification

**What.** Report *why* generation ended: `stop` (EOS or a stop string), `length` (hit the budget),
or `content_filter` (guard block).

**Where.** `src/text_io/finish.py` — a pure function; wired in `model.py → _handle()`.

**How.** The loop tracks two facts — how many tokens the engine emitted, and whether a stop string
matched — then a pure function maps them to the label:

```python
# finish.py
def classify_finish_reason(*, stop_string_hit: bool, generated: int, max_tokens: int) -> str:
    if stop_string_hit or generated < max_tokens:   # stop-string, or halted early at EOS
        return "stop"
    return "length"                                  # only a full budget is "length"
```

```python
# model.py → _handle()  (after the stream loop)
finish = classify_finish_reason(stop_string_hit=stop_string_hit, generated=generated, max_tokens=max_tokens)
```

**Why this way (RCA §9).** Before this fix, `finish` defaulted to `"length"` and only flipped on a
stop string — so a short answer that stopped at EOS was mislabeled `length`. The decision is a
pure function in `src/text_io` (not buried in the BLS) specifically so it's covered by a GPU-free
unit test (feature 13). `content_filter` is set directly by the guard short-circuit (feature 2),
so it's out of scope here.

---

## 8. KV-cache reuse — build flag + serve flag

**What.** Reuse the KV cache for a shared prompt prefix across requests, so repeated
system/few-shot prefixes aren't recomputed.

**Where.** Two places — it takes both:

```bash
# scripts/build_engines.sh — make reuse POSSIBLE at build time
trtllm-build ... --use_paged_context_fmha enable --paged_kv_cache enable \
             --max_input_len 4096 --max_seq_len 8192
```
```pbtxt
# tensorrt_llm_small/config.pbtxt — turn it ON at serve time
parameters: { key: "enable_kv_cache_reuse"          value: { string_value: "true" } }
parameters: { key: "kv_cache_free_gpu_mem_fraction" value: { string_value: "0.25" } }   # large uses 0.45
parameters: { key: "enable_chunked_context"         value: { string_value: "true" } }
parameters: { key: "gpt_model_type"                 value: { string_value: "inflight_fused_batching" } }
parameters: { key: "batch_scheduler_policy"         value: { string_value: "max_utilization" } }
```

**Why.** Paged-context FMHA at build time is what lets the runtime hash and share KV blocks;
`enable_kv_cache_reuse` is the runtime switch. `kv_cache_free_gpu_mem_fraction` splits the 24 GB
L4 between the two engines (0.25 for 0.5B, 0.45 for 1.5B) so both + the guard fit (~17.9 GB).
Server-side templating (feature 3) is what keeps the shared prefix block-aligned.

---

## 9. Engine build — the version-specific bits

**What.** Convert HF Qwen2.5 → TRT-LLM checkpoint → compiled FP16 engine.

**Where.** `scripts/build_engines.sh` (runs inside the NGC TRT-LLM container).

**The two Qwen-specific fixes baked into the script (RCA §1–§3):**

```bash
# 1) NGC 0.14.0 library bug: qwen/model.py calls check_share_embedding() with no arg.
sed -i "s/loader.check_share_embedding()/loader.check_share_embedding(config)/g" "$QWEN_MODEL"

# 2) Qwen2.5 ties word embeddings (no separate lm_head.weight) → need --use_embedding_sharing.
python3 "$EX/convert_checkpoint.py" --model_dir "hf_models/$2" \
    --output_dir "ckpt/$1" --dtype float16 --use_embedding_sharing
```

**Why it's a script, not a Dockerfile step.** Engine building is version-locked: the example
`convert_checkpoint.py` must match the TRT-LLM in the image (`TRTLLM_REF=v0.14.0` ↔ 24.10), and
the engines are large artifacts that live in `./engines/` (a volume mount), not in the image.

---

## 10. Readiness gate — fail fast before pulling 10 GB

**What.** A dependency-free preflight that confirms the box can run the stack.

**Where.** `scripts/check_env.py` (Python stdlib only; `--json` for agents, `--strict` to treat
warnings as failures).

**How.** Each check appends `{name, status, detail, fix}`; the script is READY iff no `FAIL`:

```python
add("GPU VRAM", OK if vram >= MIN_VRAM_GB else FAIL, f"{vram:.1f} GB", f"need >= {MIN_VRAM_GB} GB")
add("Triton ports 8000/8001/8002", WARN if busy else OK, ...)
ready = not any(c["status"] == FAIL for c in checks)
return 0 if ready else 1
```

It checks GPU model/VRAM/driver/compute-cap (via `nvidia-smi`), the Docker daemon + NVIDIA runtime,
the three Triton ports, and free disk. **Why:** turns "why won't it start?" into a 1-second
answer with a fix string for each problem, and gives agents a machine-readable gate.

---

## 11. Observability — metrics + per-stage traces

**What.** TTFT/ITL, KV-cache health, GPU power, and per-pipeline-stage latency on live dashboards.

**Where.** `observability/` (a separate compose stack).

**How.** Three Prometheus scrape jobs and an OTel collector that does double duty:

```yaml
# observability/prometheus.yml
- job_name: "triton"          # native Triton metrics: per-model latency, queue, KV-cache blocks
  static_configs: [{ targets: ["host.docker.internal:8002"] }]
- job_name: "dcgm"            # GPU utilization / memory / POWER → tokens/s/W
  static_configs: [{ targets: ["dcgm-exporter:9400"] }]
- job_name: "otel-spanmetrics"  # RED metrics derived from spans
  static_configs: [{ targets: ["otel-collector:8889"] }]
```

Triton emits OpenTelemetry spans (enabled by `--trace-config mode=opentelemetry` in
`start_server.sh`); the collector forwards them to Jaeger **and** derives per-stage latency metrics
via its `spanmetrics` connector:

```yaml
# observability/otel-collector-config.yaml
connectors:
  spanmetrics: { histogram: { explicit: { buckets: [2ms, 5ms, ... , 5s] } } }
# NOTE: service.name/span.name/span.kind/status.code are IMPLICIT dimensions —
# re-declaring them fails validation (RCA §8). Add only EXTRA dimensions here.
```

**Why.** One trace pipeline gives you both the Jaeger waterfall (which stage was slow on *this*
request) and Prometheus histograms (p50/p99 per stage over time) — without instrumenting the
gateway by hand.

---

## 12. The streaming client

**What.** Consume the decoupled gRPC stream: print TEXT deltas as they arrive, then the
`finish_reason`.

**Where.** `client/client_fused.py`. (`client/guard_probe.py` is the guardrail test harness.)

**How.** `start_stream` with a callback that drops `(result, error)` onto a queue; read until the
response carrying `FINISH_REASON` (only the final one has it):

```python
client.start_stream(callback=lambda result, error: results.put((result, error)))
client.async_stream_infer("text_pipeline_bls", inputs=inputs, outputs=outputs, request_id="1")
while True:
    result, error = results.get()
    text = result.as_numpy("TEXT")
    if text is not None and text.size:
        sys.stdout.write(text[0].decode("utf-8")); sys.stdout.flush()
    fr = result.as_numpy("FINISH_REASON")
    if fr is not None and fr.size:               # final response only
        finish = fr[0].decode("utf-8"); break
```

**Why.** It demonstrates the contract: the client sends *raw messages* and does **zero**
tokenization/templating/detok — the gateway owns all of it.

---

## 13. Test strategy — "single source of truth" makes the proof GPU-free

**What.** The hard correctness claims (byte-exact detok, cross-token stops, finish_reason) are
unit-tested on a laptop, with no GPU and no model download.

**Where.** `src/text_io/*` is imported by **both** the BLS (serving) and `tests/*` (tests), so
*live behavior == tested behavior*. `make test` runs all three suites.

**How.** The detok test fakes just enough tokenizer to reproduce the `�` failure mode — a
byte-level vocab where a multibyte glyph is split across tokens (`tests/_fake_tokenizer.py`):

```python
# tests/test_detokenize_incremental.py
RAW = "你好🚀".encode("utf-8"); SPLITS = [2, 4, 6, 8]      # split mid-character (worst case)
streamed = "".join(detok.add([i]) for i in ids) + detok.flush()
assert streamed == "你好🚀" and "�" not in streamed        # incremental: correct
# ...and the contrast test asserts naive per-token decode DOES emit "�" (the bug we fix).
```

`tests/test_stop.py` feeds a stop string split across two `feed()` calls; `tests/test_finish.py`
checks the `(stop_string_hit, generated, max_tokens)` truth table. **Why:** the riskiest logic is
pure Python with zero infra, so a contributor can verify the centerpiece in one second and trust
that the server runs the identical code.

---

## Where to look next

- **Run it / operate it:** [`README.md`](../README.md), [`REPORT.md`](./REPORT.md) §4.
- **Every bug we hit and why (EN/中文):** [`RCA.md`](./RCA-EN.md).
- **Guardrail behavior, measured:** [`GUARDRAILS.md`](./GUARDRAILS.md).
- **Tune it:** the `config.pbtxt` knobs table in [`README.md`](../README.md).
