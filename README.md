# triton-trtllm-text-io

**A production LLM inference gateway on Triton + TensorRT-LLM, for a single NVIDIA L4.**
Send chat `messages`; get a correct, **streamed** text response — with **routing**,
**KV-cache reuse**, **integrated input/output guardrails**, and full **observability** built
in. The serving layer that turns a raw `token_ids → token_ids` TRT-LLM engine into the
endpoint users actually expect.

> Design & experiment spec: [`../triton-llm-prepost-backend-l4.md`](../triton-llm-prepost-backend-l4.md)
> (Part I = the streaming text-I/O core; **Part II** = the production features this repo adds).
> LLM sibling of the CV project `triton-fused-prepost-backend` (image→boxes): same
> "fuse pre/post into the server" thesis, on the hardest axis — streaming text on TRT-LLM.

---

## Architecture

```
                       text_pipeline_bls  (the gateway: BLS, decoupled streaming)
 MESSAGES ─► route ─► input guard ─► chat-template+tokenize ─► tensorrt_llm_{small|large} ─► detok+stop ─► output guard ─► TEXT stream
              (II.3)     (II.4)          (server-side)             (KV reuse · II.2)        (incremental)   (chunked)
        ▲ Prometheus :8002   ▲ DCGM :9400 (power)   ▲ OpenTelemetry per-stage traces            (II.5 observability)
```

| Model | Role |
|---|---|
| `text_pipeline_bls` | **Call this.** Routing → guard → tokenize → engine → streaming detok → output guard |
| `guardrail` | Small co-located classifier (`MODE=input\|output` → `BLOCKED/CATEGORY/SCORE`) |
| `tensorrt_llm_small`, `tensorrt_llm_large` | TRT-LLM engines (KV-cache reuse on); the routing targets |

---

## Run it on your L4 server

**Prerequisites:** Ubuntu L4 box with a recent NVIDIA driver (≥535), Docker, and the
NVIDIA Container Toolkit. ~40 GB free disk. A Hugging Face login is **not** required
(all models are ungated).

```bash
# 0. Clone
git clone <YOUR_REPO_URL> triton-trtllm-text-io
cd triton-trtllm-text-io

# 1. Readiness gate — confirms GPU=L4, VRAM≥22 GB, driver, Docker, toolkit, ports, disk.
python3 scripts/check_env.py            # must print READY ✅  (exit 0). --json for agents.

# 2. Prove the streaming-detok + stop logic on the box itself — NO GPU needed.
#    (Reconstructs CJK/emoji byte-exact; shows naive per-token decode emits �.)
make test                                # or: python3 tests/test_detokenize_incremental.py

# 3. Install host deps + download Qwen2.5-0.5B & 1.5B Instruct (Apache-2.0, ~4 GB).
make setup                               # bash scripts/setup.sh

# 4. Build the two TRT-LLM engines (FP16, paged-context FMHA so KV reuse works).
make engines                             # bash scripts/build_engines.sh   (adjust TRTLLM_REF to your version)

# 5. Build the custom Triton image and launch the server (HTTP 8000 / gRPC 8001 / metrics 8002).
make server                              # bash scripts/start_server.sh

# 6. (another shell) Observability: Prometheus + Grafana + OTel + DCGM.
make obs-up                              # Grafana http://localhost:3000  (dashboard auto-loaded)

# 7. (another shell) Stream: raw messages in, text out.
make client M="用三句话介绍一下 GPU 推理 🚀"
#   or:  python3 client/client_fused.py --message "explain KV cache" --model large --stop "</done>"
```

`make help` lists every target.

### What each step gives you

| Step | Proves |
|---|---|
| 2 | The hard correctness claim (streaming detok) — verifiable with no GPU |
| 4 | You can build TRT-LLM engines with KV-cache reuse enabled |
| 5–7 | The full gateway: routing + guardrails + streaming end to end |
| 6 | TTFT/ITL, KV-cache health, GPU power (→ tokens/s/W), per-stage traces on live dashboards |

---

## Repo layout

```
model_repository/
  text_pipeline_bls/      gateway orchestrator (routing + guard cascade + streaming detok)
  guardrail/              input/output safety classifier (ungated, ~110-184M)
  tensorrt_llm_small/     0.5B routing target (config; engine lives in ./engines)
  tensorrt_llm_large/     1.5B routing target
src/text_io/              detok / stop — single source of truth (imported by the BLS + tests)
client/client_fused.py    gRPC streaming client (messages in, text out)
scripts/                  check_env · setup · build_engines · start_server
tests/                    GPU-free unit tests (byte-exact CJK/emoji detok, cross-token stops)
observability/            Prometheus + Grafana + OTel Collector + Jaeger + DCGM (compose)
Dockerfile                NGC TRT-LLM base + backend deps + src
```

## Tuning knobs (in the `config.pbtxt` files)

| Where | Knob | Effect |
|---|---|---|
| `text_pipeline_bls` | `ENABLE_GUARDRAILS`, `OUTPUT_GUARD_WINDOW_CHARS` | toggle safety; moderation window vs. TTFT (E10) |
| `text_pipeline_bls` | `SMALL_MODEL` / `LARGE_MODEL` | routing targets |
| `tensorrt_llm_*` | `enable_kv_cache_reuse`, `kv_cache_free_gpu_mem_fraction` | prefix reuse (E8); VRAM split on the 24 GB L4 |
| `guardrail` | `INPUT_MODEL` / `OUTPUT_MODEL` / `BLOCK_THRESHOLD` | swap in Llama Guard 3 / Granite Guardian; sensitivity |

## Status & caveats

- ✅ **Validated locally (no GPU):** the streaming detok + stop unit tests pass (`make test`),
  all configs/JSON parse, both Python backends compile.
- ⚠️ **Run the GPU path on the L4** — it can't run on a CPU-only laptop. The TRT-LLM tensor
  names, the `enable_kv_cache_reuse` key, and the `convert_checkpoint.py` example path are
  **version-specific**; pin `TRITON_TAG` / `TRTLLM_REF` and verify against your image.
- The `guardrail` model downloads its classifiers from Hugging Face on first load (or
  pre-bake them in the `Dockerfile`).
