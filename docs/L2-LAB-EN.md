# L2 Inference-Engine Lab — Ablate-on-your-own-rig

> 🌐 **中文版 (Chinese):** [`L2-LAB-CN.md`](./L2-LAB-CN.md)

> **This doc is the "how-to" execution manual; the direction/roadmap lives in [`AI-INFRA-DIRECTION-EN.md`](./AI-INFRA-DIRECTION-EN.md).**
> That one says "sink down to the engine layer, learn the internals from vLLM, deepen on TRT-LLM/Dynamo on home turf"; **this one turns that into an ablation lab you can run today.**
>
> **Bottom line first:** You're sitting on a rig where **every core L2 knob is already ON, and most of them can be turned OFF by editing `config.pbtxt` + `docker restart`.** The L2 roadmaps online put the spine on "read vLLM source + build toys from scratch"; for **you, with this machine,** the right spine is **ablate on your own system**: turn one knob off → predict the number first → measure the regression → then read the vLLM source that explains the delta you just measured. Reading source and writing toys drop from "the main line" to "the explanation layer." "You've only arrived when you can change it" — you don't have to wait until you can change vLLM; you can change your own serving right now, one knob at a time: change one, measure one, explain one.

---

## 0. Diagnosis: what you already have vs the roadmap's blind spot

| The roadmap assumes you | What you actually have (confirmed) |
|---|---|
| "you're on FP16, nothing enabled" | the engine config has **every core L2 knob ON**: `inflight_fused_batching`, `paged_kv_cache`, `use_paged_context_fmha`, `enable_kv_cache_reuse`, `enable_chunked_context`, `batch_scheduler_policy=max_utilization` |
| go read source elsewhere, build toys from scratch | a **running** Triton + TRT-LLM stack (`triton-llm`), dual engines (0.5B/1.5B), full observability (Prometheus/Grafana/Jaeger/DCGM) online |
| defer measurement until M1 | configs are **volume-mounted** → change a param, `docker restart triton-llm`, done — **ablation cost ≈ 0** |
| — | the `client/perf_benchmark.py` + `docs/PERFORMANCE.md` that `AI-INFRA-DIRECTION` repeatedly cites **do not exist in the working tree** (promised, never built). **The measurement spine is the one remaining vacuum.** |

---

## 1. Five principles

1. **The running machine is the lab, not a footnote.** Every concept lands on a number you measured on your own L4.
2. **Ablate to understand.** The knobs are ON; the fastest way to understand one is to **turn it off and watch what gets slower/worse.**
3. **Predict, then measure.** Before you act, write down the hypothesis + an order-of-magnitude roofline estimate; **the gap between predicted and measured is the learning signal.** Log every experiment in [`lab-notebook/`](./lab-notebook/).
4. **vLLM source is the "why," not the main line.** Reading it with a delta you measured in hand is ten times faster than reading cold.
5. **Toys must be calibrated against the rig.** A toy scheduler/allocator has to reproduce a curve you measured; reproducing it (or failing to, explainably) is what makes it a real artifact.

---

## 2. Knob map: runtime (edit config → restart) vs build (rebuild the engine)

**This is L2's first watershed**: for the same "feature," some are runtime scheduler behavior (just change a param), some are baked into the engine graph (must rebuild). Get the boundary right and ablation cost + ordering become obvious.

| Knob | Type | Where | Note |
|---|---|---|---|
| `gpt_model_type` (inflight_fused_batching ↔ `V1` static batching) | **R** | `config.pbtxt` | V1 may be deprecated in 0.14 → after changing, check `docker logs` to confirm it was accepted |
| `batch_scheduler_policy` (max_utilization ↔ guaranteed_no_evict) | **R** | `config.pbtxt` | controls how aggressively it admits/evicts |
| `enable_kv_cache_reuse` (true ↔ false) | **R** | `config.pbtxt` | prerequisite: engine built with `--use_paged_context_fmha enable` (yours is) → so it's runtime-switchable |
| `enable_chunked_context` (true ↔ false) | **R** | `config.pbtxt` | same prerequisite |
| `kv_cache_free_gpu_mem_fraction` (0.25/0.45 → sweep 0.1~0.9) | **R** | `config.pbtxt` | directly sets KV pool size → max concurrency |
| `max_tokens_in_paged_kv_cache` / `max_num_tokens` | **R** | `config.pbtxt` | another set of KV/scheduling ceilings, to contrast with fraction |
| `paged_kv_cache` / `use_paged_context_fmha` (enable ↔ disable) | **B** | `build_engines.sh` | turning off = revert to non-paged KV; expensive, but shows you exactly what PagedAttention buys |
| `max_batch_size` (=64) | **B** | `trtllm-build` | **the current hard concurrency ceiling**, see §5 M0 |
| `max_beam_width` (=1 → 4) | **B** | `trtllm-build` (you didn't pass it, defaults to 1) | beam>1 requires a rebuild |
| dtype / quantization (FP16 → FP8 / INT4-AWQ) | **B** | `convert_checkpoint.py` + `trtllm-build` | M4 home turf |

> Suggested order: **sweep all the R rows first (5 min each: change param → restart → measure), then do the B rows (rebuilds take hours).**

---

## 3. ★ The core artifact: the ablation matrix

Each row = one L2 subsystem. **Fill the "predict" column first (write the number + mechanism), then go measure.** Template: [`lab-notebook/TEMPLATE-EN.md`](./lab-notebook/TEMPLATE-EN.md).

| Knob (current) | Off/change to | R/B | Predict first (number + mechanism) | Measure | Where in vLLM |
|---|---|---|---|---|---|
| `gpt_model_type=inflight_fused_batching` | `V1` (static batch) | R | at concurrency=32, throughput drops ~?×: static batching waits for the **longest** sequence in the batch to finish before admitting new requests → head-of-line blocking, GPU idles on finished slots | throughput, TTFT P50/P99 vs concurrency | `vllm/v1/core/sched/scheduler.py` `schedule()` (iteration-level) |
| `enable_kv_cache_reuse=true` | `false` | R | with a **shared 200-tok system prompt** workload, reuse ON cuts TTFT by ~(shared/total); OFF → prefill recomputed, TTFT rises | TTFT on a shared-prefix workload; KV `reused` metric | `vllm/v1/core/kv_cache_manager.py` + block hashing |
| `enable_chunked_context=true` | `false` | R | OFF: one 3500-tok long prefill blocks other streams' decode → ITL spikes on in-flight streams; ON: prefill is chunked and interleaved with decode, ITL stays flat | ITL jitter of a decode stream (while a long prefill is admitted) | chunked-prefill in the scheduler / `long_prefill_token_threshold` |
| `batch_scheduler_policy=max_utilization` | `guaranteed_no_evict` | R | max_util admits aggressively and may evict/recompute under KV pressure (higher throughput, occasional recompute tail latency); no_evict is conservative (lower throughput, no eviction stalls) | throughput vs tail latency under KV pressure (high concurrency + long sequences) | scheduler preemption: recompute vs swap |
| `kv_cache_free_gpu_mem_fraction` | sweep 0.1→0.9 | R | max concurrent sequences ≈ proportional to KV blocks; too low → queueing (throughput cliff), too high → OOM / steals the other engine's memory | max concurrency before queueing, throughput vs fraction | block pool sizing / `num_gpu_blocks` |
| `max_batch_size=64` | rebuild=16 / 128 | B | at short context it's **the real concurrency ceiling** (see M0); changing it shifts the throughput ceiling directly | throughput vs max_batch_size (two curves: short / long context) | scheduler `max_num_seqs` |
| `max_beam_width=1` | rebuild=4 | B | beam=4 makes KV+decode ~4×, throughput drops ~4×; quality? | throughput, output quality | beam kernels (read to understand) |
| dtype=FP16 | **FP8** (Ada supports) / **INT4-AWQ** | B | weight-only INT4 mainly speeds up **decode** (small-batch decode is bandwidth-bound, weight traffic dominates), **TTFT barely benefits** (prefill is compute-bound); FP8 KV → 2× KV capacity → higher concurrency | tok/s, TTFT, accuracy delta | — (TRT-LLM home turf) |

---

## 4. Measurement rigor (the watershed between L2 and "can tweak an API")

1. **Measure the right layer.** The path is `BLS (2 CPU instances) → guardrail (1 GPU instance) → engine`. Measuring through `text_pipeline_bls` gives you the **convolution of gateway + guard + engine** (your `AI-INFRA-DIRECTION` §1.1 already caught it: "guard is single-instance serial, BLS instance count caps engine batch"). To study the **engine's** scheduler/KV, **hit the `tensorrt_llm_small/large` model directly** (raw `input_ids` in, bypassing BLS/guard), or you'll charge the gateway's serialization to the engine. Tools: Triton `perf_analyzer` or TRT-LLM `gptManagerBenchmark`.
2. **Design the workload to expose the knob.** KV reuse is only visible with a **shared prefix**; chunked context only shows its ITL spike with **one long prefill + several in-flight decodes**. **"Designing a load that exposes the knob" is half the experiment.**
3. **L4 = single GPU, no NVLink, Ada (sm_89).** FP8 works (Ada has FP8 tensor cores); **FP4 is Blackwell-only → read-only, can't run it**; TP/PP won't run on one GPU → the roadmap's "#6 understanding the principle is enough" is right — don't spend hands-on budget here.
4. **Small-model/big-memory trap.** 0.5B/1.5B on an L4 — decode can be so cheap you're **bottlenecked by Python/gRPC overhead, not the GPU.** You have to push concurrency until the GPU is saturated (watch DCGM SM utilization) before a batch sweep means anything.
5. **Methodology:** warm up; report median + P99 (N ≥ a few dozen); `docker stop triton-fused` first to clear the field (shared ports + memory); record DCGM power / SM utilization as corroboration.

---

## 5. Milestones (rewritten: each carries one ablation + one artifact)

### M0 · Foundations — **already closed with your real numbers (see [`lab-notebook/0001-m0-kv-memory-EN.md`](./lab-notebook/0001-m0-kv-memory-EN.md))**

KV/token formula: `2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes` (Qwen2.5 is **GQA**, `num_kv_heads=2` ≪ attention heads).

| | num_layers | num_kv_heads | head_dim | **predicted B/token** | engine measured (startup log) | gap |
|---|---|---|---|---|---|---|
| **0.5B** (14 heads, hidden 896) | 24 | 2 | 64 | **12,288** | 2.18 GiB / 190,656 tok = 12,277 | **0.09%** |
| **1.5B** (12 heads, hidden 1536) | 28 | 2 | 128 | **28,672** | 7.31 GiB / 273,792 tok = 28,669 | **0.01%** |

**Artifact 1 (pass):** your hand-calc predicts the engine's actual KV allocation to **3 significant figures**. Verify:
```bash
docker logs triton-llm 2>&1 | grep -iE 'blocks in KV cache|max tokens in paged|maxNumSequences'
# Allocated 7.31 GiB for max tokens in paged KV cache (273792)   ← 1.5B, fraction 0.45 × 16.25 GiB avail
# Allocated 2.18 GiB for max tokens in paged KV cache (190656)   ← 0.5B, fraction 0.25 × 8.73  GiB avail
```

**Artifact 2 (the L2 insight the roadmap never teaches — "which resource tops out first"):**
- KV capacity ceiling: the 1.5B can hold 273,792 tok → ≈ **133** concurrent sequences at 2K context.
- But build-time `max_batch_size=64` → log says `maxNumSequences: 64`. **Two ceilings; at short context, batch_size=64 tops out first, not KV.**
- Crossover: `273,792 / 64 ≈ 4278 tok`. **Average context < 4278 → batch_size-bound; > 4278 → KV-memory-bound.** This is exactly why the ablation-matrix rows `max_batch_size` (B) and `kv_cache_free_gpu_mem_fraction` (R) should be swept together.
- Third lesson: across the two KV computations, `available` drops from **16.25 → 8.73 GiB** (the first engine eats weights+KV first) → **when co-locating models, load order decides how much the second engine gets.**

**Artifact 3:** use the numbers above to explain "why bigger batch raises throughput but worsens latency" + "why decode is bandwidth-bound" (roofline). Being able to explain it = M0 pass.

### M1 · Serving-layer comparison + metrics (ablation: none, establish baseline)
Same model, same machine: stand up vLLM and your TRT-LLM, measure TTFT/ITL/throughput/P99 directly at the engine layer. **Artifact:** a vLLM vs TRT-LLM comparison table + tradeoff analysis. **Prerequisite = the measurement spine (§6).**

### M2 · Scheduler + continuous batching ★most important (ablation: `gpt_model_type` inflight→V1, `batch_scheduler_policy`, `max_batch_size`)
Read `vllm/v1/core/sched/`, against your measured throughput collapse from inflight→V1. **Artifact:** a minimal pure-Python continuous-batching simulator — given a batch of requests of varying lengths + a KV budget, simulate per-iteration admit/evict and how prefill/decode mix. **Calibration:** it must reproduce the throughput-vs-concurrency curve you measured on your own engine (including the batch_size=64 knee).

### M3 · KV-cache / PagedAttention (ablation: `enable_kv_cache_reuse`, `kv_cache_free_gpu_mem_fraction` sweep, `paged_kv_cache` at the B level)
Read the block manager + prefix caching, against your measured TTFT for `enable_kv_cache_reuse` on a shared-prefix workload. **Artifact:** a toy paged KV block allocator + prefix-cache hit logic (block table, refcount, hit rate). **Calibration:** your hit-rate prediction matches the engine's KV `reused` metric.

### M4 · Quantization + speculative decoding (ablation: dtype at B level FP8/INT4-AWQ; draft model)
Rebuild TRT-LLM in FP8 / INT4-AWQ, measure accuracy vs throughput (note: the speedup is mostly in decode); then enable speculative decoding and measure the speedup ratio. **Artifact:** a quantization + speculative-decoding measurement report.

### M5 · Ongoing (TP/PP read-only + open-source contribution)
L4 can't run TP/PP → read the principles + Dynamo's disaggregated P/D. Land a mergeable PR in vLLM (benchmark/docs/small fix).

---

## 6. The measurement spine (the current vacuum — prerequisite for every milestone)

The `client/perf_benchmark.py` that `AI-INFRA-DIRECTION` promises doesn't exist. Three pieces needed:
1. **Async load generator** — can hit `tensorrt_llm_{small,large}` directly (bypassing BLS/guard, see §4.1) or go through the BLS; configurable concurrency C, input/output length distributions, shared-prefix ratio.
2. **Metric collection** — from the decoupled stream, compute TTFT (first TEXT delta) / ITL (delta interval) / throughput / P50/P99; pull Prometheus + DCGM at the same time.
3. **Ablation runner** — sweep one knob → edit `config.pbtxt` → `docker restart triton-llm` → wait for ready → measure → emit a table.

Metrics quick reference:
```bash
curl -s localhost:8002/metrics | grep -iE 'trt_llm|inference_request_duration'   # engine/KV/inflight-batcher metrics (confirm the exact names on your build)
docker logs triton-llm 2>&1 | grep -iE 'blocks in KV cache|maxNumSequences'      # startup KV / concurrency ceilings
```

---

## 7. Toy artifacts (calibrated against the rig, not built in a vacuum)

- **M2 continuous-batching simulator**: input a batch of `(arrival, prompt_len, output_len)` + KV budget + max_batch_size, output per-iteration batch occupancy / admit / evict, plus total throughput and per-request TTFT/finish time. **Pass condition:** reproduce the shape of your measured throughput-vs-concurrency curve + the batch_size=64 knee.
- **M3 paged KV allocator**: block table + refcount + prefix hits, with hit-rate stats. **Pass condition:** given a shared-prefix workload, your predicted hit rate matches the engine's KV `reused` metric.

> When a toy can't reproduce reality, **don't force it** — the gap itself is the core lesson of M2/M3 (the real scheduler also does things your toy didn't model: chunking, recompute, fragmentation). Write the gap into the lab notebook.

---

## 8. Relationship to this project

- **Decoupling:** `text_pipeline_bls` only consumes a compiled engine. R-level knobs: edit `config.pbtxt` + restart; B-level: rebuild the engine via `build_engines.sh`, serving untouched.
- **This repo = an end-to-end lab**, but **to measure the engine you must bypass the gateway layer** (§4.1).
- **Build reality:** building TRT-LLM from source is heavy (M4 quantization / B-level ablations go inside the build container); R-level ablations need only the existing `.venv` + restart.

> In one line: **direction in `AI-INFRA-DIRECTION`, execution here; the spine is "ablate on your own machine," M0 is already closed with real numbers, and the next gap is the measurement spine (§6).**
