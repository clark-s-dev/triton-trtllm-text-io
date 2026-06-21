# AI-infra direction notes — the inference engine layer (vLLM / TensorRT-LLM) + a kernel-layer map

> 🌐 **中文版 (Chinese):** [`AI-INFRA-DIRECTION-CN.md`](./AI-INFRA-DIRECTION-CN.md)

> Starting from this repo's **serving layer**, this records the map and route for sinking down into the "engine layer (AI infra)" and the "kernel layer."
> The kernel-layer map is based on this repo's submodule **TensorRT-LLM v0.14.0** (commit `b088016`), from read-only investigation.
> This is a **direction/route note**, not a project deliverable.
> Note: the perf benchmarks referenced here (`docs/PERFORMANCE.md` + `client/perf_benchmark.py`) currently live on the `feat/perf-benchmark` branch, not yet merged into main.

---

## 0. One picture: the three layers of the inference stack, and "where am I"

| Layer | Examples | Relation to my current state |
|---|---|---|
| ① serving / orchestration | NVIDIA Triton Inference Server, KServe, Ray Serve, routing / multi-model | ✅ been doing it all along (this repo's `text_pipeline_bls` gateway) |
| ② **engine / runtime** | **vLLM, TensorRT-LLM, SGLang** | ⬅️ **the "AI infra" I want to focus on is this layer** |
| ③ kernel / operator | CUDA / CUTLASS / OpenAI-Triton, FlashAttention | 🤔 the open question of whether to sink further |

**Bottom line first:** "AI infra (vLLM/TRT-LLM)" = the **engine runtime layer**, sandwiched between the serving layer I do now and the kernel layer; its core capability in one phrase — **inference performance engineering** (continuous batching, paged KV cache, scheduling, quantization, speculative decoding, distributed TP/PP/EP).

> ⚠️ Don't confuse the two "Tritons": this repo uses **NVIDIA Triton *Inference Server*** (a serving framework); kernel dev usually means **OpenAI Triton** (a DSL for writing kernels). The "fused" in `triton-*` / `*-fused-*` means "fuse pre/post into the server" (Python/BLS layer), **not kernel fusion**.

---

## 1. The engine layer = where AI infra lands (recommended direction)

### 1.1 Not a from-scratch pivot, a "half-layer sink"
What's already banked and **directly transferable**:
- TRT-LLM engine build, the `tensorrtllm` backend, KV-cache reuse, streaming detok — already **using** this layer.
- This **TTFT / E2E / throughput benchmark + bottleneck localization** (guard single-instance serial, BLS instance count capping engine batch, see `PERFORMANCE.md`) — the mindset of "find the bottleneck in the stack" **is itself the core of infra work.**
- At NVIDIA = **TRT-LLM / Dynamo / NIM home turf.**

The only thing that really has to change: **from "treat the engine as a black box and tune params" → "understand the engine internals — able to change, tune, and compare."**

### 1.2 Four concrete moves
1. **Learn engine internals from vLLM source** (open-source, readable Python): continuous batching / PagedAttention / scheduler / block manager. TRT-LLM is strong but a compiled black box, good for "home-turf depth"; vLLM is good for "seeing the principle clearly." Complementary — do both.
2. **Build up the performance methodology**: memory-bound decode vs compute-bound prefill, roofline, goodput, the batch-latency tradeoff; tools **Nsight Systems / Compute** + torch profiler. This is the watershed between infra and "can tweak an API."
3. **Watch the problems that define this field right now (2026), pick one and master it:**
   - **Disaggregated prefill/decode** (NVIDIA **Dynamo** is built around it; vLLM has P/D too)
   - prefix caching, chunked prefill, speculative decoding
   - **FP8 / FP4 quantization** (Blackwell FP4), large-scale **MoE** serving, KV cache offload
   - multi-GPU **TP / PP / EP** + NCCL
4. **Downgrade the kernel line to "can read"**: infra needs to "know which kernel is the bottleneck, and be able to integrate FlashAttention / FlashInfer / CUTLASS," but not write from scratch. **The engine layer is the sweet spot between operators and serving** — more systems/perf-flavored, no need to sink as deep as CUTLASS.

### 1.3 The lowest-effort entry: use the existing repo as a bridge
> **Add a vLLM backend to the existing gateway, side by side with TRT-LLM, and A/B vLLM vs TRT-LLM with `client/perf_benchmark.py`.**

One project proves all three at once: ① already know the serving layer ② can get hands on vLLM ③ can do engine-level perf comparison. Then dig deep into one feature (prefix caching / chunked prefill / FP8) and **quantify** the gain — the direction is set.

### 1.4 Hard skills to build (honest list)
Engine source reading (vLLM first) · GPU profiling (Nsight) · distributed inference (TP/PP/EP, NCCL) · quantization runtimes (FP8/AWQ/GPTQ). Operators only need "read + integrate," not "write from scratch."

### 1.5 NVIDIA-internal vs general
- **Internal home turf**: TensorRT-LLM, **Dynamo** (distributed/disaggregated serving), NIM (model microservices).
- **General / job market**: vLLM, SGLang — also **the best reading for learning engine internals.**

---

## 2. Kernel-layer map (if sinking further) — TensorRT-LLM v0.14.0

### 2.1 The 5-layer chain for an operator, from the bottom up to "callable in a model"
| Layer | Path | What it does | File count |
|---|---|---|---|
| ① CUDA kernel | `cpp/tensorrt_llm/kernels/<class>/` | the actual compute | **1664** |
| ② TRT plugin | `cpp/tensorrt_llm/plugins/<name>Plugin/` | wrap a kernel as a TensorRT layer | 64 |
| ③ plugin registration | `cpp/tensorrt_llm/plugins/api/tllmPlugin.cpp` | register the plugin creator with TRT | 1 |
| ④ Python binding | `tensorrt_llm/functional.py` + `tensorrt_llm/plugin/plugin.py` | build the graph with `trt.PluginField` | — |
| ⑤ used in a model | `tensorrt_llm/layers/` → `tensorrt_llm/models/<model>/model.py` | assemble into a model | — |

> **Parallel bypass** `cpp/tensorrt_llm/thop/` (14 ops, e.g. `dynamicDecodeOp` / `weightOnlyQuantOp` / `fp8Op`): if you want a custom op **called directly from PyTorch** (TRT-LLM's PyTorch flow), go here, not through plugins.

Each plugin = a directory with three files (the skeleton `identityPlugin` is just 198 lines): `XxxPlugin.h / .cpp / CMakeLists.txt`; the `.cpp` must implement the `IPluginV2DynamicExt` set (`clone / getOutputDimensions / supportsFormatCombination / configurePlugin / getWorkspaceSize / enqueue / serialize`), where **`enqueue` is where the kernel is actually launched.**

### 2.2 Kernel directory taxonomy (center of gravity at a glance)
| kernel dir | what it is | perf weight |
|---|---|---|
| `decoderMaskedMultiheadAttention` | decode-phase MMHA / XQA (per-token attention) | 🔥 hottest |
| `contextFusedMultiHeadAttention` | prefill-phase FMHA | 🔥 |
| `cutlass_kernels` / `internal_cutlass_kernels` | CUTLASS GEMM (incl. quantized / MoE GEMM) | 🔥 |
| `weightOnlyBatchedGemv` | weight-only quantized GEMV (small-batch decode matvec) | high |
| `mixtureOfExperts` | MoE | high |
| `selectiveScan` | Mamba / SSM selective scan | medium |
| `lora` / `beamSearchKernels` / `speculativeDecoding` / `unfusedAttentionKernels` | LoRA / beam / speculative decoding / RoPE·mask parts | medium |

On the plugin side, ~25 of them, mostly one-to-one with kernels (`gptAttentionPlugin` is the core paged-KV attention; `*QuantGemmPlugin` / `quantizePerToken` are the quantization family; `nccl` is multi-GPU comms).

### 2.3 Where to start (by goal, with difficulty)
| Goal | Entry point | Difficulty | First step |
|---|---|---|---|
| learn the mechanism / add a custom op | `plugins/identityPlugin/` (official skeleton) + `cumsumLastDimPlugin` (a simple real example) | ⭐⭐ | copy `identityPlugin`, rename → launch your own kernel in `enqueue` → register in `tllmPlugin.cpp` → expose in `functional.py`, **push one vertical slice end-to-end** |
| see how a whole new op family is added | `kernels/selectiveScan/` ↔ `selectiveScanPlugin` + `mambaConv1dPlugin` + `lruPlugin` (that's how Mamba was added) | ⭐⭐⭐ | read it layer by layer as a template: kernel→plugin→python |
| real performance optimization | `decoderMaskedMultiheadAttention` (MMHA/XQA), `cutlass_kernels` (GEMM) | ⭐⭐⭐⭐⭐ | **not recommended as a starting point** — the heaviest optimization in the repo, CUTLASS the deepest; warm up with the first two |

Companion docs: `docs/source/architecture/add-model.md` (top-level assembly), `docs/source/installation/build-from-source-linux.md` (how to build).

---

## 3. Relationship to this project

- **Operator/engine dev is decoupled from serving**: the outer `text_pipeline_bls` only consumes a **compiled engine**. Change a kernel/plugin in TRT-LLM → rebuild the engine (`scripts/build_engines.sh` calls TRT-LLM) → the serving layer runs the new engine with no changes.
- **This repo can be the end-to-end test bench**: after rebuilding with a new op / new engine, use `client/perf_benchmark.py` to measure TTFT / throughput deltas directly — the gain is immediately visible.
- **Build reality**: building TRT-LLM from source is heavy (CUDA toolkit + CUTLASS, docker build, compilation measured in hours); that lightweight `.venv` setup only runs serving — operator/engine work needs the TRT-LLM build environment.

---

## 4. Next steps (optional)

> **Execution manual:** this doc is "direction / why"; for "how to actually do it on this L4," see [`L2-LAB-EN.md`](./L2-LAB-EN.md) — it turns this repo's knobs (KV reuse / chunked / scheduler / quantization) into an ablate-on-your-own-rig lab, with M0 already closed against real startup logs (hand-calc KV predicts the engine's allocation to 3 sig figs); the lab notebook is in [`lab-notebook/`](./lab-notebook/).

- **Bridge project in phases**: add a vLLM backend → vLLM vs TRT-LLM A/B (reuse `perf_benchmark.py`) → dig into one feature (prefix caching / chunked prefill / FP8) and quantify the gain.
- **vLLM source close-reading route**: `LLMEngine` → scheduler (continuous batching) → block manager (paged KV) → PagedAttention kernel.

> One-line summary: **already standing on the serving layer; sinking half a layer down to the engine layer is AI infra, and the core is inference performance engineering; learn principles from vLLM, deepen on TRT-LLM/Dynamo on NVIDIA home turf; adding a vLLM backend to the existing repo for an A/B is the lowest-effort entry.**
