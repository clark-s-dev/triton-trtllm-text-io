# Decode Acceleration on a Single L4 — Quantization, Speculative Decoding & the Bandwidth Wall (M4)

*Measured on one NVIDIA L4 (NGC `tritonserver:24.10-trtllm`, TensorRT-LLM 0.14.0), Qwen2.5-0.5B / 1.5B.*
*Companion docs: the architecture/status report is [`REPORT.md`](./REPORT.md); the L2 lab manual is*
*[`L2-LAB.md`](./L2-LAB.md); the full predict-then-measure write-ups are lab notebooks*
*[0009](./lab-notebook/0009-fp8-quantization.md)/[0011](./lab-notebook/0011-int4-awq.md) (quantization),*
*[0016](./lab-notebook/0016-speculative-decoding.md) (speculative decoding),*
*[0017](./lab-notebook/0017-nsight-decode-bandwidth.md) (kernel-level bandwidth proof).*

---

## 1. The one idea

Token generation ("decode") on a small model is **limited by memory bandwidth, not compute**. To make
one token, the GPU reads *every weight* out of memory and does almost no arithmetic with it. On the L4,
the decode kernels run at **~83% of peak memory bandwidth but only ~0.2% of peak compute** (§2).

Three different techniques speed decode up, and all three are just different ways to deal with that one
wall:

| Lever | What it does to the wall | Lossy? | Notebook |
|---|---|---|---|
| **Quantization** (FP8 / INT4) | reads **fewer bytes** per token (smaller weights) | yes (accuracy) | [0009](./lab-notebook/0009-fp8-quantization.md) / [0011](./lab-notebook/0011-int4-awq.md) |
| **Speculative decoding** | produces **more tokens** per weight-read (verify K at once) | **no** | [0016](./lab-notebook/0016-speculative-decoding.md) |
| **Continuous batching** | shares **one weight-read across B requests** | no | [0003](./lab-notebook/0003-continuous-vs-static-batching.md) / [0008](./lab-notebook/0008-max-batch-size.md) |

This report covers the first two and shows how the third one quietly competes with speculative decoding.

---

## 2. The bandwidth wall, proven at the kernel level (notebook 0017)

**Why decode is bandwidth-bound.** At batch=1 each decode step multiplies a 1-row activation by every
weight matrix. That's ~1 FLOP per weight byte read — the *arithmetic intensity* is ~1, while the L4's
"roofline ridge" (where compute and bandwidth balance) is ~400 FLOP/byte. We're **400× into the
memory-bound regime**.

**The measurement.** Nsight Compute can't run here (the CUDA-13 driver is newer than the profiler in the
24.10 image — see §6), so we used `nsys` kernel timing + exact weight-byte counts. For the 0.5B engine at
batch=1:

| Decode kernel | Bytes read / call | Achieved bandwidth | % of L4 peak (300 GB/s) |
|---|---|---|---|
| `lm_head` GEMV (FP16) | 272 MB | 246 GB/s | **82%** |
| transformer GEMMs (FP16, per token) | 716 MB | 253 GB/s | **84%** |
| *whole token (incl. attention/norms)* | 988 MB | 200 GB/s | 67% |

Compute utilization over the same kernels: **~0.2% of peak**. The kernels are *saturating memory
bandwidth and leaving the math units idle* — that is the definition of bandwidth-bound, now measured
rather than argued.

**One surprise that matters:** the final `lm_head` projection is **left in FP16** (the quantizer skips it
for quality). Its kernel takes the *exact same time* in the FP16, FP8, and INT4 engines — a fixed
bandwidth floor. That single fact explains why quantization gives sub-linear speedups (next section).

**Cross-dtype comparison** (the same decode step, re-compiled in each precision):

| precision | dominant linear kernel | GEMM time-share | `lm_head` (FP16) share | decode ITL | achieved BW | % peak BW | % peak compute |
|---|---|---|---|---|---|---|---|
| FP16 | `cudaCoreGemm` | 60.5% | 23.6% | 4.94 ms | 200 GB/s | **67%** | 0.17% |
| FP8 (W8A8) | `sm89_xmma_gemm_e4m3` | 50.3% | 29.9% | 3.96 ms | 159 GB/s | **53%** | 0.21% |
| INT4-AWQ (W4A16) | `weight_only` GEMV | 34.4% | 40.0% | 3.04 ms | 148 GB/s | **49%** | 0.27% |

Reading across: quantization swaps in a smaller-byte GEMM kernel (`cudaCoreGemm` → FP8 `sm89_xmma` → INT4
`weight_only`) whose share *shrinks* (60.5% → 34.4%), while the un-quantized FP16 `lm_head` share *grows*
(23.6% → 40.0%) — the fixed floor. Every row keeps **bandwidth ≫ compute**: still bandwidth-bound.

![Nsight decode-kernel analysis — bandwidth-bound, and the lm_head floor](./decode-roofline.png)

*Left: decode weight kernels run at 49–67% of peak memory bandwidth but only ~0.2% of peak compute —
bandwidth-bound. Right: as quantization shrinks the transformer GEMM, the un-quantized FP16 `lm_head`
grows from 24% to 40% of decode time — why quantization speedup is sub-linear. (Source: `nsys` kernel
timing + weight-byte accounting; `ncu` hardware counters are blocked on this box — see §7.)*

---

## 3. Lever 1 — Quantization: read fewer bytes (notebooks 0009 / 0011)

Re-compile the engine with smaller weights. Same model, fewer bytes to stream → faster decode + smaller
engine. **Batch=1 decode, 0.5B:**

| Precision | Decode speed (ITL) | vs FP16 | Engine size |
|---|---|---|---|
| FP16 | 4.94 ms/token | — | 1216 MB |
| **FP8** (W8A8) | 3.96 ms/token | **1.25× faster** | 876 MB |
| **INT4-AWQ** (W4A16) | 3.04 ms/token | **1.6× faster** | 710 MB |

**The honest caveat:** speedup is **sub-linear** — INT4 cuts transformer-weight bytes by 4× but is only
1.6× faster, because the un-quantized FP16 `lm_head` (§2) and attention don't shrink. And quantization
**trades accuracy** — this perf rig can't measure quality; FP8 is usually near-lossless, INT4 is riskier
and needs a real eval before shipping.

---

## 4. Lever 2 — Speculative decoding: more tokens per weight-read (notebook 0016)

A small **draft** model (0.5B) proposes K tokens; the big **target** model (1.5B) **verifies all K in a
single forward pass** and keeps the longest correct prefix. This is a win *because* decode is
bandwidth-bound: the target reads its weights once whether it scores 1 token or K+1, so verifying a batch
of guesses is nearly free — *if the guesses are good*.

It is **lossless**: the output is exactly what the target model would have produced alone (we confirmed it
— the output is identical regardless of K).

**Result 1 — how many tokens to guess (K).** Greedy, mixed prompts, vs the plain 1.5B (74.7 tok/s):

| draft_len K | acceptance | speedup |
|---|---|---|
| 1 | 0.80 | 1.09× |
| **2** | 0.72 | **1.18× ← best** |
| 4 | 0.59 | 1.14× |
| 6 | 0.47 | 1.01× |
| 8 | 0.39 | **0.90× (slower!)** |

Guess too few and you barely win; guess too many and the draft's wasted work makes you *slower* than not
speculating. The sweet spot here is **K=2**.

**Result 2 — it depends entirely on the workload.** Same models, same K=4, just different prompts:

| Workload | acceptance | speedup |
|---|---|---|
| **Predictable** (code, lists, tables) | 0.86 | **1.50×** |
| Mixed | 0.59 | 1.14× |
| **Creative** (open-ended prose) | 0.46 | **0.97× (slower)** |

**Acceptance rate is a property of the workload, not the model.** The draft nails predictable text and
flails on creative text. This is the single most important practical fact about speculative decoding.

**Result 3 — sampling needs the right acceptance rule.** With randomness (temperature 0.8):

| Acceptance rule | acceptance | speedup |
|---|---|---|
| logits (rejection sampling) | 0.85 | **1.47×** |
| token-equality | 0.33 | **0.68× (much slower)** |

If you sample, you **must** accept by comparing probabilities (logits), not by checking if the two models
picked the same token — two independent samples rarely match, so naïve acceptance collapses.

**Result 4 — it fights continuous batching.** K=4, greedy, as you serve more requests at once:

| Concurrency | baseline tok/s | spec-decode tok/s | speedup |
|---|---|---|---|
| 1 | 74.7 | 85.1 | **1.14×** |
| 2 | 127.2 | 108.7 | 0.86× |
| 4 | 209.2 | 133.8 | 0.64× |
| 8 | 309.4 | 161.6 | **0.52×** |

Speculative decoding **only wins at batch=1**. The moment you batch, normal serving shares one weight-read
across many requests (that's what continuous batching *is*), which is a bigger win than speculation — and
speculation's extra verification work now costs real time. They compete for the same idle compute.

---

## 5. So what should I actually use?

| If you are… | Use | Why |
|---|---|---|
| **Latency-sensitive, low traffic** (interactive chat, batch≈1) | **Speculative decoding** (K≈2, logits acceptance) + quantization | Cuts single-stream latency, lossless. Best on predictable/structured output. |
| **Throughput-bound, high traffic** (batch serving) | **Quantization + continuous batching** — *not* speculative decoding | Batching already exploits the bandwidth wall; speculation only slows you down past batch≈2. |
| **Quality-critical** | **FP8** before INT4; verify on a real eval; speculative decoding is free (lossless) | INT4 is riskier; speculation changes nothing about output quality. |

**The headline:** quantization buys throughput at some accuracy risk; speculative decoding buys
single-stream latency for free but evaporates under load. They are not interchangeable.

---

## 6. How to reproduce

```bash
# Quantized engines + the FP8/INT4 sweeps live in notebooks 0009/0011 (scripts/build_engines.sh path).

# Speculative decoding (notebook 0016):
make specdec-engines      # build draft(0.5B) + target(1.5B) engines  (scripts/build_specdec_engines.sh)
make specdec              # run the full sweep -> lab/results_0016.jsonl  (lab/run_0016_sweep.sh)

# Kernel-level bandwidth proof (notebook 0017):
make profile-decode       # nsys decode-kernel timing across FP16/FP8/INT4 -> lab/ncu/*.kern.txt

# The speedup model, calibrated to the sweep, runs GPU-free in the test suite:
make test                 # includes tests/test_specdec_model.py  (28 GPU-free tests)
python3 lab/specdec_model.py    # prints modeled vs measured speedup per K
```

All experiments run by stopping the live `triton-llm` server to free the GPU, then restoring it — the
serving stack and these lab engines are decoupled.

---

## 7. Limitations & honest caveats

- **Quantization quality is unmeasured.** This is a *performance* rig (random/synthetic decode); FP8/INT4
  accuracy must be checked on a real perplexity/task eval before production. INT4 carries more risk.
- **Nsight Compute is blocked on this box.** The L4's CUDA-13 / 580.x driver is newer than the
  `ncu`/`nsys` 2024.2.1 shipped in the NGC 24.10 image, so the hardware performance counters
  (`NVPA_STATUS_ERROR`) won't initialize — `--privileged` and `--clock-control none` don't help; it needs
  Nsight 2025.x. We got equivalent (per-kernel) evidence from `nsys` kernel timing + byte accounting.
- **The speculative-decoding numbers include orchestration overhead.** The measured per-draft-step cost
  (≈0.42× a target step, vs ~0.32× from pure weight ratio) includes Python loop / sync / logit-transfer
  cost; a production C++ orchestrator would be tighter and could shift the optimal K slightly higher.
- **Small models, single L4.** The 0.5B→1.5B pair and batch=1 economics are specific to this rig; bigger
  models / different draft-target ratios move every number (but not the qualitative story).
