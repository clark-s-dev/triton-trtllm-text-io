# `lab/` — L2 toy artifacts (M2 + M3), calibrated to this rig

The two **toy artifacts** promised in [`docs/L2-LAB.md`](../docs/L2-LAB.md) §7. Both are
pure-Python, GPU-free, dependency-free, and — the whole point — **calibrated against
numbers measured on the real engine** in [`docs/lab-notebook/`](../docs/lab-notebook/),
not written in a vacuum. They compose: the simulator (M2) uses the allocator (M3) for KV
accounting.

| File | Milestone | Models | Calibration target (engine) |
|---|---|---|---|
| [`paged_kv.py`](./paged_kv.py) | **M3** | paged KV block table · prefix-hash reuse · refcount · LRU evict | `reused = 2418` blocks (notebook [0004](../docs/lab-notebook/0004-kv-cache-reuse.md)) |
| [`cbatch_sim.py`](./cbatch_sim.py) | **M2** | iteration-level continuous/static scheduler · admit/evict · prefill/decode mix | throughput-vs-concurrency **knee** at `max_batch_size` (notebook [0008](../docs/lab-notebook/0008-max-batch-size.md)) |

## Run them

```bash
python3 lab/paged_kv.py            # reproduces engine reused=2418 to the block
python3 lab/cbatch_sim.py          # prints the max_batch_size knee table + continuous-vs-static
make test                          # the calibration asserts run as part of the GPU-free suite
#   ( tests/test_paged_kv.py · tests/test_cbatch_sim.py )
```

No install needed — stdlib only, Python 3.10+.

## What they reproduce (and the gaps that are the lesson)

The full predict-vs-measure write-ups are the lab notebooks:
[**0013**](../docs/lab-notebook/0013-toy-continuous-batching-sim.md) (M2 simulator) and
[**0014**](../docs/lab-notebook/0014-toy-paged-kv-allocator.md) (M3 allocator). In short:

- **M3 allocator** reproduces `reused=2418` *exactly* (= `(40−1) × 62` blocks: one cold
  request paving a 3968-token / 62-block shared prefix, then 39 warm hits). The `2418 =
  39 × 62` arithmetic even back-solves the request count the notebook didn't record.
- **M2 simulator** reproduces the **knee** (throughput rises with concurrency, plateaus at
  `max_batch_size`), the bs16 plateau (~2169 tok/s vs measured 2202), the continuous-vs-
  static head-of-line gap (avg occupancy 30 vs 17, matching `sum(len)/max(len)`), and the
  M0 KV-vs-batch-size crossover (~2979-token context for the 0.5B pool).
- **Documented gaps** (§7's "don't force the fit"): the linear iteration-cost model is
  ~14% optimistic at `B=128` (it doesn't model compute-bound saturation — the roofline
  compute ceiling); the continuous/static ratio comes out 1.5× vs the measured 2.1×
  (sublinear cost compresses it, plus the real V1's non-streaming + forced no-evict); and
  `max_utilization` recompute-storms under *severe* overcommit, revealing that its
  advantage over `no_evict` is regime-dependent — a lesson beyond what notebook 0006 saw.

## The cost model (where the magic numbers come from)

`cbatch_sim.py` uses one decode-iteration cost line, fitted to this rig's notebook 0008:

```
T(B) = T_WEIGHT + B · T_TOKEN          # ms;  T_WEIGHT=6.1, T_TOKEN=0.071
```

`T_WEIGHT` (one weight load, amortized over the whole batch) and `T_TOKEN` (marginal
per-token compute) are back-solved from the measured `B=16 → 2202 tok/s` and
`B=64 → 6086 tok/s`. Prefill cost `T_PREFILL_TOKEN=0.0065` is anchored to the `C=1,
in=128, TTFT=7 ms` point in notebook 0003. Change the engine and you re-fit two numbers.

---

## M4 measurement harnesses — **need the GPU** (not the GPU-free toys above)

These two run *inside* the NGC trtllm container (they import `tensorrt_llm`), so they are
launched via `docker run` (see the `Makefile`), unlike the stdlib-only toys.

| File | Notebook | What it measures |
|---|---|---|
| [`specdec_bench.py`](./specdec_bench.py) + [`run_0016_sweep.sh`](./run_0016_sweep.sh) | [**0016**](../docs/lab-notebook/0016-speculative-decoding.md) | Draft-Target speculative decoding: per-step **acceptance rate**, mean accepted/iter, throughput, and **speedup** vs a plain-FP16 target baseline. Sweeps `draft_len K`, logits-vs-token acceptance, easy-vs-hard prompts, and the **batch crossover** (where spec-decode stops beating continuous batching). |
| [`decode_probe.py`](./decode_probe.py) | [**0017**](../docs/lab-notebook/0017-nsight-decode-bandwidth.md) | A minimal batch=1 decode workload to put under `ncu` (`scripts/profile_decode.sh`): kernel-level **Speed-of-Light** (Memory% ≫ Compute%) proving decode is DRAM-bandwidth-bound, across FP16/FP8/INT4. |

```bash
make specdec-engines     # build draft(0.5B) + target(1.5B) engines  (scripts/build_specdec_engines.sh)
make specdec             # run the 0016 sweep -> lab/results_0016.jsonl
make profile-decode      # run the 0017 Nsight decode-kernel profiling -> lab/ncu/*.csv
```

The spec-decode loop is faithful to TensorRT-LLM v0.14 `examples/run.py:run_draft_target_model()`
but cleanly instrumented; the baseline runs on the **plain** `qwen2.5-1.5b-fp16` engine because a
`draft_tokens_external` target emits only one token per call without drafts.
