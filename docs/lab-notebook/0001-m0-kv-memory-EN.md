# 0001 · M0 — KV-cache memory: hand-calc vs the engine's actual allocation

> 🌐 **中文版 (Chinese):** [`0001-m0-kv-memory-CN.md`](./0001-m0-kv-memory-CN.md)

> Lab usage: see [`../L2-LAB-EN.md`](../L2-LAB-EN.md). This entry closes M0 and doubles as a filled-in example of [`TEMPLATE-EN.md`](./TEMPLATE-EN.md).

| Field | Value |
|---|---|
| Date | 2026-06-21 |
| Milestone | M0 |
| Knob | none (establish baseline intuition: how much memory KV takes, and what caps concurrency) |
| Engine/layer | small (0.5B, fraction 0.25) + large (1.5B, fraction 0.45), read from startup logs |

## ① Hypothesis
KV-cache memory = `2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes` × tokens. The hand-calc should predict the engine's actual KV pool.

## ② Predict (before acting)
- **Mechanism:** Qwen2.5 uses GQA, `num_kv_heads=2` ≪ attention heads, so KV is an order of magnitude smaller than if computed from attention heads. FP16 = 2 bytes.
- **Numbers (real arch pulled from each `config.json`):**

| | num_layers | num_kv_heads | head_dim = hidden/heads | predicted B/token |
|---|---|---|---|---|
| 0.5B | 24 | 2 | 896/14 = 64 | `2×24×2×64×2` = **12,288** |
| 1.5B | 28 | 2 | 1536/12 = 128 | `2×28×2×128×2` = **28,672** |

## ③ Setup
```bash
# real arch
for m in 0.5B 1.5B; do grep -iE '"(num_hidden_layers|num_attention_heads|num_key_value_heads|hidden_size)"' \
  hf_models/Qwen2.5-${m}-Instruct/config.json; done
# engine's actual allocation
docker logs triton-llm 2>&1 | grep -iE 'blocks in KV cache|max tokens in paged|maxNumSequences|available:'
```

## ④ Measure (startup logs)
| | KV pool | tokens | blocks × tok/block | measured B/token | vs predicted |
|---|---|---|---|---|---|
| 0.5B | 2.18 GiB | 190,656 | 2979 × 64 | 12,277 | **0.09%** |
| 1.5B | 7.31 GiB | 273,792 | 4278 × 64 | 28,669 | **0.01%** |

- `tokens_per_block = 64` (`273792/4278 = 64.0`, `190656/2979 = 64.0`, both divide evenly).
- fraction check: 1.5B `0.45 × 16.25 GiB avail = 7.31 GiB` ✓; 0.5B `0.25 × 8.73 GiB avail = 2.18 GiB` ✓.
- `maxNumSequences: 64` (= build-time `max_batch_size 64`).

## ⑤ Gap analysis
The hand-calc predicts the KV bytes to **3 significant figures**; the mechanism holds completely. Two unexpected findings:
1. **Two concurrency ceilings.** The 1.5B's KV can hold 273,792 tok → ≈ **133** sequences at 2K context; but `max_batch_size=64` caps concurrency at **64**. **Crossover `273792/64 ≈ 4278 tok`: average context < 4278 → batch_size-bound; > 4278 → KV-memory-bound.**
2. **Load order.** Across the two KV computations, `available` drops from **16.25 → 8.73 GiB** (the first engine eats weights+KV first) → when co-locating models, the second one gets less.

## ⑥ vLLM mechanism
Compared with vLLM: `num_gpu_blocks` is derived from available memory / block bytes, and `max_num_seqs` is a separate, independent ceiling — the same "KV capacity vs max_batch_size, two ceilings" idea. (Backfill when reading `kv_cache_manager.py` + the scheduler in M2/M3.)

## ⑦ Conclusion / next step
**Interview-ready:** "On an L4, Qwen2.5-1.5B in FP16 with GQA (kv_heads=2) → 28 KB/token; fraction 0.45 gives 7.3 GB of KV ≈ 273K tokens; but at short context what actually caps concurrency is the build-time `max_batch_size=64`, not KV — crossover around 4.3K context." That one sentence covers the KV formula, GQA, the memory budget, and "which resource tops out first."
**Next step:** build the measurement spine (L2-LAB §6) → M2: flip `gpt_model_type` inflight→V1, measure the throughput collapse, and verify the batch_size=64 knee.
