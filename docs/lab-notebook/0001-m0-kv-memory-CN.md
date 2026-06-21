# 0001 · M0 — KV-cache 显存手算 vs 引擎实际分配

> 🌐 **English version:** [`0001-m0-kv-memory-EN.md`](./0001-m0-kv-memory-EN.md)

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB-CN.md)。这条是 M0 的闭环,也是 [`TEMPLATE.md`](./TEMPLATE-CN.md) 的一个填好的范例。

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-21 |
| 里程碑 | M0 |
| 旋钮 | 无(建立基线认知:KV 占多少显存、并发被谁卡住) |
| 引擎/层 | small(0.5B,fraction 0.25)+ large(1.5B,fraction 0.45),读启动日志 |

## ① 假设
KV-cache 显存 = `2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes` × tokens。手算应当能预测引擎实际分配的 KV pool。

## ② 预测(动手前)
- **机理:** Qwen2.5 是 GQA,`num_kv_heads=2` ≪ attention heads,所以 KV 比按 attention heads 算小一个量级。FP16 = 2 bytes。
- **数字(从各自 `config.json` 取真实架构):**

| | num_layers | num_kv_heads | head_dim = hidden/heads | 预测 B/token |
|---|---|---|---|---|
| 0.5B | 24 | 2 | 896/14 = 64 | `2×24×2×64×2` = **12,288** |
| 1.5B | 28 | 2 | 1536/12 = 128 | `2×28×2×128×2` = **28,672** |

## ③ 实验设置
```bash
# 真实架构
for m in 0.5B 1.5B; do grep -iE '"(num_hidden_layers|num_attention_heads|num_key_value_heads|hidden_size)"' \
  hf_models/Qwen2.5-${m}-Instruct/config.json; done
# 引擎实际分配
docker logs triton-llm 2>&1 | grep -iE 'blocks in KV cache|max tokens in paged|maxNumSequences|available:'
```

## ④ 实测(启动日志)
| | KV pool | tokens | blocks × tok/block | 实测 B/token | vs 预测 |
|---|---|---|---|---|---|
| 0.5B | 2.18 GiB | 190,656 | 2979 × 64 | 12,277 | **0.09%** |
| 1.5B | 7.31 GiB | 273,792 | 4278 × 64 | 28,669 | **0.01%** |

- `tokens_per_block = 64`(`273792/4278 = 64.0`,`190656/2979 = 64.0` 都整除)。
- fraction 核对:1.5B `0.45 × 16.25 GiB avail = 7.31 GiB` ✓;0.5B `0.25 × 8.73 GiB avail = 2.18 GiB` ✓。
- `maxNumSequences: 64`(= build 时 `max_batch_size 64`)。

## ⑤ Gap 分析
手算预测 KV 字节到 **3 位有效数字**,机理完全成立。两个意外收获:
1. **两个并发天花板。** 1.5B 的 KV 能放 273,792 tok → 2K context 时 ≈ **133** 条;但 `max_batch_size=64` 把并发卡在 **64**。**交叉点 `273792/64 ≈ 4278 tok`:平均 context < 4278 → batch_size 绑定;> 4278 → KV 内存绑定。**
2. **加载顺序。** 两次 KV 计算时 available 从 **16.25 → 8.73 GiB**(第一个引擎先吃权重+KV)→ 共置模型时第二个拿得少。

## ⑥ vLLM 机制
对照 vLLM:`num_gpu_blocks` 由可用显存 / block 字节算出,`max_num_seqs` 是另一条独立上限——和这里"KV 容量 vs max_batch_size 两个天花板"是同一回事。(M2/M3 读 `kv_cache_manager.py` + scheduler 时回填。)

## ⑦ 结论 / 下一步
**能讲给面试官:** "L4 上 Qwen2.5-1.5B,FP16 + GQA(kv_heads=2)→ 28 KB/token;fraction 0.45 给 7.3 GB KV ≈ 273K token;但短 context 下真正卡并发的是 build 时的 `max_batch_size=64`,不是 KV——交叉点约 4.3K context。" 这一句同时覆盖了 KV 公式、GQA、显存预算、和"哪个资源先到顶"。
**下一步:** 搭测量脊梁(L2-LAB §6)→ M2 把 `gpt_model_type` inflight→V1,量吞吐崩塌,验证 batch_size=64 拐点。
