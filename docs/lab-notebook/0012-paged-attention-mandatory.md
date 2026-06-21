# 0012 · 实验10:关掉 PagedAttention?(§2 旋钮图最后一项)

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md)。这是 §2 旋钮图里唯一没作为独立实验跑的旋钮:**关掉分页 KV**。结论:**它根本不是个"可调旋钮"——是连续批处理的地基,关不掉。**

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-21 |
| 里程碑 | M3 |
| 旋钮 | `paged_kv_cache` / `--kv_cache_type`:paged → **continuous**(B) |
| 引擎/层 | small(0.5B),直连 engine |

## ① 假设
关掉分页 KV(改连续/contiguous KV)→ 看 PagedAttention 到底省了什么。预期:连续 KV 每条序列**预留 max_seq_len** → 浪费巨大 → 能同时跑的请求骤减、吞吐崩。

## ② 实测(三连击,一步比一步说明问题)
1. **连续 KV 引擎能 build**(`trtllm-build --kv_cache_type continuous`,13 秒 → `CONTIG_BUILT`)。
2. **但 inflight 批处理直接拒绝加载它**:
   ```
   TrtGptModelInflightBatching requires GPT attention/Mamba Conv 1d plugin
   with packed input and paged KV cache.
   ```
   → **连续批处理 = 必须分页 KV。两者绑死,关不掉分页就用不了 inflight。**
3. **只有退回 V1 静态批(+ guaranteed_no_evict)才肯加载连续 KV**。但 V1+连续 KV 跑短请求(in128/out128,C=64)→ **0/128 完成,全超时**(wall 243s)。退化到没法服务。

**为什么 V1+连续 跑不动:**
```
连续 KV 池 = 2648 块 = 169K token
连续模式每条序列预留 max_seq_len = 8192 token
→ 能塞下的序列 ≈ 169K / 8192 ≈ 20 条  (不管请求多短!)
→ C=64 根本放不下;V1 静态批又没法连续补位 → 全卡死超时
```
对比分页(实验 [0007](./0007-kv-mem-fraction.md)):分页**按实际长度**分块,128-token 的请求只占 2 块 → 同样的池子能塞几百条。

## ③ Gap 分析
预测"浪费巨大、吞吐崩"是对的,但**比预想更极端**:不是"慢一点",而是**这个配置在现代 serving 栈里根本不成立**——
- inflight 执行器**强制要求**分页 KV(报错明说),所以"关掉分页"≠ 调个参数,而是 **= 放弃连续批处理**,直接吃 [0003](./0003-continuous-vs-static-batching.md) 的 2.1× 吞吐 / 195× 首字代价。
- 即便退到 V1+连续,连续 KV 的 max_seq_len 预留让它**连 C=64 短请求都服务不了**(0/128 超时)。

## ④ 结论(大白话)
- **PagedAttention 不是一个"开/关"旋钮,而是连续批处理的地基。** 现代执行器(TRT-LLM inflight / vLLM)从架构上就假设分页 KV——你**关不掉它还想要连续批处理**。
- "关掉分页"的真实含义 = 回到 V1 静态批 + 连续 KV:① 吃 0003 的静态批大坑;② 连续 KV 每条预留 max_seq_len,池子瞬间被少数长预留占满(20 条封顶),短请求也塞不进 → 实测直接 0/128 超时。
- 这就是**为什么整套现代推理栈默认且只用分页 KV**:它让"按实际长度打包变长请求"成为可能,而连续批处理完全建立在这之上。**省的不是一点显存,是让连续批处理这件事能成立。**

## ⑤ 收尾
至此 L2-LAB §3 消融矩阵(8 行)+ §2 旋钮图(含本条"关不掉的分页")全部跑完。`max_num_tokens` 与 fraction 同机制(已并入 [0007](./0007-kv-mem-fraction.md))。**整个矩阵闭合。**
