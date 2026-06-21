# 0011 · 实验9:INT4-AWQ 权重量化(decode 提速)— 最后一个 case

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md)。消融矩阵 dtype 行的 INT4 变体(B,需校准)。**消融矩阵全部跑完。**

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-21 |
| 里程碑 | M4(量化) |
| 旋钮 | dtype:FP16 → **INT4-AWQ**(W4A16,权重 4-bit,group 128)(B) |
| 引擎/层 | small(0.5B),直连 engine |

## ① 假设
weight-only INT4 主要加速 **decode**(小批 decode 带宽受限、权重流量占大头);**高批(compute-bound)收益消失甚至变慢**(要把 4-bit 反量化回 FP16 算)。

## ② 预测(动手前)
INT4 在 C=1 最快;C=32 收益消失或更慢。

## ③ 实验设置
**重编 + AWQ 校准**:`quantize.py --qformat int4_awq --awq_block_size 128`(modelopt 激活感知校准,32 样本)→ `trtllm-build`。
> **build 坑:** int4_awq 的 checkpoint 默认带 `kv_cache_quant_algo=FP8`,与 `--use_paged_context_fmha` 冲突(`FP8 Paged Context FMHA only works with fp8 quantization workflow`)。修复:int4 引擎 build 时**去掉** `--use_paged_context_fmha`(用 `--kv_cache_type paged`),运行时也关掉 chunked/reuse(它们依赖 paged-context-fmha)。

## ④ 实测(三种精度对照,tok/s)
| 并发 C | FP16 | FP8 | **INT4-AWQ** | INT4 ITL p50 |
|---|---|---|---|---|
| 1   | 197 | 218 | **290**(+47% vs FP16) | 3.07 ms(FP16 5.0) |
| 8   | 1239 | — | **1765**(+42%) | 3.76 ms |
| 32  | 3596 | 4950 | **5394**(+50% vs FP16,+9% vs FP8) | 4.33 ms(FP16 6.4) |

```
吞吐(tok/s),三种精度:
        C=1          C=8           C=32
FP16    ██ 197       ████ 1239     ████████████ 3596
FP8     ██ 218       —             █████████████████ 4950
INT4    ███ 290      ██████ 1765   ██████████████████ 5394   ← 全程最快

引擎体积:FP16 1215 MB → FP8 875 MB → INT4 709 MB(最小)
decode 每 token(ITL p50):FP16 5.0–6.4ms → INT4 3.07–4.33ms（快 ~35%）
```

## ⑤ Gap 分析(预测错得有价值)
**预测「INT4 高批收益消失」是错的** —— INT4 在 C=32 仍最快(+50% vs FP16)。原因:
- **0.5B 太小,即便 C=32,decode 仍是 weight-bandwidth-bound**:权重读取(~1GB FP16)主导,batch 32 的激活计算量还很小,没到 compute-bound。所以「砍权重字节到 1/4」一路都提速,不像大模型那样高批转 compute-bound 后收益消失。
- TRT-LLM 的 **weight-only GEMV kernel**(算子图里的 `weightOnlyBatchedGemv`,见 AI-INFRA-DIRECTION §2.2)对小 batch 高度优化。
- **精度没测(最大保留):** INT4(4-bit 权重)比 FP8 激进得多,精度风险更大;AWQ(激活感知)能缓解但不消除。本台随机 token **测不了精度**,必须用真实 eval(perplexity/任务)。**INT4 的质量风险 > FP8**,这是用它之前必须验的。

## ⑥ vLLM / roofline 机制
decode 在小 batch 是 **memory-bound**(roofline 的带宽墙):每个 token 都要把全部权重从显存读一遍,算得很少。量化把权重字节砍掉 → 直接撞松带宽墙 → decode 提速。这是 M0「为什么 decode 带宽受限」的直接验证。

## ⑦ 结论(大白话)
- 在这个 0.5B 上,**INT4-AWQ 是吞吐冠军**:+42~50% vs FP16,C=32 还超 FP8;引擎最小(709MB);每 token decode 快 ~35%。
- 因为小模型 decode 一直卡在「读权重」这道带宽墙上,**把权重砍到 4-bit 一路都管用**(大模型高批会转 compute-bound,收益才衰减)。
- **代价是精度**:4-bit 比 FP8 激进,务必用真实数据集验质量——性能台量不了这个。
- 一句话:**decode 受带宽限 → 量化砍权重字节 = 直接提速**(roofline 实锤);选 FP8 还是 INT4 是「质量 vs 速度」的取舍,FP8 更稳、INT4 更快更小但风险更高。
