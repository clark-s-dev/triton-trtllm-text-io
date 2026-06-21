# 0009 · 实验7:FP8 量化(精度 vs 吞吐)— M4 主场

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md)。消融矩阵 dtype 行(B,需重编 + 校准)。L4 是 Ada(sm_89),**有 FP8 张量核**,所以能跑(FP4 是 Blackwell-only,跑不了)。

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-21 |
| 里程碑 | M4(量化) |
| 旋钮 | dtype:FP16 → **FP8**(W8A8 + FP8 KV cache)(B) |
| 引擎/层 | small(0.5B),直连 engine |

## ① 假设
FP8 用 Ada 的 FP8 张量核加速 GEMM → prefill 和 decode 都更快;FP8 KV → KV 容量翻倍。代价:精度可能下降。

## ② 预测(动手前)
吞吐升、ITL 降;compute-bound(高并发/prefill)增益最大;精度略降(需单独验)。

## ③ 实验设置
**重编 + 校准**:`quantize.py --qformat fp8 --kv_cache_dtype fp8`(modelopt PTQ,32 条校准样本)→ `trtllm-build`。
> **build 坑(RCA 级):** NGC 24.10 镜像里 modelopt 0.17 因 **缺 `setuptools`** 而 import 失败(`ModuleNotFoundError: setuptools`);校准还需 `datasets`。修复:校准前在容器里 `pip install setuptools datasets`。

## ④ 实测(FP8 vs FP16 基线,变长 out 16–256)
| 并发 C | FP8 吞吐 | FP16 吞吐 | 加速 | FP8 ITL p50 | FP16 ITL p50 |
|---|---|---|---|---|---|
| 1   | 218 tok/s | 197 tok/s | **+10%** | 4.0 ms | 5.0 ms |
| 32  | **4950 tok/s** | 3596 tok/s | **+38%** | 4.7 ms | 6.4 ms |
| 64  | 6805 tok/s | 6086 tok/s | +12% | 5.3 ms | — |
| 128 | 7350 tok/s | 6630 tok/s | +11% | 10.1 ms | — |

```
吞吐加速比(FP8/FP16):
C=1    ▏+10%    单流 decode:带宽受限,FP8 权重小一点 → 略快
C=32   ████████ +38%   ← 最大增益:compute-bound,FP8 张量核发力
C=64   ██ +12%
C=128  ██ +11%   极高并发被其他瓶颈(KV带宽/调度)拉平
```
**附带好处:**
- 引擎体积:FP8 **917 MB** vs FP16 1274 MB(**−28%**)。
- 每 token decode 快 ~20–27%(ITL 4.0–4.7 vs 5.0–6.4 ms)。
- `kv_cache_quant_algo=FP8` → KV 每 token 字节减半 → **KV 容量 ~2×**(能塞更多并发,本实验没专门压这点)。

## ⑤ Gap 分析
预测方向对,细节有惊喜:
- **增益不是单调随并发的**:C=32 最大(+38%),到 C=128 收窄到 +11%。中等并发时 GEMM 是瓶颈,FP8 张量核直接砍一半计算;但极高并发时瓶颈转移(KV 带宽 / 调度开销),FP8 帮不上 → 增益收窄。
- **精度没测**(诚实):本实验用随机 token 只量性能,**测不了精度**。FP8 通常接近无损(业界常报 <1% 退化),但**必须用真实 eval(perplexity / 任务准确率)单独验**才能下结论。这是本条的最大局限。

## ⑥ 结论(大白话)
- FP8 在 L4 上是**真香**:+11~38% 吞吐(中等并发最划算)、decode 每字快 ~20%、引擎小 28%、KV 容量翻倍。几乎全是 runtime 收益,只在 build 期多一步校准。
- **但**:精度要另外用真实数据集验证——性能台量不了这个。
- FP4 想都别想(Blackwell-only,L4 跑不了)。

**下一步:** [实验8 — beam search 宽度(吞吐代价)](./0010-beam-width.md);之后 INT4-AWQ。
