# NNNN · <一句话标题:在测哪个旋钮 / 哪个问题>

> 🌐 **English version:** [`TEMPLATE-EN.md`](./TEMPLATE-EN.md)

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB-CN.md)。**先填①②(动手前),再填④⑤。** ①②和④⑤之间的 gap 才是学习信号。

| 字段 | 内容 |
|---|---|
| 日期 | YYYY-MM-DD |
| 里程碑 | M? |
| 旋钮 | `<param>`:`<from>` → `<to>`(R/B) |
| 引擎/层 | small(0.5B) / large(1.5B);直连 engine / 走 BLS |

## ① 假设(Hypothesis)
<一句话:关掉/改这个旋钮,我预期什么变,往哪个方向变。>

## ② 预测(Predict — 动手前写死)
- **数字:** <预测的 TTFT / ITL / throughput / 并发 / 显存,带量级估算>
- **机理:** <为什么是这个方向、这个量级。引 roofline / KV 公式 / 调度行为。>

## ③ 实验设置(可复现)
- **Workload:** 并发 C=?,输入 len 分布?,输出 len?,共享前缀比例?(为暴露这个旋钮专门设计 —— 见 L2-LAB §4.2)
- **命令:**
  ```bash
  # 改参数 → restart → 等 ready → 测
  ```
- **控制变量:** warmup ?;N=?;是否 docker stop triton-fused;其他旋钮保持默认值。

## ④ 实测(Measure)
| 指标 | baseline(旋钮 ON) | ablation(旋钮 OFF/改) | delta |
|---|---|---|---|
| TTFT P50 / P99 |  |  |  |
| ITL P50 / P99 |  |  |  |
| throughput (tok/s) |  |  |  |
| 最大并发 / 显存 |  |  |  |
| 旁证(DCGM SM% / 功耗、KV 指标) |  |  |  |

## ⑤ Gap 分析(预测 vs 实测)
<对上了吗?没对上差在哪?是我机理错了,还是真实系统做了我没建模的事(chunked / recompute / 碎片 / 网关串行)?>

## ⑥ vLLM 机制(解释层)
<读了哪个文件,它怎么实现这个行为,和我量到的 delta 对得上吗?> `vllm/v1/core/...`

## ⑦ 结论 / 下一步
<一句话能讲给面试官的结论 + 这条引出的下一个实验。>
