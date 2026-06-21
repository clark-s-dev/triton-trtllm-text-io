# 0010 · 实验8:beam search 宽度(吞吐代价)

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md)。消融矩阵 beam 行(B,需重编 + 运行时参数)。

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-21 |
| 里程碑 | M5 周边 |
| 旋钮 | `max_beam_width`:1 → 4;请求 `beam_width`:1 → 4(B + R) |
| 引擎/层 | small(0.5B),beam4 引擎,直连 engine |

## ① 假设
beam=4 = 每步同时维护 4 条候选 → KV ~4×、compute ~4× → 吞吐掉 ~4×;但输出质量可能更好。

## ② 预测(动手前)
吞吐掉 ~4×,ITL 升;TTFT(prefill)基本不变(beam 只影响 decode)。

## ③ 实验设置
**重编** beam4 引擎(reuse ckpt,`--max_beam_width 4`,13 秒)。同一引擎上比 `beam_width=1` vs `4`,C=8 与 C=32。beam 与 chunked context 冲突,故 beam 引擎关掉 chunked/reuse。
> **两处都要设的坑:** 光 build 时 `--max_beam_width 4` 不够——运行时 `config.pbtxt` 里还有个 `max_beam_width` **参数**默认是 `"1"`,会把 beam 卡死在 1(`Requested beam width 4 is larger than configured max beam width 1`)。**build 旗标 + 运行时参数,两处都改成 4。**

## ④ 实测
| 并发 | beam=1 | beam=4 | req/s 代价 | ITL p50 (1→4) |
|---|---|---|---|---|
| C=8  | 10.1 req/s | 7.7 req/s | **−24%**(1.3×) | 5.5 → 7.8 ms |
| C=32 | 35.3 req/s | 15.1 req/s | **−57%**(2.3×) | 6.3 → 15.8 ms |

```
吞吐代价随负载放大:
C=8   (GPU 有余量)  beam4 ███████░░░ 比 beam1 慢 1.3×   ← 余量吸收了多出的 beam 计算
C=32  (GPU 吃紧)    beam4 ████░░░░░░ 比 beam1 慢 2.3×   ← 趋向理论 4×

ITL p50:  C=8: 5.5→7.8ms (+43%)   C=32: 6.3→15.8ms (+150%)
TTFT:     基本不变(beam 只影响 decode,不影响 prefill)
```

## ⑤ Gap 分析
预测的"4×"只在 GPU 吃满时才接近:
- **代价随负载放大**:C=8 时 GPU 有空闲算力,beam=4 的 4× 计算被余量吸收 → 只慢 1.3×;C=32 时 GPU 接近饱和 → 慢 2.3×,趋向 4×。**"beam=N 慢 N×"只在算力饱和时成立**;低负载下余量会掩盖代价(又一次小模型/有余量陷阱)。
- **质量没测**(诚实):beam 的卖点是探索多条路径选最优、输出质量更好——但本台用随机 token,**测不了质量**。要用真实任务(翻译 BLEU / 推理准确率)才能看到 beam 的收益面。

## ⑥ 结论(大白话)
- beam=4 = 同时押 4 个候选答案、最后选最好的 → **decode 贵 ~4×(KV 和算力都 ×4)**。延迟代价在高负载下才完全显现(C=32 慢 2.3×、ITL 2.5×)。
- 生产里大多数在线服务**用 sampling 不用 beam**:beam 贵,且对开放式生成(聊天)收益有限;它更适合翻译/受限生成这类"要全局最优"的任务。
- 工程坑:beam 要 **build 旗标 + 运行时参数两处都设**。

**下一步(最后一个 case):** [实验9 — INT4-AWQ 权重量化(decode 提速)](./0011-int4-awq.md)。
