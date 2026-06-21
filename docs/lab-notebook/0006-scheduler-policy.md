# 0006 · 实验4:调度策略 max_utilization vs guaranteed_no_evict(KV 压力下)

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md)。消融矩阵第四行(M2 调度器)。**只有把 KV 压满才看得到区别。**

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-21 |
| 里程碑 | M2 |
| 旋钮 | `batch_scheduler_policy`:`max_utilization` → `guaranteed_no_evict`(R) |
| 引擎/层 | small(0.5B),直连 engine,**fraction 调到 0.05 制造 KV 压力** |

## ① 假设
KV 紧张时,激进的 max_utilization 吞吐更高(admit 更多),但偶尔要"反悔重算"造成尾延迟;保守的 no_evict 吞吐低但稳。

## ② 预测(动手前)
max_util 吞吐更高、尾延迟偶发;no_evict 吞吐低、延迟稳。

## ③ 实验设置
**前两次失败:** C=64、长序列在 fraction=0.25 下根本压不满(池子 6857 块太大,`paused=0`,两种策略没区别)。→ 把 `kv_cache_free_gpu_mem_fraction` 降到 **0.05**(池子缩到 1372 块 ≈ 88K token),负载需求 2496 块 >> 池子 → **真正的 KV 压力**。C=64,in2000+out500。

## ④ 实测(fraction=0.05,KV 压满)
| 策略 | 吞吐 | TTFT p50 | TTFT p99 | ITL p99 | peak paused |
|---|---|---|---|---|---|
| **max_utilization** | 2346 tok/s | **2004 ms** | 9013 ms | 186 ms | 1 |
| guaranteed_no_evict | 2442 tok/s | **6763 ms** | 8100 ms | 183 ms | 0 |

**首 token 延迟 TTFT p50(本实验的主信号):**
```
max_util   ███                 2004 ms   ← 激进 admit,请求早开工,首字快
no_evict   ██████████          6763 ms   ← 保守,没 KV 就在队列里等,首字慢 3.4×
```
**吞吐(几乎一样,no_evict 略高):**
```
max_util   ███████████████████████  2346 tok/s   （驱逐→重算,有点浪费)
no_evict   ████████████████████████ 2442 tok/s   （不驱逐,无重算浪费,略高)
```

## ⑤ Gap 分析
**预测错了重点!** 我以为差异在"吞吐",结果吞吐几乎一样(no_evict 还略高);真正的差异在 **TTFT(3.4×)**。
- **max_util**:激进 admit → 请求早早开工 → 首字快(2s);KV 满了就**驱逐+重算**(paused=1、p99 尾巴 9s) → 重算是浪费,吞吐略低。
- **no_evict**:保证已 admit 的不被驱逐 → KV 不够就让请求**在队列里干等** → 首字慢(6.8s);但跑起来不被打断、无重算 → 吞吐略高、ITL 稳。
- 还有个前提教训:**没压满 KV 时两者完全一样**——得先把池子缩到小于负载需求,策略才有意义。

## ⑥ vLLM 机制
对照 vLLM 的 preemption:KV 不够时要么 **recompute**(丢弃后重算,≈max_util 的驱逐重算)要么 **swap**(换出到 CPU)。no_evict 则干脆不抢占、改为限制 admit。

## ⑦ 结论(大白话)
KV 紧张时:
- **max_utilization = "先上车再说,挤不下就请人下车、回头重新排队"** → 首字快(eager),代价是重算浪费 + 尾延迟。
- **guaranteed_no_evict = "上了车保证送到站,没座的在站台等"** → 首字慢(排队),但跑起来稳、无浪费。
- 默认用 max_util,是为了**低延迟 + 高利用率**(大多数在线服务要的)。

**下一步:** [实验5 — KV 显存占比 → 能并发多少](./0007-kv-mem-fraction.md)。
