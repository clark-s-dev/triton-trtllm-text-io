# 0013 · toy 工件:连续批处理模拟器(M2)— 对着实验台校准

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md) §7。这是 M2 的 toy 产物:[`lab/cbatch_sim.py`](../../lab/cbatch_sim.py)。
> 校准目标全部来自我自己在 L4 上量到的数:[0003](./0003-continuous-vs-static-batching.md)(连续 vs 静态)、[0006](./0006-scheduler-policy.md)(调度策略)、[0008](./0008-max-batch-size.md)(batch_size 拐点)、[0001](./0001-m0-kv-memory.md)(两个天花板)。

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-22 |
| 里程碑 | M2(调度器 / 连续批处理),toy 工件 |
| 旋钮 | 无(toy 复现已测旋钮:`gpt_model_type`、`max_batch_size`、`batch_scheduler_policy`) |
| 引擎/层 | 纯 Python 模拟器,复用 [`lab/paged_kv.py`](../../lab/paged_kv.py) 做 KV 记账;无 GPU |

## ① 假设
一个 **iteration-level** 的玩具调度器 + 一条线性 iteration 成本模型 `T(B) = T_w + B·t_token`,应当足以复现我在真引擎上量到的两件事:(1) 吞吐随并发上升到 `max_batch_size` **拐点**后封顶;(2) 连续批处理对静态批(V1)的吞吐/延迟优势。**toy 复现得上,说明我把机理建对了;复现不上的地方,就是真实调度器多做的事。**

## ② 预测(动手前)
- 拐点:吞吐在 `C = max_batch_size` 处封顶;`C > max_batch_size` 时多余请求排队,吞吐平、TTFT 爆。
- 连续 vs 静态:连续 ~2× 吞吐(0003 实测 2.1×),静态 TTFT 因 head-of-line 高一两个数量级。
- 成本模型:一次权重加载被整个 batch 摊薄(`T_w`),所以小 batch 翻倍并发≈翻倍吞吐,大 batch 收益递减。

## ③ 实验设置(可复现)
```bash
python3 lab/cbatch_sim.py          # 打印拐点表 + 连续/静态对照
python3 tests/test_cbatch_sim.py   # 断言拐点形状、连续>静态、KV 交叉点、策略抢占
```
- **成本模型常数**(`T_w=6.1ms`、`t_token=0.071ms`)拟合自 [0008](./0008-max-batch-size.md) 的 0.5B 引擎:用 `B=16→2202`、`B=64→6086` 两点反解。`t_prefill=0.0065ms/tok` 锚定 0003 的 `C=1 in=128 TTFT=7ms`。
- 吞吐取**饱和窗口**(occupancy ≥ 0.9×peak 的迭代),对齐引擎 benchmark 的 warmup 方法学(L2-LAB §4.5)。
- closed-loop:`n_clients=C`,完成一个补一个 —— 和 `perf_benchmark.py` 一致。

## ④ 实测(toy 输出 vs 引擎实测 [括号])
**拐点(吞吐 tok/s vs 并发,饱和值):**
| C | toy bs16 | toy bs64 | toy bs128 | 引擎 [bs16 / bs64 / bs128] |
|---|---|---|---|---|
| 16  | 2169 | 2169 | 2169 | [2202 / — / —] |
| 32  | 2169 | 3717 | 3717 | [2516 / 3596 / 3810] |
| 64  | 2169 | **5783** | 5783 | [~2533 / **6086** / 5999] |
| 128 | 2169 | 5783 | **8019** | [~2533 / 6630 / **6999**] |

每条曲线在自己的 `max_batch_size` 处封顶 —— **拐点复现,§7 通过条件达成**。bs16 平台 2169(引擎 2202,差 1.5%)、bs64@C64 5783(引擎 6086,差 5%)都对得上。

**连续 vs 静态(bs=64,C=32,in=128,out 16–256):**
| 模式 | 吞吐 | TTFT p50 | 平均在跑 |
|---|---|---|---|
| 连续(inflight) | 3532 tok/s | 9.1 ms | 29.8 / 32 |
| 静态(V1) | 2344 tok/s | 715.9 ms | 17.6 / 32 |
- 吞吐比 **1.51×**,TTFT 差 **79×**;静态平均占用 17.6 ≈ `sum(len)/max(len)`(变长负载下短请求早做完、槽位空转的解析预测),**head-of-line 机理对上了**。

**两个天花板交叉点(no_evict,reserve 全 footprint):** ctx 2080→peak 64(batch_size 绑定),3532→54、4532→41、8032→23(KV 绑定,= `min(64, pool/footprint)`)。交叉点 ≈ ctx 2944–2979,正是 [0001](./0001-m0-kv-memory.md) 手算的 `190656/64 = 2979`。

## ⑤ Gap 分析(预测 vs 实测)—— 复现不上的地方就是这一课
1. **`B=128` toy 偏乐观 +14.6%**(8019 vs 6999)。线性成本模型只建模"权重加载摊薄",没建模**算力饱和**:真 L4 在大 batch 下 compute-bound,`T(B)` 超线性增长(实测 bs64→bs128@C128 只 +5.6%,见 0008 ⑤),toy 的线性 `T(B)` 抓不到这个上弯。**这正是 roofline 的 compute 屋顶。**
2. **连续/静态只有 1.51× 而非 2.1×**。占用比(29.8/17.6 ≈ 1.69)是对的,但被**亚线性成本模型压扁**了(`T_w` 摊薄让低占用的静态没被按比例惩罚);加上真 V1 还**整段非流式返回 + 强制 no_evict**(0003 ⑤)——这两件 toy 没建模的事让真实差距更大。
3. **max_util 在严重超额认购下会 recompute 风暴**(footprint-heavy 负载:553 次抢占、TTFT 3765ms vs no_evict 19ms),但在**轻度** KV 压力下(0006 的 fraction=0.05,peak paused=1)max_util 反而 TTFT 更低。**toy 揭示了 0006 没覆盖的一课:max_util 的优劣是 regime-dependent 的** —— 轻度压力 eager-admit 赢,重度压力抖动输。真引擎用 chunked/swap 缓解,toy 只有 evict-to-back+recompute,所以风暴更纯粹。

> 这三条都不是 bug,是 toy 的边界。按 §7:**不硬凑**——把 gap 写下来,它就是 M2 的核心收获。

## ⑥ vLLM 机制(解释层)
对照 `vllm/v1/core/sched/scheduler.py` 的 `schedule()`:每个 step 重新 admit/evict,正是我 `Simulator.run()` 主循环每 iteration 做的事。`max_num_seqs` = 我的 `max_batch_size`(拐点);`num_gpu_blocks` = `kv_pool_blocks`(KV 天花板)。抢占两条路 **recompute vs swap**:我只实现了 recompute(evict→丢弃→重排队),对应 `max_utilization`;`guaranteed_no_evict` 则改为 admission 时 reserve 全 footprint、绝不抢占。

## ⑦ 结论 / 下一步
**能讲给面试官:** "我写了个 80 行的 iteration-level 连续批处理模拟器,用一条 `T(B)=T_w+B·t` 成本模型(拟合自我自己 L4 上的 batch sweep)复现了吞吐-并发拐点(bs16 平台 2169 vs 实测 2202)和连续 vs 静态的 head-of-line(占用 30 vs 17)。它在 `B=128` 偏乐观 14%——因为没建模算力饱和——这个 gap 恰好就是 roofline 的 compute 屋顶。"
**下一步:** [0014 — toy 分页 KV 分配器(M3)](./0014-toy-paged-kv-allocator.md),已被本模拟器复用做 KV 记账。
