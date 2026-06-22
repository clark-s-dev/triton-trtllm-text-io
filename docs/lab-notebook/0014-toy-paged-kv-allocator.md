# 0014 · toy 工件:分页 KV 块分配器(M3)— 对着实验台校准

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md) §7。这是 M3 的 toy 产物:[`lab/paged_kv.py`](../../lab/paged_kv.py)。
> 校准目标来自 [0004](./0004-kv-cache-reuse.md)(KV 复用,共享前缀)和 [0007](./0007-kv-mem-fraction.md)(KV 池大小 → 并发)。本分配器也被 M2 模拟器([0013](./0013-toy-continuous-batching-sim.md))复用做 KV 记账。

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-22 |
| 里程碑 | M3(KV 缓存 / PagedAttention),toy 工件 |
| 旋钮 | 无(toy 复现 `enable_kv_cache_reuse` ON/OFF 的 `reused` 指标) |
| 引擎/层 | 纯 Python 分配器:block table + 引用计数 + 前缀哈希 + LRU 驱逐;无 GPU |

## ① 假设
把"分页 KV + 前缀复用"拆成三件最小机制——**固定大小块的 block table、按前缀滚动哈希做复用键、引用计数 + LRU 驱逐**——应当能从机理上复现引擎在共享前缀负载下报出的 `reused` KV 块数。

## ② 预测(动手前)
- 0004 负载:输入 4000 tok,前 **3968 = 62 块**(`tokens_per_block=64`)是所有请求共享前缀,尾 32 tok 各异。
- **第一个请求冷**(铺路,reused=0),其后每个**整段命中前缀 62 块**。
- 引擎报 `reused=2418`。`2418 / 62 = 39` → 模型应是 **1 冷 + 39 热**,即 `(N−1)×62` 且 **N=40**。块级命中率 `2418 / (40×62) = 97.5%`。reuse OFF → `reused=0`。

## ③ 实验设置(可复现)
```bash
python3 lab/paged_kv.py            # 复现 0004 的 reused=2418
python3 tests/test_paged_kv.py     # 断言 2418、前缀性质、引用计数、LRU、OOM
```
- 滚动哈希 `block_hash(i) = FNV1a(block_hash(i-1), tokens_i)` —— 把父块哈希链进去,使键成为**前缀身份**:哈希相同 ⇔ 前缀完全相同。复用按块顺序走,**遇到第一个 miss 就停**(前缀性质)。
- 只有**整块**入缓存;尾部不满块永不复用(对齐真引擎)。
- `free()` 只减引用计数**不回收**:refcount=0 的缓存块仍可复用,只有池满时才 LRU 驱逐。

## ④ 实测(toy 输出 vs 引擎实测)
| 配置 | toy `reused` 块 | toy 命中率 | 引擎(0004) |
|---|---|---|---|
| reuse **ON**(N=40) | **2418** | 97.5% | `reused = 2418` |
| reuse **OFF** | 0 | 0% | `reused = 0` |

- `2418 = (40−1)×62`,**对到块**。模型即 `reused = warm_requests × prefix_blocks`。
- 被 M2 模拟器复用时(N=20、in=4000、C=1):热请求 TTFT **6.3ms** vs reuse-OFF **74.8ms**(整段 8 块 chunked prefill)——对上 0004 实测的 **8.5ms vs 82ms**。复用确实把"重算开头"省掉了。
- 单元测试另外钉死:前缀性质(块 3 起分叉 → 只复用 2 块)、引用计数(两并发同前缀 → 一份物理块 refcount=2)、LRU(20 块小池跑 30 个异构请求 → 驱逐发生且无块泄漏)、OOM(块全 live → `ok=False`)。

## ⑤ Gap 分析(预测 vs 实测)
1. **N 是反推的。** 0004 没记 `--num-requests`;但 `2418 = 39×62` 唯一地把它钉在 **N=40(1 冷 + 39 热)**。这反过来是个漂亮的交叉验证:**引擎的 `reused` 计数器字面就是 `热请求数 × 前缀块数`**。0004 里 C=1 和 C=4 都报 2418,与"只有 1 个冷请求"一致(头一个铺好缓存,并发的后来者也命中)。
2. **toy 没建模的(= 真引擎多做的):** chunked prefill 与复用的交互、抢占时的 swap-to-CPU、滑窗注意力。这些在重负载/长序列下会让真实命中率偏离 toy 的干净 `(N−1)×62`。
3. **容量压力下的碎片化。** toy 的 LRU 在小池下能复现"驱逐发生"(测试已验证),但真引擎的块碎片化、跨请求的部分命中比 toy 复杂——把这条留给 0007 的 fraction 扫描去量(toy 只证明了机制方向)。

> 按 §7:复现得上的(reused、命中率、TTFT 量级)说明机理对了;复现不上的(N 反推、碎片化)写下来就是 M3 的边界课。

## ⑥ vLLM 机制(解释层)
对照 `vllm/v1/core/kv_cache_manager.py` + `block_pool.py`:vLLM 也按块做 **content hash**(把 parent block hash 链进去,和我的 FNV 滚动哈希同构),`ref_cnt` 管共享,`FreeKVCacheBlockQueue` 做 LRU evictor —— refcount=0 的块进 free queue 尾、仍可被 hash 命中救回,正是我 `free()` 不回收缓存块的那条。我的 `reused=2418` 就是 vLLM `prefix cache hit` 计数的玩具版。

## ⑦ 结论 / 下一步
**能讲给面试官:** "我写了个分页 KV 分配器:固定块 + 前缀滚动哈希 + 引用计数 + LRU。喂进 0004 的共享前缀负载,它把引擎报的 `reused=2418` 复现到了块——而且 `2418=39×62` 反推出当时跑了 40 个请求(1 冷 39 热),说明引擎那个计数器字面就是‘热请求 × 前缀块’。" 这一句覆盖了 PagedAttention 的分页、前缀缓存、引用计数三件事。
**下一步:** 两个 toy 已闭环;把 M1 的 vLLM vs TRT-LLM 同机对照补上(harness + 报告骨架见 [0015](./0015-m1-vllm-vs-trtllm.md)),L2 主线就齐了。
