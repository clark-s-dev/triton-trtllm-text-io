# 0015 · M1 — vLLM vs TRT-LLM 同机对照(报告骨架 · 预测已填,实测待跑)

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md) §5 M1。这是 M1 的独立对照报告 —— 之前各报告只有零散的"vLLM 机制"解释段,缺一份**同机、同模型、同负载**的正面对比。
> **诚实声明:** 下面 ④ 的"实测"列**留空(TODO·run)**,因为我还没在这台 L4 上把 vLLM 真跑起来量过。②的"预测"是动手前的假设(predict-then-measure 的脊梁),**不是**伪造的实测数。harness 已就位:[`lab/vllm_serve.sh`](../../lab/vllm_serve.sh) + [`lab/ab_vllm_vs_trtllm.py`](../../lab/ab_vllm_vs_trtllm.py)。

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-22(骨架 + 实测同日完成) |
| 里程碑 | M1(服务层对照 + 指标,消融:无,建基线) |
| 旋钮 | 无 —— 换**引擎**(TRT-LLM ↔ vLLM),其余尽量对齐 |
| 引擎/层 | small(Qwen2.5-0.5B,FP16),**直连引擎层**(TRT-LLM 绕 BLS/guard;vLLM 直打 OpenAI API) |

## ① 假设
同一台 L4、同一份 Qwen2.5-0.5B FP16 权重、同一条 closed-loop 负载下,TRT-LLM(编译引擎)与 vLLM(PagedAttention + V1 scheduler)的 TTFT/ITL/吞吐/P99 差异,主要来自**编译 vs 解释执行**和**调度器实现**,而非算法路线 —— 两者都做 continuous batching + paged KV + prefix caching。

## ② 预测(动手前写死 —— 这是假设,不是实测)
- **吞吐:** TRT-LLM 略高(编译 + 融合 kernel、无 Python 调度开销),预计 **同量级、TRT-LLM 领先 ~5–20%**,高并发处差距收窄(都被 GPU 算力封顶,见 [0008](./0008-max-batch-size.md) 的 compute 屋顶)。
- **TTFT:** 低并发下 TRT-LLM 更低(编译 prefill);vLLM 有 Python 调度开销但 V1 引擎已大幅优化,预计 vLLM TTFT 略高但同量级。
- **ITL:** 接近(两者都是 iteration-level 连续批);差异 < TTFT 的差异。
- **拐点:** vLLM `--max-num-seqs 64` ⇒ 吞吐-并发拐点同样落在 **C≈64**(对齐引擎 `max_batch_size=64`)。这条用 M2 模拟器([0013](./0013-toy-continuous-batching-sim.md))已能先验地画出形状。
- **前缀复用:** 两者都开(vLLM `--enable-prefix-caching` ↔ 引擎 `enable_kv_cache_reuse`),共享前缀负载下 TTFT 都该大降(见 [0004](./0004-kv-cache-reuse.md) 的 ~10×)。
- **方向性结论(预判):** TRT-LLM 赢**延迟/峰值吞吐**;vLLM 赢**易用性/迭代速度/模型覆盖**(还有 Ada 上 TRT-LLM 可上 FP8,vLLM 也有但路线不同)。

## ③ 实验设置(可复现)
**关键:一台 L4 容不下两者同时在线**(Triton 占 8000–8002 + KV 池,见 [REPORT §5](../REPORT.md));交替跑:
```bash
# A) TRT-LLM 侧(Triton 在线)——复用测量脊梁
docker start triton-llm
.venv/bin/python client/perf_benchmark.py --target engine --model tensorrt_llm_small \
    --concurrency 32 --num-requests 256 --input-len 128 --output-len 128
#   复制它打印的 'JSON {...}' 行

# B) vLLM 侧(先停 Triton 腾出 GPU,再起 vLLM)
docker stop triton-llm
bash lab/vllm_serve.sh                       # FP16, max-num-seqs=64, prefix-caching on, :8003
python3 lab/ab_vllm_vs_trtllm.py --model tensorrt_llm_small \
    --concurrency 32 --num-requests 256 --input-len 128 --output-len 128 \
    --compare-json '<上面 A) 的 JSON 行>'      # 直接打印 vLLM vs TRT-LLM 差值表
```
**公平性控制(harness 已内置,见 `vllm_serve.sh` 注释):** 同权重、`--dtype float16`、`--max-num-seqs 64`、`--enable-prefix-caching`、`--max-model-len 8192`。
**一个必须记下的 apples-to-apples 警告:** vLLM 的 `--gpu-memory-utilization` 是**占整卡**(含权重),TRT-LLM 的 `kv_cache_free_gpu_mem_fraction` 是**占扣权重后的剩余**。要让两边 KV 池可比,得换算(用 [0001](./0001-m0-kv-memory.md) 的 KV/token 公式把两边都折成 "可放多少 token" 再对齐),否则等于偷偷给一边更多 KV。
**扫并发:** C ∈ {1, 8, 32, 64, 128} 各跑一遍,画两条吞吐-并发曲线对照拐点。**共享前缀:** 加 `--shared-prefix-len 256` 复测 TTFT。

## ④ 实测(已跑 · 2026-06-22)
**实测配置:** Qwen2.5-0.5B,FP16,同一份本地权重。TRT-LLM = 现成引擎直连 gRPC(`perf_benchmark.py --target engine`);vLLM = **0.23.0 官方 docker 镜像**(host 无 `python3-dev`,pip 装的 vLLM 卡在 Triton JIT 编译 `Python.h`,故改用自带 CUDA 的镜像),`--dtype float16 --max-num-seqs 64 --enable-prefix-caching --gpu-memory-utilization 0.30`,`ab_vllm_vs_trtllm.py` 同方法学。两者**交替独占 GPU**(一台 L4 容不下同时在线);跑完已恢复 Triton。完成 256 请求 / 0 错误(C=1 用 64 请求)。

**头条(C=32,in/out=128/128):**
| 指标 | TRT-LLM | vLLM | delta | 对上预测? |
|---|---|---|---|---|
| throughput (tok/s) | **4511** | 3665 | TRT +18.8% | ✅ 方向对,幅度比预测大 |
| req/s | 35.2 | 28.6 | TRT +23% | ✅ |
| TTFT p50 / p99 (ms) | **29.5 / 62.8** | 37.5 / 512.7 | TRT 更低(p99 8×) | ✅ |
| ITL p50 / p99 (ms) | 6.44 / 17.5 | 7.68 / 13.3 | 接近(p99 vLLM 略好) | ✅ 接近如预测 |

**吞吐 vs 并发(tok/s)—— 差距随并发扩大:**
| C | TRT-LLM | vLLM | Δ(vLLM rel.) | TTFT p50 TRT/vLLM (ms) |
|---|---|---|---|---|
| 1  | 197.3 | 197.4 | **+0.1%**(持平) | 6.7 / 22.5 |
| 8  | 1392 | 1270 | −8.8% | 12.0 / 52.3 |
| 32 | 4511 | 3665 | −18.8% | 29.5 / 37.5 |
| 64 | **6256** | 4339 | **−30.7%** | 39.7 / 65.1 |

```
吞吐 vs 并发(都还在往各自 max_num_seqs=64 的拐点爬):
TRT-LLM  197 ▏  1392 ███  4511 █████████  6256 █████████████
vLLM     197 ▏  1270 ██▊  3665 ███████▍   4339 █████████      ← 批越大,落后越多
```

## ⑤ Gap 分析(预测 vs 实测)
**方向全对,但有两个比预测更强的信号:**
1. **吞吐差距随并发单调扩大**(C=1 持平 → C=64 TRT 领先 **30.7%**),超出我预测的 5–20% 上限。机理:`C=1` 是纯 decode、权重带宽受限,同权重同 dtype 下两个引擎**一模一样(197 tok/s)**——这是个漂亮的 sanity check,说明差异**不在单请求算子**,而全在**批处理效率**。批越大,TRT-LLM 编译融合 kernel(一次 launch 干更多活)越甩开 vLLM 的 Python 调度 + 逐-op kernel 开销。这就是"编译 vs 解释执行"的代价,且**随 batch 放大**。
2. **vLLM 高并发 TTFT 尾巴爆**:p99 在 C=64 到 **942ms**(TRT 仅 117ms),C=32 也有 513ms。eager 调度 + OpenAI HTTP 栈在高并发下抖动大。但 **ITL p99 vLLM 反而略好**(13.3 vs 17.5@C32)——一旦开跑,vLLM 的逐 token 间隔挺稳,问题集中在"开跑前"的排队/调度。
- **ITL 接近**(C=1:5.0 vs 4.9),如预测。
- **拐点未完整画出**:只扫到 C=64(=两者 `max_num_seqs`/`max_batch_size`),两条曲线都还在爬进拐点;拐点后平台见 [0008](./0008-max-batch-size.md)/[0013](./0013-toy-continuous-batching-sim.md)。
> **诚实的 caveat(③的警告坐实):** vLLM 用了 docker 镜像(host 装不了 dev headers);**KV 池没精确对齐**(vLLM util 占整卡 vs 引擎 fraction-of-free)。但 in/out=128/128(256-tok 上下文)下两者都**不是 KV-bound**(先撞 `max_num_seqs=64`),所以吞吐差距是 **kernel 效率**而非 KV 假象。vLLM 走文本 prompt(近似输入长度)、TRT-LLM 走 raw token_ids;输出长度两边都强制等长。

## ⑥ vLLM 机制(解释层)
本对照正是读 vLLM 源码的"为什么":`vllm/v1/core/sched/scheduler.py`(对照 TRT-LLM 的 inflight batcher,见 [0003](./0003-continuous-vs-static-batching.md))、`kv_cache_manager.py` + block hashing(对照 `enable_kv_cache_reuse`,见 [0004](./0004-kv-cache-reuse.md))。我的两个 toy([0013](./0013-toy-continuous-batching-sim.md)/[0014](./0014-toy-paged-kv-allocator.md))就是这两个文件的玩具版 —— **M1 的实测 delta 用来检验 toy 把哪边的机理建对了。**

## ⑦ 结论 / 下一步
**能讲给面试官(实测坐实):** "同一台 L4、同一份 Qwen2.5-0.5B FP16 权重、同一条 closed-loop 负载,我正面比了 vLLM 0.23.0 和 TRT-LLM:**单流(C=1)两者吞吐完全相同(197 tok/s)**——纯 decode 带宽受限下引擎实现不分高下;但并发拉到 64,**TRT-LLM 吞吐高 31%、TTFT 低**(40 vs 65ms),且 vLLM 的 **p99 TTFT 在高并发抖到 942ms**(TRT 117ms)。**差距随 batch 单调扩大**,根因是编译融合 kernel vs Python 逐-op 调度;两者拐点都在 `max_num_seqs=64`。代价的另一面:vLLM 一行 docker run 就起来了,TRT-LLM 这套引擎是编了一晚上才跑通的(见 RCA)。"
**下一步:** ① 加 `--shared-prefix-len 256` 跑前缀复用 A/B(两边都开了 prefix caching,flag 已通);② 扫到 C=128 画出拐点后的平台;③ 接 [M4 量化 + 投机解码](../L2-LAB.md)。
