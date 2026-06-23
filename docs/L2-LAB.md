# L2 推理引擎实验台 — Ablate-on-your-own-rig

> **这份文档是「怎么做」的执行手册;方向/路线在 [`AI-INFRA-DIRECTION.md`](./AI-INFRA-DIRECTION.md)。**
> 那篇说「往引擎层下沉、vLLM 学原理、TRT-LLM/Dynamo 主场深耕」;**这篇把它变成一台你现在就能跑的消融实验台。**
>
> **结论先行:** 你手上是一台「**每个 L2 旋钮都已拧到 ON、且大半个旋钮改 `config.pbtxt` + `docker restart` 就能拧 OFF**」的实验台。
> 网上的 L2 roadmap 把脊梁放在「读 vLLM 源码 + 凭空写 toy」;对**有这台机器的你**,正确的脊梁是
> **在自己的系统上做消融**:关掉一个旋钮 → 先预测数字 → 量回归 → 再用 vLLM 源码解释你刚量到的 delta。
> 读源码和写 toy 从「主线」降级为「解释层」。"能改才算到"——你不用等能改 vLLM,你现在就能改自己的 serving,改一个、测一个、解释一个。

---

## 0. 诊断:你已有什么 vs roadmap 的盲点

| roadmap 假设你 | 你实际有的(已确认) |
|---|---|
| "你现在是 FP16,没开" | 引擎 config 里 **L2 核心旋钮全 ON**:`inflight_fused_batching`、`paged_kv_cache`、`use_paged_context_fmha`、`enable_kv_cache_reuse`、`enable_chunked_context`、`batch_scheduler_policy=max_utilization` |
| 去别处读源码、凭空搭 toy | 一台**正在跑**的 Triton + TRT-LLM 栈(`triton-llm`),双引擎(0.5B/1.5B),全套 observability(Prometheus/Grafana/Jaeger/DCGM)在线 |
| 把测量留到 M1 才做 | configs 是 **volume-mount** 的 → 改参数 `docker restart triton-llm` 即生效,**消融成本 ≈ 0** |
| —— | ~~`client/perf_benchmark.py` 不存在~~ → **已建**(§6 测量脊梁就位),M0–M3 已用真实数字闭环,toy 工件(§7)已交付,**M1 vLLM vs TRT-LLM 同机实测(0015)、M4 量化 + 投机解码(0016)+ kernel 级带宽实锤(0017)均已完成**。 |

---

## 1. 五条原则

1. **跑着的机器是实验室,不是脚注。** 每个概念都落到一个你在自己 L4 上量出来的数字。
2. **靠消融理解(ablate-to-understand)。** 旋钮已 ON,理解它最快的方式是**关掉看什么变慢/变坏**。
3. **先预测,再测量(predict-then-measure)。** 动手前先写下假设 + 一个 roofline 量级估算;**预测与实测的 gap 才是学习信号**。每条实验记进 [`lab-notebook/`](./lab-notebook/)。
4. **vLLM 源码是「为什么」,不是「主线」。** 带着你测到的 delta 去读对应文件,比冷读快十倍。
5. **toy 要对着实验台校准。** toy scheduler/allocator 要能复现你量到的曲线;复现得上(或可解释地复现不上)才算真 artifact。

---

## 2. 旋钮地图:runtime(改 config→restart)vs build(重编引擎)

**这是 L2 的第一个分水岭**:同样一个"特性",有的是运行时调度器行为(改参数即可),有的烧进了引擎图(必须重编)。搞清楚边界,消融成本和顺序就清楚了。

| 旋钮 | 类型 | 在哪改 | 备注 |
|---|---|---|---|
| `gpt_model_type`(inflight_fused_batching ↔ `V1` 静态批) | **R** | `config.pbtxt` | V1 在 0.14 可能被弃用 → 改完看 `docker logs` 确认是否被接受 |
| `batch_scheduler_policy`(max_utilization ↔ guaranteed_no_evict) | **R** | `config.pbtxt` | 控制 admit/evict 激进程度 |
| `enable_kv_cache_reuse`(true ↔ false) | **R** | `config.pbtxt` | 前提:引擎已用 `--use_paged_context_fmha enable` 编译(你已是)→ 所以可运行时切 |
| `enable_chunked_context`(true ↔ false) | **R** | `config.pbtxt` | 同上前提 |
| `kv_cache_free_gpu_mem_fraction`(0.25/0.45 → 扫 0.1~0.9) | **R** | `config.pbtxt` | 直接决定 KV pool 大小 → 最大并发 |
| `max_tokens_in_paged_kv_cache` / `max_num_tokens` | **R** | `config.pbtxt` | 另一组 KV/调度上限,可与 fraction 对照 |
| `paged_kv_cache` / `use_paged_context_fmha`(enable ↔ disable) | **B** | `build_engines.sh` | 关掉 = 退回非分页 KV,代价大,但能让你看到 PagedAttention 到底省了什么 |
| `max_batch_size`(=64) | **B** | `trtllm-build` | **当前并发的硬上限**,见 §5 M0 |
| `max_beam_width`(=1 → 4) | **B** | `trtllm-build`(你没传,默认 1) | beam>1 必须重编 |
| dtype / 量化(FP16 → FP8 / INT4-AWQ) | **B** | `convert_checkpoint.py` + `trtllm-build` | M4 主场 |

> 执行顺序建议:**先把所有 R 行刷一遍(每行 5 分钟:改参数 → restart → 测),再做 B 行(重编以小时计)。**

---

## 3. ★ 核心产物:消融矩阵

每一行 = 一个 L2 子系统。**先填「预测」列(写下数字 + 机理),再去填实测。** 模板见 [`lab-notebook/TEMPLATE.md`](./lab-notebook/TEMPLATE.md)。

| 旋钮(现值) | 关/改成 | R/B | 先预测(数字 + 机理) | 测什么 | 读 vLLM 哪里 |
|---|---|---|---|---|---|
| `gpt_model_type=inflight_fused_batching` | `V1`(静态批) | R | 并发=32 时吞吐掉 ~?×:静态批要等 batch 里**最长**序列做完才放新请求 → head-of-line blocking,GPU 在已完成 slot 上空转 | throughput、TTFT P50/P99 vs 并发 | `vllm/v1/core/sched/scheduler.py` 的 `schedule()`(iteration-level) |
| `enable_kv_cache_reuse=true` | `false` | R | **共享 200-tok system prompt** 的 workload 下,reuse ON 让 TTFT 降 ~(shared/total);OFF → 重复算 prefill,TTFT 抬头 | 共享前缀 workload 的 TTFT;KV `reused` 指标 | `vllm/v1/core/kv_cache_manager.py` + block hashing |
| `enable_chunked_context=true` | `false` | R | OFF:一条 3500-tok 长 prefill 卡住其他流 decode → 在途流 ITL 尖刺;ON:prefill 切块与 decode 交织,ITL 平 | 一条 decode 流的 ITL 抖动(在长 prefill 进入时) | scheduler 里 chunked-prefill / `long_prefill_token_threshold` |
| `batch_scheduler_policy=max_utilization` | `guaranteed_no_evict` | R | max_util 激进 admit、KV 压力下会 evict/recompute(吞吐高、偶发 recompute 尾延迟);no_evict 保守(吞吐低、无驱逐尾延迟) | KV 压力下(高并发+长序列)吞吐 vs 尾延迟 | scheduler 的 preemption:recompute vs swap |
| `kv_cache_free_gpu_mem_fraction` | 扫 0.1→0.9 | R | 最大并发序列 ≈ 正比于 KV blocks;太低 → 排队(吞吐悬崖),太高 → OOM/抢另一引擎显存 | 排队前最大并发、吞吐 vs fraction | block pool 大小 / `num_gpu_blocks` |
| `max_batch_size=64` | 重编=16 / 128 | B | 短 context 下它是**真正的并发上限**(见 M0);改它直接平移吞吐天花板 | 吞吐 vs max_batch_size(短/长 context 两条曲线) | scheduler 的 `max_num_seqs` |
| `max_beam_width=1` | 重编=4 | B | beam=4 让 KV+decode ~4×、吞吐掉 ~4×;质量? | throughput、输出质量 | beam kernels(读懂即可) |
| dtype=FP16 | **FP8**(Ada 支持)/ **INT4-AWQ** | B | weight-only INT4 主要加速 **decode**(小批 decode 带宽受限、权重流量占大头),**TTFT 几乎不受益**(prefill compute-bound);FP8 KV → KV 容量 2× → 并发更高 | tok/s、TTFT、精度 delta | —(TRT-LLM 主场) |

---

## 4. 测量的严谨性(L2 与「会调 API」的分水岭)

1. **量对层。** 路径是 `BLS(2 CPU 实例)→ guardrail(1 GPU 实例)→ engine`。穿过 `text_pipeline_bls` 测到的是**网关+护栏+引擎的卷积**(你 `AI-INFRA-DIRECTION.md` §1.1 已抓到"护栏单实例串行、BLS 实例数卡 engine batch")。研究**引擎**调度/KV 时,**直接打 `tensorrt_llm_small/large` 模型**(raw `input_ids` 进,绕开 BLS/guard),否则把网关串行算到引擎头上。工具:Triton `perf_analyzer` 或 TRT-LLM `gptManagerBenchmark`。
2. **workload 要为旋钮量身设计。** KV reuse 只有**共享前缀**看得到;chunked context 要**一条长 prefill + 若干在途 decode**才暴露 ITL 尖刺。**「设计能暴露旋钮的负载」本身就是实验的一半。**
3. **L4 = 单卡、无 NVLink、Ada(sm_89)。** FP8 能做(Ada 有 FP8 tensor core);**FP4 是 Blackwell-only → 只能读不能跑**;TP/PP 单卡跑不起来 → roadmap "#6 了解原理即可"是对的,别在这花动手预算。
4. **小模型/大显存陷阱。** 0.5B/1.5B 在 L4 上 decode 可能便宜到你**被 Python/gRPC 开销卡住而非 GPU**。必须把并发推到吃满 GPU(看 DCGM 的 SM 利用率),batch sweep 才有意义。
5. **方法学:** warmup;取中位数 + P99(N≥几十);先 `docker stop triton-fused` 清场(共享端口+显存);记录 DCGM 功耗/SM 利用率作旁证。

---

## 5. 里程碑(重写版:每个挂一个消融 + 一个产物)

### M0 · 原理地基 —— **已用你的真实数字闭环(见 [`lab-notebook/0001-m0-kv-memory.md`](./lab-notebook/0001-m0-kv-memory.md))**

KV/token 公式:`2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes`(Qwen2.5 是 **GQA**,`num_kv_heads=2` ≪ attention heads)。

| | num_layers | num_kv_heads | head_dim | **预测 B/token** | 引擎实测(启动日志) | gap |
|---|---|---|---|---|---|---|
| **0.5B** (14 heads, hidden 896) | 24 | 2 | 64 | **12,288** | 2.18 GiB / 190,656 tok = 12,277 | **0.09%** |
| **1.5B** (12 heads, hidden 1536) | 28 | 2 | 128 | **28,672** | 7.31 GiB / 273,792 tok = 28,669 | **0.01%** |

**产物 1(过关):** 你的手算预测引擎实际 KV 分配到 **3 位有效数字**。验证命令:
```bash
docker logs triton-llm 2>&1 | grep -iE 'blocks in KV cache|max tokens in paged|maxNumSequences'
# Allocated 7.31 GiB for max tokens in paged KV cache (273792)   ← 1.5B, fraction 0.45 × 16.25 GiB avail
# Allocated 2.18 GiB for max tokens in paged KV cache (190656)   ← 0.5B, fraction 0.25 × 8.73  GiB avail
```

**产物 2(roadmap 没教的 L2 洞察 ——「哪个资源先到顶」):**
- KV 容量上限:1.5B 能放 273,792 tok → 2K context 时 ≈ **133** 条并发序列。
- 但 build 时 `max_batch_size=64` → 日志 `maxNumSequences: 64`。**两个天花板,短 context 下是 batch_size=64 先到顶,不是 KV。**
- 交叉点:`273,792 / 64 ≈ 4278 tok`。**平均 context < 4278 → batch_size 绑定;> 4278 → KV 内存绑定。** 这就是消融矩阵 `max_batch_size`(B)和 `kv_cache_free_gpu_mem_fraction`(R)两行该一起扫的原因。
- 第三课:两次 KV 计算时 available 从 **16.25 → 8.73 GiB**(第一个引擎先吃掉权重+KV)→ **共置模型,加载顺序决定第二个引擎能拿多少**。

**产物 3:** 用上面的数字解释「为什么 batch 越大吞吐越高但延迟变差」+「为什么 decode 是带宽受限」(roofline)。能讲明白 = M0 过关。

### M1 · 服务层对照 + 指标(消融:无,建基线)
同模型同机起 vLLM 和你的 TRT-LLM,直连引擎层测 TTFT/ITL/吞吐/P99。**产物:** vLLM vs TRT-LLM 对比表 + tradeoff 分析。**前置 = 测量脊梁(§6)。**
> **✅ 已实测(2026-06-22):** harness = [`lab/vllm_serve.sh`](../lab/vllm_serve.sh) + [`lab/ab_vllm_vs_trtllm.py`](../lab/ab_vllm_vs_trtllm.py)(同方法学,`--compare-json` 出差值表)。同机同权重(Qwen2.5-0.5B FP16)实测见 [notebook 0015](./lab-notebook/0015-m1-vllm-vs-trtllm.md):**C=1 两者持平 197 tok/s;C=64 TRT-LLM 吞吐高 31%(6256 vs 4339)、TTFT 低、vLLM p99 尾巴爆 942ms**——差距随 batch 扩大(编译 kernel vs Python 调度)。剩:前缀复用 A/B + 扫到 C=128。

### M2 · 调度器 + 连续批处理 ★最关键(消融:`gpt_model_type` inflight→V1、`batch_scheduler_policy`、`max_batch_size`)
读 `vllm/v1/core/sched/`,对照你 inflight→V1 的实测吞吐崩塌。**产物:** 纯 Python 最小连续批处理模拟器——给定一批不同长度请求 + KV 预算,模拟每 iteration 如何 admit/evict、如何混 prefill/decode。**校准:** 它要能复现你在自己引擎上量到的 throughput vs 并发曲线(含 batch_size=64 拐点)。
> **✅ 产物已交付:** [`lab/cbatch_sim.py`](../lab/cbatch_sim.py) + [notebook 0013](./lab-notebook/0013-toy-continuous-batching-sim.md)。复现了拐点、连续 vs 静态(0003)、策略抢占(0006)、KV/batch 两天花板交叉(0001)。

### M3 · KV-cache / PagedAttention(消融:`enable_kv_cache_reuse`、`kv_cache_free_gpu_mem_fraction` 扫描、`paged_kv_cache` B 级)
读 block manager + prefix caching,对照 `enable_kv_cache_reuse` 在共享前缀 workload 上的 TTFT 实测。**产物:** toy 分页 KV 块分配器 + 前缀缓存命中逻辑(block table、引用计数、命中率)。**校准:** 命中率预测对上你 KV `reused` 指标。
> **✅ 产物已交付:** [`lab/paged_kv.py`](../lab/paged_kv.py) + [notebook 0014](./lab-notebook/0014-toy-paged-kv-allocator.md)。把引擎 `reused=2418` 复现到块(命中率 97.5%)。被 M2 模拟器复用做 KV 记账。

### M4 · 量化 + 投机解码 —— **已收尾**
TRT-LLM 重编 FP8 / INT4-AWQ,测精度 vs 吞吐(加速主要在 decode);再开投机解码测加速比。**产物:量化 + 投机解码实测报告。**
> **📄 易懂综述(一页读懂 M4):** [`REPORT-decode-acceleration.md`](./REPORT-decode-acceleration.md) —— 把三种 decode 加速(量化 / 投机解码 / 连续批)统一到「带宽墙」一条主线 + 一张「该用哪个」决策表。
> **✅ 量化(2026-06-21):** FP8([0009](./lab-notebook/0009-fp8-quantization.md))+ INT4-AWQ([0011](./lab-notebook/0011-int4-awq.md)):decode 提速、引擎更小、KV 容量翻倍;精度需真实 eval 另验(性能台量不了)。
> **✅ 投机解码(2026-06-23):** draft 0.5B → target 1.5B,harness [`lab/specdec_bench.py`](../lab/specdec_bench.py) + [notebook 0016](./lab-notebook/0016-speculative-decoding.md)。单流贪心**最优 K=2 → mixed 1.18×、可预测文本 1.50×、创作类 0.97×(反而慢)**;采样**必须用 logits 接受**(1.47×,否则崩到 0.68×);**无损**(输出与 K 无关);**一上 batch 就被连续批处理反超(b2 已 0.86×)** → 投机解码是**低并发 / 延迟敏感**利器,不是高吞吐手段。最优 K 比 roofline 预测(K=4)小,因为 draft 成本(实测 r≈0.42)比纯权重比(0.32)高——`(1+aK)/(1+rK)` 模型骨架对、r 必须实测。
> **✅ kernel 级实锤(2026-06-23):** [notebook 0017](./lab-notebook/0017-nsight-decode-bandwidth.md):本机 CUDA13 驱动比 NGC 24.10 的 `ncu`/`nsys` 新 → PerfWorks 计数器起不来(RCA),改用 **nsys CUDA trace + 逐 kernel 字节记账**,量到 decode 权重 kernel 跑在 **~83% 峰值显存带宽、~0.2% 峰值算力**——「decode 带宽受限」从 roofline 推断升级为**实测**;并发现**未量化的 lm_head GEMV 是量化次线性加速的固定带宽地板**。

### M5 · 持续(TP/PP 只读 + 开源贡献)
L4 跑不了 TP/PP → 读原理 + Dynamo 的 disaggregated P/D。给 vLLM 提一个能 merge 的 PR(benchmark/文档/小修)。

---

## 6. 测量脊梁(✅ 已建 —— 所有里程碑的前置)

`AI-INFRA-DIRECTION.md` 承诺的 [`client/perf_benchmark.py`](../client/perf_benchmark.py) **已就位**(异步 load generator + TTFT/ITL/吞吐/P99,可直连 engine 绕开 BLS/guard)。它本来需要三件,现都已有:
1. **异步 load generator** —— 可直连 `tensorrt_llm_{small,large}`(绕开 BLS/guard,见 §4.1),也可走 BLS;可设并发 C、输入/输出长度分布、共享前缀比例。
2. **指标采集** —— 从 decoupled stream 算 TTFT(首个 TEXT delta)/ ITL(delta 间隔)/ throughput / P50/P99;同时拉 Prometheus + DCGM。
3. **ablation runner** —— 扫一个旋钮 → 改 `config.pbtxt` → `docker restart triton-llm` → 等 ready → 测 → 出表。

指标速查:
```bash
curl -s localhost:8002/metrics | grep -iE 'trt_llm|inference_request_duration'   # 引擎/KV/inflight-batcher 指标(确认你这版的确切名字)
docker logs triton-llm 2>&1 | grep -iE 'blocks in KV cache|maxNumSequences'      # 启动期 KV/并发上限
```

---

## 7. toy 工件(对着实验台校准,不是凭空写)—— ✅ 已交付

两个 toy 都在 [`lab/`](../lab/)(纯 Python、无 GPU、无依赖),`make test` 里跑校准断言,predict-vs-measure 写在 lab notebook 0013/0014。

- **M2 连续批处理模拟器** → [`lab/cbatch_sim.py`](../lab/cbatch_sim.py)([notebook 0013](./lab-notebook/0013-toy-continuous-batching-sim.md))。输入一批 `(arrival, prompt_len, output_len)` + KV 预算 + max_batch_size,iteration-level 输出 batch 占用 / admit / evict + throughput + 每请求 TTFT/完成时刻。**通过条件达成:** 复现了 throughput-vs-并发拐点(bs16 平台 2169 vs 实测 2202;bs64@C64 5783 vs 6086)+ 连续 vs 静态 head-of-line(占用 30 vs 17)。复用 M3 分配器做 KV 记账。
- **M3 分页 KV 分配器** → [`lab/paged_kv.py`](../lab/paged_kv.py)([notebook 0014](./lab-notebook/0014-toy-paged-kv-allocator.md))。block table + 引用计数 + 前缀滚动哈希 + LRU 驱逐。**通过条件达成:** 把引擎 KV `reused=2418` 复现到块(`=39×62`,1 冷 39 热)。

> toy 复现不上时**不要硬凑**——gap 本身是 M2/M3 的核心一课(真实调度器还做了你 toy 没建模的事:chunked、recompute、碎片化)。**已记的 gap:** 成本模型在 `B=128` 偏乐观 +14%(没建模算力饱和)、连续/静态比 1.5× vs 实测 2.1×、max_util 在重度超额认购下会 recompute 风暴(见 0013 ⑤)。

---

## 8. 和本项目的关系

- **解耦:** `text_pipeline_bls` 只吃编译好的 engine。R 级旋钮改 `config.pbtxt` + restart;B 级重编 `build_engines.sh` 的引擎,serving 不动。
- **本 repo = 端到端实验台**,但**测引擎要绕开网关层**(§4.1)。
- **构建现实:** 从源码编 TRT-LLM 很重(M4 量化 / B 级消融进 build 容器);R 级消融在现成 `.venv` + restart 就够。

> 一句话:**方向在 `AI-INFRA-DIRECTION.md`,执行在这里;脊梁是「在自己机器上消融」。M0–M3 用真实数字闭环、测量脊梁(§6)、toy 工件(§7)、M1 vLLM vs TRT-LLM 同机实测(0015)、M4 量化(0009/0011)+ 投机解码(0016)+ kernel 级实锤(0017)均已交付。L2 消融脊梁基本走完;剩 M5(TP/PP 只读 + 开源 PR)。**

---

## 9. 交付记录(2026-06-22)

把"还差的深化/解释层"补成真 artifact 的一次性过程记录。

**起点(缺口):** §7 的两个 toy(M2 模拟器、M3 分配器)当时只在文档里被描述、仓库无代码;M1 只有各报告里零散的"vLLM 机制"解释段,无独立的同机正面对照。

**M2/M3 toy(`lab/`):** 写了 [`paged_kv.py`](../lab/paged_kv.py)(分页块 + 引用计数 + 前缀滚动哈希 + LRU 驱逐)与 [`cbatch_sim.py`](../lab/cbatch_sim.py)(iteration-level 调度器,复用前者做 KV 记账),成本模型 `T(B)=6.1+0.071·B` ms 拟合自 [0008](./lab-notebook/0008-max-batch-size.md)。校准断言并入 `make test`(共 **22 个 GPU-free 测试**):M3 复现引擎 `reused=2418` 到块;M2 复现 `max_batch_size` 吞吐拐点(bs16 平台 2169 vs 实测 2202)+ 连续 vs 静态 head-of-line。predict-vs-measure 与已知 gap 写进 [0013](./lab-notebook/0013-toy-continuous-batching-sim.md)/[0014](./lab-notebook/0014-toy-paged-kv-allocator.md)。

**M1 实测(harness + [0015](./lab-notebook/0015-m1-vllm-vs-trtllm.md)):**
1. 先用现成引擎量 TRT-LLM 基线(直连 gRPC,C=1/8/32/64)。
2. 装 vLLM 踩坑:本机 `sudo`/`apt` 被权限闸门挡死 → 装不了 `python3-dev` → pip 装的 vLLM 死于 Triton JIT 的 `fatal error: Python.h: No such file or directory`。**改用 `vllm/vllm-openai:latest` docker 镜像(0.23.0,自带 CUDA)绕过**(已回填进 `vllm_serve.sh` 注释)。
3. 停 `triton-llm` 腾 GPU(一台 L4 容不下同时在线)→ 起 vLLM(同权重 FP16、`max-num-seqs 64`、prefix-caching)→ 同方法学 A/B → 跑完 **恢复 `triton-llm`**。
4. 结果:**C=1 持平 197 tok/s;C=64 TRT-LLM 吞吐 +31%(6256 vs 4339)、TTFT 更低、vLLM p99 TTFT 爆 942ms**;差距随 batch 扩大(编译 kernel vs Python 调度)。

**收尾验证:** `make test` 22/22 绿;系统恢复原状(`triton-llm` up、GPU ~17.8 GB、vLLM 容器已删)。

---

## 10. 交付记录(2026-06-23)— M4 收尾(投机解码 + kernel 级带宽实锤)

把 M4 还差的两块补成真 artifact:**投机解码实测**(0016)和**「decode 带宽受限」的 kernel 级证据**(0017)。

**起点(缺口):** M4 量化已闭环(0009/0011),但「投机解码」只在路线里写着没跑;且 M0/0009/0011 一直用「decode 带宽受限」下结论,却始终是 roofline **手算推断**,无 kernel 级实测。

**投机解码(0016,artifact = [`lab/specdec_bench.py`](../lab/specdec_bench.py) + [`lab/run_0016_sweep.sh`](../lab/run_0016_sweep.sh) + [`lab/specdec_model.py`](../lab/specdec_model.py)):**
1. 新引擎构建 [`scripts/build_specdec_engines.sh`](../scripts/build_specdec_engines.sh)(`build_engines.sh` 的姊妹):draft 0.5B(`--gather_generation_logits`)+ target 1.5B(`--speculative_decoding_mode draft_tokens_external --max_draft_len 10`),复用 0009 的 Qwen tied-emb 修复。
2. harness 忠实复刻 TRT-LLM 0.14 `examples/run.py:run_draft_target_model()` 的 draft→verify 循环,干净打点(接受率 / mean accept-per-iter / 吞吐 / 加速),baseline 用**普通 fp16 引擎**(spec target 无 draft 时每 call 只出 1 token)。
3. 20 组实测(K 扫描 / easy-hard / logits-tokens / batch 交叉),`docker run` 在 NGC 容器里跑 → [`lab/results_0016.jsonl`](../lab/results_0016.jsonl)。
4. **结论:** 最优 **K=2**(mixed 1.18×、可预测文本 1.50×、创作 0.97× 反慢);采样**必须 logits 接受**(1.47× vs tokens 0.68×);**无损**(K 无关,贪心输出恒 1765 tok);**batch≥2 即被连续批反超**(b2 0.86× → b8 0.52×)→ 投机解码 = 低并发延迟工具。逐迭代耗时线性拟合 `T_iter(K)=16.6+5.59K ms`(残差<1%),back 出 draft 步 5.59ms、r≈0.42(纯权重比 0.32 会高估最优 K)。校准断言并入 `make test`([`tests/test_specdec_model.py`](../tests/test_specdec_model.py),+6 测试 → 共 28)。

**kernel 级实锤(0017,artifact = [`scripts/profile_decode.sh`](../scripts/profile_decode.sh) + [`lab/decode_probe.py`](../lab/decode_probe.py) + `lab/ncu/*.kern.txt`):**
1. **工具坑(RCA):** 本机驱动 580/CUDA13 比 NGC 24.10 的 `ncu`/`nsys` 2024.2.1 新 → `ncu` `Failed to prepare kernel`、`nsys --gpu-metrics` `NVPA_STATUS_ERROR`(`--privileged`/`--clock-control none` 都不解,是版本不兼容非权限)。
2. **绕过:** `nsys --trace=cuda`(CUPTI 计时,不走 PerfWorks)能跑 → 拿 kernel 归因 + duration;配合**逐 kernel 精确权重字节**算达到带宽。
3. **结论:** decode 权重 kernel 实测 **~83% 峰值显存带宽 vs ~0.2% 峰值算力**(lm_head GEMV 82%、transformer GEMM 84%),算术强度 ~1 FLOP/byte(屋脊 400,低 400×)→ 带宽受限**实测坐实**。意外发现:**lm_head 未量化(三 dtype duration 完全相同)= 固定带宽地板**,解释了 0009/0011 量化加速为何次线性(INT4 砍 4× 字节只快 1.6×)。

**收尾验证:** `make test` **28/28 绿**(原 22 + 投机解码模型 6);**`triton-llm` 已 `docker start` 恢复**(ready=200,4 模型全 READY,GPU ~17.8 GB,与实验前一致)。新增引擎(`engines/qwen2.5-0.5b-draft`、`qwen2.5-1.5b-target`)与既有 serving 解耦,不影响网关。
