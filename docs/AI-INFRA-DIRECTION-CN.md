# AI Infra 方向笔记 — 引擎层 (vLLM / TensorRT-LLM) 与算子层地图
# AI-infra direction notes — the inference **engine layer**, plus a **kernel-layer map** of TensorRT-LLM

> 🌐 **English version:** [`AI-INFRA-DIRECTION-EN.md`](./AI-INFRA-DIRECTION-EN.md)

> 从这个 repo 的**服务层**出发,记录往「引擎层 (AI infra)」和「算子层」下沉的地图与路线。
> 算子层地图基于本仓子模块 **TensorRT-LLM v0.14.0** (commit `b088016`),只读调研所得。
> 这是一份**方向/路线笔记**,不是项目交付物。
> 注:文中引用的性能基准(`docs/PERFORMANCE.md` + `client/perf_benchmark.py`)目前在
> `feat/perf-benchmark` 分支,尚未并入 main。

---

## 0. 一张图:推理栈三层,以及「我在哪」

| 层 | 代表 | 关系到本人的现状 |
|---|---|---|
| ① 服务 / 编排层 | Triton Inference Server、KServe、Ray Serve、路由 / 多模型 | ✅ 一直在做(本 repo `text_pipeline_bls` 网关) |
| ② **引擎 / 运行时层** | **vLLM、TensorRT-LLM、SGLang** | ⬅️ **想 focus 的 "AI infra" 就是这层** |
| ③ kernel / 算子层 | CUDA / CUTLASS / OpenAI-Triton、FlashAttention | 🤔 是否再下沉的纠结点 |

**结论先行:** "AI infra (vLLM/TRT-LLM)" = **引擎运行时层**,夹在我现在做的服务层和算子层中间;
它的核心能力一句话——**推理性能工程**(continuous batching、paged KV cache、调度、量化、投机解码、分布式 TP/PP/EP)。

> ⚠️ 两个 "Triton" 别混:本 repo 用的是 **NVIDIA Triton *Inference Server***(服务框架);算子开发常说的是
> **OpenAI Triton**(写 kernel 的 DSL)。`triton-*` / `*-fused-*` 里的 "fused" 指「把 pre/post 融进 server」
> (Python/BLS 层),**不是 kernel fusion**。

---

## 1. 引擎层 = AI infra 的落点(推荐方向)

### 1.1 不是从零转,是「往下沉半层」
已攒下、可**直接迁移**的东西:
- TRT-LLM engine build、`tensorrtllm` backend、KV-cache reuse、streaming detok —— 已经在**用**这层。
- 这套 **TTFT / E2E / 吞吐基准 + 瓶颈定位**(护栏单实例串行、BLS 实例数卡住 engine batch,见 `PERFORMANCE.md`)
  ——「在栈里找瓶颈」的思维**本身就是 infra 工作的核心**。
- 在 NVIDIA = **TRT-LLM / Dynamo / NIM 主场**。

真正要变的只有一点:**从「把引擎当黑盒调参」→「读懂引擎内部、能改能调能对比」。**

### 1.2 四个具体动作
1. **用 vLLM 源码学引擎内部**(开源、Python 可读):continuous batching / PagedAttention / scheduler / block manager。
   TRT-LLM 性能强但编译黑盒,适合「主场深耕」;vLLM 适合「看清原理」。两个互补,都要。
2. **补性能方法论**:memory-bound decode vs compute-bound prefill、roofline、goodput、batch-latency 权衡;
   工具 **Nsight Systems / Compute** + torch profiler。这是 infra 与「会调 API」的分水岭。
3. **盯住当下定义这领域的题目(2026),挑一个吃透:**
   - **Disaggregated prefill/decode**(NVIDIA **Dynamo** 即围绕它建,vLLM 也有 P/D)
   - prefix caching、chunked prefill、speculative decoding
   - **FP8 / FP4 量化**(Blackwell FP4)、大规模 **MoE** 服务、KV cache offload
   - 多卡 **TP / PP / EP** + NCCL
4. **算子那条线降级为「读得懂」即可**:infra 要「知道哪个 kernel 是瓶颈、会集成 FlashAttention / FlashInfer /
   CUTLASS」,但不必从零写。**引擎层是算子和服务之间的甜点区**——系统/性能味更重,不必沉到 CUTLASS 那么深。

### 1.3 最省力的切入:拿现有 repo 当桥
> **给现有 gateway 加一个 vLLM backend,和 TRT-LLM 并排,用 `client/perf_benchmark.py` 做 vLLM vs TRT-LLM 的 A/B。**

一个项目同时证明:① 服务层本来就会 ② 能上手 vLLM ③ 会做引擎级性能对比。之后再深挖一个特性
(prefix caching / chunked prefill / FP8)并把性能提升**量化**出来,方向即立。

### 1.4 要补的硬技能(诚实清单)
引擎源码阅读(vLLM 优先)· GPU profiling(Nsight)· 分布式推理(TP/PP/EP、NCCL)· 量化运行时(FP8/AWQ/GPTQ)。
算子只要「读 + 集成」,不要求「从零写」。

### 1.5 NVIDIA 内部 vs 通用
- **内部主场**:TensorRT-LLM、**Dynamo**(分布式/分离式服务)、NIM(模型微服务)。
- **通用 / 招聘市场**:vLLM、SGLang——也是**学引擎内部的最佳读物**。

---

## 2. 算子层地图(如果要再下沉)— TensorRT-LLM v0.14.0

### 2.1 一个算子从底层到「能在模型里调用」的 5 层链路
| 层 | 路径 | 干什么 | 文件量 |
|---|---|---|---|
| ① CUDA kernel | `cpp/tensorrt_llm/kernels/<类>/` | 真正的计算 | **1664** |
| ② TRT 插件 | `cpp/tensorrt_llm/plugins/<名>Plugin/` | 把 kernel 包成 TensorRT 层 | 64 |
| ③ 插件注册 | `cpp/tensorrt_llm/plugins/api/tllmPlugin.cpp` | 把 plugin creator 注册给 TRT | 1 |
| ④ Python 绑定 | `tensorrt_llm/functional.py` + `tensorrt_llm/plugin/plugin.py` | 用 `trt.PluginField` 构图 | — |
| ⑤ 模型里用 | `tensorrt_llm/layers/` → `tensorrt_llm/models/<model>/model.py` | 组装成模型 | — |

> **并行旁路** `cpp/tensorrt_llm/thop/`(14 个 op,如 `dynamicDecodeOp` / `weightOnlyQuantOp` / `fp8Op`):
> 要的是**直接从 PyTorch 调用**的自定义 op(TRT-LLM 的 PyTorch flow)就走这里,不走插件。

每个插件 = 一个目录三件套(骨架 `identityPlugin` 仅 198 行):`XxxPlugin.h / .cpp / CMakeLists.txt`;
`.cpp` 必须实现 `IPluginV2DynamicExt` 那套(`clone / getOutputDimensions / supportsFormatCombination /
configurePlugin / getWorkspaceSize / enqueue / serialize`),其中 **`enqueue` 才是真正 launch kernel 的地方**。

### 2.2 kernel 目录分类(重心一目了然)
| kernel 目录 | 是什么 | 性能权重 |
|---|---|---|
| `decoderMaskedMultiheadAttention` | decode 阶段 MMHA / XQA(逐 token 注意力) | 🔥 最热 |
| `contextFusedMultiHeadAttention` | prefill 阶段 FMHA | 🔥 |
| `cutlass_kernels` / `internal_cutlass_kernels` | CUTLASS GEMM(含量化 / MoE GEMM) | 🔥 |
| `weightOnlyBatchedGemv` | weight-only 量化 GEMV(decode 小批 matvec) | 高 |
| `mixtureOfExperts` | MoE | 高 |
| `selectiveScan` | Mamba / SSM selective scan | 中 |
| `lora` / `beamSearchKernels` / `speculativeDecoding` / `unfusedAttentionKernels` | LoRA / beam / 投机解码 / RoPE·mask 等零件 | 中 |

插件侧约 25 个,基本和 kernel 一一对应(`gptAttentionPlugin` 是核心 paged-KV 注意力;
`*QuantGemmPlugin` / `quantizePerToken` 是量化族;`nccl` 是多卡通信)。

### 2.3 从哪切入最合适(按目标分,带难度)
| 目标 | 切入点 | 难度 | 第一步 |
|---|---|---|---|
| 学机制 / 加一个自定义算子 | `plugins/identityPlugin/`(官方骨架)+ `cumsumLastDimPlugin`(简单真例) | ⭐⭐ | 复制 `identityPlugin` 改名 → `enqueue` 里 launch 自己的 kernel → `tllmPlugin.cpp` 注册 → `functional.py` 暴露,**把垂直切片打通一次** |
| 看完整新算子族怎么加 | `kernels/selectiveScan/` ↔ `selectiveScanPlugin`+`mambaConv1dPlugin`+`lruPlugin`(Mamba 就这么加的) | ⭐⭐⭐ | 当模板逐层读:kernel→plugin→python |
| 真做性能优化 | `decoderMaskedMultiheadAttention`(MMHA/XQA)、`cutlass_kernels`(GEMM) | ⭐⭐⭐⭐⭐ | **不建议作起点**——全仓优化最重、CUTLASS 最深,先用前两条热身 |

配套文档:`docs/source/architecture/add-model.md`(顶层组装)、`docs/source/installation/build-from-source-linux.md`(怎么编)。

---

## 3. 和本项目的关系

- **算子 / 引擎开发与 serving 解耦**:外层 `text_pipeline_bls` 只吃**编译好的 engine**。在 TRT-LLM 里改完
  kernel/plugin → 重编引擎(`scripts/build_engines.sh` 调的就是 TRT-LLM)→ serving 层不改就能跑新引擎。
- **本 repo 可当端到端测试台**:新算子 / 新引擎重编后,用 `client/perf_benchmark.py` 直接量 TTFT / 吞吐对比,
  收益立刻可见。
- **构建现实**:从源码编 TRT-LLM 很重(CUDA toolkit + CUTLASS,docker build,编译以小时计);`.venv` 那套轻量
  环境只够跑 serving,做算子 / 引擎得进 TRT-LLM 的构建环境。

---

## 4. 下一步(可选)

> **执行手册:** 这篇是「方向/为什么」;落到「在这台 L4 上怎么动手做」见 [`L2-LAB.md`](./L2-LAB-CN.md)
> —— 把本 repo 的旋钮(KV reuse / chunked / scheduler / 量化)变成一台 ablate-on-your-own-rig 实验台,
> M0 已用真实启动日志闭环(手算 KV 预测引擎分配到 3 位有效数字),lab notebook 在 [`lab-notebook/`](./lab-notebook/)。

- **桥项目分阶段**:加 vLLM backend → vLLM vs TRT-LLM A/B(复用 `perf_benchmark.py`)→ 深挖一个特性
  (prefix caching / chunked prefill / FP8)并量化收益。
- **vLLM 源码精读路线**:`LLMEngine` → scheduler(continuous batching)→ block manager(paged KV)→ PagedAttention kernel。

> 一句话总结:**已站在服务层,往下半层到引擎层就是 AI infra,核心是推理性能工程;vLLM 学原理、
> TRT-LLM/Dynamo 在 NVIDIA 主场深耕;用现有 repo 加 vLLM backend 做 A/B 是最省力的切入。**
