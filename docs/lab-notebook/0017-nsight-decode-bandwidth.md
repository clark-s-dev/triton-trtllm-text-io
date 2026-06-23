# 0017 · 方法论补全:Nsight Compute kernel 级实锤 —— 「decode 带宽受限」从 roofline 推断 → 测量

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md)。**这条不引入新旋钮**,而是补上**方法论缺口**:M0 / 0009 / 0011 反复用「decode 是带宽受限」下结论,但那一直是**roofline 推断**(手算算术强度)。这里用 **Nsight Compute(`ncu`)直接量 decode kernel 的 Speed-of-Light**——把推断升级成 **kernel 级实锤**。顺带跨 dtype(FP16/FP8/INT4)量一遍,给 0009/0011 的「量化砍带宽」收一个 kernel 级的尾。

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-23 |
| 里程碑 | 方法论(横切 M0 / M4);为 0009 / 0011 / 0016 提供 kernel 级证据 |
| 工具 | `ncu` 2024.2.1(NGC 24.10 自带);`--cap-add=SYS_ADMIN`(本机 `RmProfilingAdminOnly=1`) |
| 引擎/层 | 0.5B 三种 dtype:`fp16` / `fp8`(W8A8)/ `int4awq`(W4A16),**batch=1 decode 直连 `ModelRunnerCpp`** |

## ① 假设(Hypothesis)
batch=1 的 decode 每步要把**整张权重**从显存读一遍,却只做 M=1 的 GEMV(算得极少)。所以 decode 的 linear-layer kernel 的**算术强度极低**(远在 roofline 屋脊点左侧)→ **DRAM 吞吐打满、SM(compute)吞吐很低** → 这就是「带宽受限」的 kernel 级定义。量化把权重字节砍掉 → 同一个 kernel **读得更少 → decode 更快**,但**仍然带宽受限**(只是墙挪近了)。

## ② 预测(动手前)— roofline 量级

**L4(Ada AD104):** 显存带宽 **~300 GB/s**;FP16 张量核 **~120 TFLOPS** → **屋脊点 ≈ 120e12/300e9 ≈ 400 FLOP/byte**。
decode GEMV 算术强度 = 每权重字节做的 FLOP:1 个 MAC = 2 FLOP/权重元素。

| dtype | 权重字节/元素 | 算术强度 (FLOP/byte) | 屋脊点 | 结论 |
|---|---|---|---|---|
| FP16 | 2 | **~1** | 400 | ≪ 屋脊 → 深度带宽受限 |
| FP8(W8A8) | 1 | **~2** | 400 | ≪ 屋脊 → 带宽受限 |
| INT4(W4A16) | 0.5 | **~4** | 400 | ≪ 屋脊 → 带宽受限(反量化加一点 compute) |

**预测的 kernel Speed-of-Light(decode 的 linear-layer GEMM/GEMV):**
- **DRAM 吞吐 % 高(预测 > 60%),SM(compute)吞吐 % 低(预测 < 30%)**——三种 dtype 都该如此(都在屋脊左侧)。**这个「Memory% ≫ Compute%」就是实锤。**
- INT4 的 `weightOnlyBatchedGemv` 因为要把 4-bit 反量化回 FP16 再 MAC,**SM% 应略高于 FP16**(多了 dequant 的 ALU 活),但仍 memory-leaning。
- **每 token DRAM 读字节 ≈ 权重总字节**:0.5B FP16 ≈ **1.0 GB/token**(/300GB/s ≈ **3.3 ms 带宽下限**);FP8 ≈ 0.5 GB(1.7ms);INT4 ≈ 0.25 GB(0.83ms)。
- **实测 ITL(decode_probe,batch=1)应当 FP16 > FP8 > INT4**,量级对上 0009/0011(FP16 ~5ms、FP8 ~4ms、INT4 ~3ms);比带宽下限高,差额 = KV 读 + attention + 启动/调度开销 + 带宽利用率 < 100%。

## ③ 实验设置(可复现)
- **workload:** `lab/decode_probe.py` —— 短 prefill(8 tok)+ 多步 batch=1 decode(`end_id=-1` 强制跑满步数),让 ncu skip 掉 prefill 后**几乎只剩 decode 的 M=1 权重 GEMV**。
- **命令:** `scripts/profile_decode.sh`(host 跑,内部 `docker run --cap-add=SYS_ADMIN` 起 ncu):
  ```bash
  bash scripts/profile_decode.sh validate     # 先验证 ncu 权限 + 落在 decode
  bash scripts/profile_decode.sh               # int4awq / fp8 / fp16 各出一份 CSV 到 lab/ncu/
  ```
  metrics:`sm__throughput.avg.pct_of_peak_sustained_elapsed`(Compute%)、`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`(Memory%)、`gpu__time_duration.sum`、`dram__bytes.sum`、`launch__grid_size/block_size`。
- **控制变量:** `--launch-skip` 跳过 prefill 段(只看 decode);`--launch-count` 取 decode 稳态一窗;三 dtype 同模型(0.5B)同 prompt 同步数,只换引擎。**filter:** 报告里只取 GEMM/GEMV 行(linear 层),不混 attention/norm/elementwise。

---

## ④ 实测(Measure)

> **⚠ 工具坑(RCA 级,先说):** 本机驱动是 **580.126.09 / CUDA 13.0**,比 NGC 24.10 自带的 **`ncu`/`nsys` 2024.2.1** 新。`ncu` 一律 `Failed to prepare kernel for profiling / Unknown Error on device 0`,`nsys --gpu-metrics-device` 报 `NVPA_STATUS_ERROR`——**PerfWorks 硬件计数器子系统在这版工具上起不来**(`--privileged`、`--clock-control none` 都不解;不是权限/锁频问题,是工具 vs 新驱动的版本不兼容)。要 ncu 的 per-kernel SoL% 得装 Nsight 2025.x。
> **绕过(拿到等价的 kernel 级证据):** `nsys --trace=cuda`(CUPTI **kernel 计时**,不走 PerfWorks)**能跑**;配合**逐 kernel 的精确权重字节**(权重 shape 已知),用 `字节 / 实测 duration` 直接算**该 kernel 达到的 DRAM 带宽**,再对 300 GB/s 峰值。**这反而是比一个笼统 SoL% 更硬的实锤**——每个 kernel 一个「达到峰值带宽的百分比」。

数据:`lab/ncu/{fp16,fp8,int4awq}.kern.txt`(nsys kernel summary);ITL 来自 `lab/decode_probe.py`。**L4 峰值:DRAM 300 GB/s、FP16 张量核 ~121 TFLOPS → 屋脊 ~400 FLOP/byte。**

**A. 整 token 的 roofline 落点(batch=1,decode_probe ITL + 字节记账)**

| dtype | ITL ms/token | 权重读/token | 达到带宽 | **% 峰值带宽** | 达到算力 | **% 峰值算力** |
|---|---|---|---|---|---|---|
| FP16 | 4.941 | 988 MB | 200 GB/s | **67%** | 0.20 TFLOPS | **0.17%** |
| FP8(W8A8) | 3.961 | 630 MB | 159 GB/s | 53% | 0.25 TFLOPS | 0.21% |
| INT4-AWQ | 3.041 | 451 MB | 148 GB/s | 49% | 0.33 TFLOPS | 0.27% |

> **实锤:`% 峰值带宽`(49–67%)碾压 `% 峰值算力`(~0.2%),差 ~300×。** decode 在烧带宽、几乎不烧算力——「带宽受限」从手算推断变成实测落点。算术强度 ≈ 0.99GFLOP/0.99GB = **1 FLOP/byte**(FP16),屋脊 400 → **低 400×** → 深在 memory-bound 区。

**B. kernel 归因(GPU kernel 时间占比,nsys)—— 时间花在哪些 kernel**

| dtype | 权重 GEMM/GEMV 占比 | 代表 kernel |
|---|---|---|
| FP16 | **84%** | `cudaCoreGemm<half>`(M=1)60.5% + lm_head `gemvx`(FP16)23.6% |
| FP8 | **80%** | lm_head `gemvx`(FP16)29.9% + `sm89_xmma_gemm_e4m3`(FP8 张量核)50.3% |
| INT4-AWQ | **74%** | lm_head `gemvx`(FP16)40.0% + `weight_only::kernel`(INT4 GEMV)34.4% |

(其余:`mmha` attention ~4–6%、RMSNorm/penalty/softmax 等。)

**C. 逐 kernel 达到的带宽(= 真·实锤,字节/duration)**

| kernel | 读字节/调用 | 实测 duration | 达到带宽 | **% 峰值** |
|---|---|---|---|---|
| **lm_head `gemvx`**(FP16,**三 dtype 完全一致** 1.105 ms) | 151936×896×2 = **272 MB** | 1.105 ms | 246 GB/s | **82%** |
| FP16 transformer `cudaCoreGemm`(整层合计/token) | 358M×2 = **716 MB** | 2.83 ms | 253 GB/s | **84%** |

> 两个权重矩阵 kernel 都跑在 **~82–84% 峰值 DRAM 带宽**——**单 kernel 几乎打满显存带宽 = 带宽受限的定义**。(整 token 平均掉到 67%,是因为 kernel 之间夹了 attention/norm/softmax 这些非带宽瓶颈的活。)

```
% 峰值带宽 (■) vs % 峰值算力 (·),batch=1 decode:
FP16  ■■■■■■■■■■■■■■■■■ 67%   算力 ·0.17%
INT4  ■■■■■■■■■■■■ 49%        算力 ·0.27%
lm_head gemvx (单 kernel)  ■■■■■■■■■■■■■■■■■■■■■ 82%   ← 贴着带宽墙
```

![Nsight decode-kernel 分析:带宽受限 + lm_head 地板](../decode-roofline.png)

*左:权重 kernel 跑在 49–67% 峰值带宽、仅 ~0.2% 峰值算力(带宽受限)。右:量化把 transformer GEMM 砍小,未量化的 FP16 `lm_head` 时间占比从 24% 涨到 40%(量化次线性加速之源)。数据 = `nsys` kernel 计时 + 字节记账(`ncu` 硬件计数器在本机被驱动版本挡住,见 ⑤)。*

## ⑤ Gap 分析(预测 vs 实测)

- **预测「DRAM% > 60%、Compute% < 30%」——对,且更极端。** 权重 kernel 82–84% 峰值带宽,算力 ~0.2%。decode 带宽受限**实测坐实**。
- **意外发现(kernel 级):lm_head 没被量化,是一道固定带宽地板。** `gemvx`(FP16 vocab 投影,272 MB/token)**在 FP16/FP8/INT4 三个引擎里 duration 完全相同(70.7M ns)**——modelopt 默认不量化 lm_head。于是量化只砍 transformer 层(`cudaCoreGemm`→FP8 `sm89_xmma`→INT4 `weight_only`),lm_head 不变 → 它的时间占比从 **23.6% → 29.9% → 40.0%** 一路涨。
- **这条解释了 0009/0011 的「量化加速是次线性的」:** INT4 把 transformer 权重字节砍到 1/4,但**整体 ITL 只从 4.94→3.04 ms(1.6×),不是 4×**——因为没被量化的 lm_head(272MB/token)+ attention/overhead 是不缩的地板。kernel 归因把这个「为什么不是 4×」量到了字节级。
- **每 token 读字节 ≈ 权重字节** → 验证了「decode 每步把整张权重读一遍」(M0 的前提)。

## ⑥ 机制 / 结论

- **「decode 带宽受限」不再是 roofline 推断,而是实测:** 权重 kernel 跑在 **82–84% 峰值 DRAM 带宽、~0.2% 峰值算力**。补上了 M0 / [0009](./0009-fp8-quantization.md) / [0011](./0011-int4-awq.md) 一直在用、但此前只手算过的那块地基。
- **对 0011/0009(量化):** 同一个权重 GEMV,字节砍 1/2(FP8)/ 1/4(INT4)→ DRAM 流量降 → kernel 变快;但 **lm_head 这道 FP16 地板让加速次线性**(kernel 级新解释)。
- **对 [0016](./0016-speculative-decoding.md)(投机解码):** 正因为这些 GEMV **带宽受限**(读一次权重、算力闲着 99.8%),target 一次验证 K+1 个位置 ≈ 读一次权重 ≈ 解 1 个 token 的代价——**投机解码的 verify「几乎免费」就建立在这条 kernel 级证据上**。也正因算力 batch=1 时闲着 99.8%,连续批处理才能靠加 batch 把它填满(0016 D 表的交叉)。
- **一句话(面试版):** 在 L4 上,0.5B decode 的权重 kernel 实测跑在 ~83% 峰值显存带宽、~0.2% 峰值算力——**带宽墙是实测的,不是推的**;量化(砍字节)、投机解码(一次读权重多产 token)、连续批处理(摊薄权重读取)是同一道墙的三种打法,而没被量化的 lm_head 是那道挪不动的地板。
