# Measured Nsight Compute Speed-of-Light — decode (batch=1, Qwen2.5-0.5B, L4)

Captured with ncu 2025.3.1 (downloaded; the NGC-24.10 ncu 2024.2.1 is too old for the CUDA-13 driver)
after stopping the DCGM exporter (it held the profiling counters). Median over captured decode kernels.
Metrics: gpu__dram_throughput (Memory %), sm__throughput (Compute %), dram__bytes.sum, gpu__time_duration.sum.

| kernel | DRAM % (peak) | SM % (peak) | DRAM bytes/call | duration | achieved GB/s |
|---|---|---|---|---|---|
| lm_head GEMV (FP16; identical in all 3 engines) | 96.6 | 35 | 324 MB | 1.12 ms | 289 |
| FP16 transformer `cudaCoreGemm` | 80.6 | 10 | 9.96 MB | 40.8 us | 244 |
| FP8 transformer `sm89_xmma` GEMM | 29.0 | 4.7 | 1.29 MB | 15.6 us | 83 |
| INT4 transformer `weight_only` GEMV | 42.3 | 11.8 | 2.57 MB | 16.3 us | 158 |

Verdict: the dominant kernel (lm_head, ~1.12 ms — far longer than any other) is **bandwidth-saturated at
96.6% of peak** across every precision (the un-quantized FP16 floor). FP16 transformer GEMM is also
bandwidth-bound (80.6%). The *quantized* transformer GEMMs shrink so much at M=1 that they no longer
saturate DRAM (29–42%) — they go latency/occupancy-bound — another reason quant decode speedup is sublinear.
Raw CSVs: sol_{fp16,fp8,int4awq}.csv.
