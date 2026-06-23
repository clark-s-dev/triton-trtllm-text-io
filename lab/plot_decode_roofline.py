#!/usr/bin/env python3
"""plot_decode_roofline.py — regenerate docs/decode-roofline.png for notebook 0017 / the report.

Pure plotting of the measured Nsight numbers (nsys kernel timing + weight-byte accounting; ncu HW
counters are blocked on this box by the CUDA-13 driver vs the 24.10 image's 2024.2.1 tools). Values
come from lab/ncu/*.kern.txt (kernel-time shares) and lab/decode_probe.py (ITL → achieved bandwidth).

    .venv/bin/python lab/plot_decode_roofline.py        # writes docs/decode-roofline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DTYPES = ["FP16", "FP8", "INT4-AWQ"]
# roofline placement (batch=1, Qwen2.5-0.5B): % of L4 peak (300 GB/s DRAM, 121 TFLOPS FP16)
BW_PCT  = [67, 53, 49]          # achieved memory bandwidth
CMP_PCT = [0.17, 0.21, 0.27]    # achieved compute
# GPU-kernel-time share (nsys): transformer GEMM (quantized) vs un-quantized FP16 lm_head vs attention
GEMM   = [60.5, 50.3, 34.4]
LMHEAD = [23.6, 29.9, 40.0]
ATTN   = [3.7, 4.8, 6.3]
OTHER  = [100 - g - l - a for g, l, a in zip(GEMM, LMHEAD, ATTN)]


def main(out="docs/decode-roofline.png"):
    x = np.arange(len(DTYPES))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    w = 0.38
    axA.bar(x - w / 2, BW_PCT, w, label="memory bandwidth", color="#2b8cbe")
    axA.bar(x + w / 2, CMP_PCT, w, label="compute (FLOPs)", color="#e34a33")
    for i, (b, c) in enumerate(zip(BW_PCT, CMP_PCT)):
        axA.text(i - w / 2, b + 1.5, f"{b}%", ha="center", fontsize=9, fontweight="bold")
        axA.text(i + w / 2, c + 1.5, f"{c}%", ha="center", fontsize=9, color="#a63603")
    axA.set_xticks(x); axA.set_xticklabels(DTYPES)
    axA.set_ylabel("% of L4 peak"); axA.set_ylim(0, 80)
    axA.set_title("Decode is bandwidth-bound\n(batch=1, weight kernels: ~50–67% BW vs ~0.2% compute)", fontsize=10)
    axA.legend(loc="upper right", fontsize=9); axA.grid(axis="y", alpha=0.3)

    axB.bar(x, GEMM, label="transformer GEMM (quantized)", color="#74a9cf")
    axB.bar(x, LMHEAD, bottom=GEMM, label="lm_head GEMV (stays FP16 — fixed floor)", color="#fdae6b")
    bot2 = [g + l for g, l in zip(GEMM, LMHEAD)]
    axB.bar(x, ATTN, bottom=bot2, label="attention (mmha)", color="#a1d99b")
    bot3 = [b + a for b, a in zip(bot2, ATTN)]
    axB.bar(x, OTHER, bottom=bot3, label="norm / sample / other", color="#d9d9d9")
    for i, l in enumerate(LMHEAD):
        axB.text(i, GEMM[i] + l / 2, f"{l}%", ha="center", va="center", fontsize=9, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(DTYPES)
    axB.set_ylabel("% of GPU kernel time"); axB.set_ylim(0, 100)
    axB.set_title("Where decode time goes\n(quantize the transformer → un-quantized lm_head floor grows 24%→40%)", fontsize=10)
    axB.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), fontsize=8, ncol=1); axB.grid(axis="y", alpha=0.3)

    fig.suptitle("Nsight decode-kernel analysis — Qwen2.5-0.5B, NVIDIA L4 (nsys timing + byte accounting; notebook 0017)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
