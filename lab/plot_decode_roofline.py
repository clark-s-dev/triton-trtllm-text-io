#!/usr/bin/env python3
"""plot_decode_roofline.py — regenerate docs/decode-roofline.png for notebook 0017 / the report.

Left panel  = MEASURED Nsight Compute Speed-of-Light (ncu 2025.3.1; lab/ncu/sol_summary.md):
              DRAM (Memory) % vs SM (Compute) % per dominant decode kernel.
Right panel = GPU-kernel-time share (nsys cuda trace; lab/ncu/*.kern.txt): the FP16 lm_head floor.

    .venv/bin/python lab/plot_decode_roofline.py        # writes docs/decode-roofline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Panel A: measured ncu SoL (batch=1, Qwen2.5-0.5B) ---
KERNELS = ["lm_head GEMV\n(all dtypes)", "FP16 transf\nGEMM", "INT4 transf\nGEMV", "FP8 transf\nGEMM"]
DRAM = [96.6, 80.6, 42.3, 29.0]   # gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
SM   = [35.0, 10.2, 11.8, 4.7]    # sm__throughput.avg.pct_of_peak_sustained_elapsed

# --- Panel B: nsys GPU-kernel-time share, showing the un-quantized FP16 lm_head floor grow ---
DTYPES = ["FP16", "FP8", "INT4-AWQ"]
GEMM   = [60.5, 50.3, 34.4]
LMHEAD = [23.6, 29.9, 40.0]
ATTN   = [3.7, 4.8, 6.3]
OTHER  = [100 - g - l - a for g, l, a in zip(GEMM, LMHEAD, ATTN)]


def main(out="docs/decode-roofline.png"):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    x = np.arange(len(KERNELS)); w = 0.38
    axA.bar(x - w / 2, DRAM, w, label="DRAM / memory throughput", color="#2b8cbe")
    axA.bar(x + w / 2, SM, w, label="SM / compute throughput", color="#e34a33")
    for i, (d, s) in enumerate(zip(DRAM, SM)):
        axA.text(i - w / 2, d + 1.5, f"{d:.0f}%", ha="center", fontsize=9, fontweight="bold")
        axA.text(i + w / 2, s + 1.5, f"{s:.0f}%", ha="center", fontsize=9, color="#a63603")
    axA.set_xticks(x); axA.set_xticklabels(KERNELS, fontsize=8.5)
    axA.set_ylabel("% of peak (Nsight Compute SoL)"); axA.set_ylim(0, 105)
    axA.axhline(100, ls=":", c="gray", lw=0.8)
    axA.set_title("Measured kernel Speed-of-Light (ncu 2025.3.1, batch=1)\n"
                  "dominant lm_head GEMV = 97% DRAM (bandwidth-bound); compute idle", fontsize=10)
    axA.legend(loc="upper right", fontsize=8.5); axA.grid(axis="y", alpha=0.3)

    xb = np.arange(len(DTYPES))
    axB.bar(xb, GEMM, label="transformer GEMM (quantized)", color="#74a9cf")
    axB.bar(xb, LMHEAD, bottom=GEMM, label="lm_head GEMV (stays FP16 — fixed floor)", color="#fdae6b")
    bot2 = [g + l for g, l in zip(GEMM, LMHEAD)]
    axB.bar(xb, ATTN, bottom=bot2, label="attention (mmha)", color="#a1d99b")
    bot3 = [b + a for b, a in zip(bot2, ATTN)]
    axB.bar(xb, OTHER, bottom=bot3, label="norm / sample / other", color="#d9d9d9")
    for i, l in enumerate(LMHEAD):
        axB.text(i, GEMM[i] + l / 2, f"{l}%", ha="center", va="center", fontsize=9, fontweight="bold")
    axB.set_xticks(xb); axB.set_xticklabels(DTYPES)
    axB.set_ylabel("% of GPU kernel time (nsys)"); axB.set_ylim(0, 100)
    axB.set_title("Where decode time goes\n(quantize transformer → un-quantized lm_head floor grows 24%→40%)", fontsize=10)
    axB.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), fontsize=8, ncol=1); axB.grid(axis="y", alpha=0.3)

    fig.suptitle("Nsight decode-kernel analysis — Qwen2.5-0.5B, NVIDIA L4 (notebook 0017)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
