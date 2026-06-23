#!/usr/bin/env bash
# Nsight Compute profiling of batch=1 DECODE (notebook 0017): per-kernel Speed-of-Light proving
# decode is DRAM-bandwidth-bound. Handles the TWO env-specific gotchas this box has (RCA in 0017 ⑤):
#
#   1. The ncu in the NGC 24.10 image (2024.2.1) is too OLD for this box's CUDA-13 / 580 driver
#      (PerfWorks won't init -> "Unknown Error"). Fix: download ncu 2025.3.1 (the CUDA-13 build) from
#      NVIDIA's public CUDA repo and `dpkg-deb -x` it (no root needed -- sudo/apt are gated here),
#      then run that binary inside the NGC container (which has the TRT-LLM workload).
#   2. The DCGM exporter holds the GPU profiling counters (single-owner). Fix: stop it for the run,
#      restart on exit (trap). NOTE: this briefly blanks the Grafana GPU panels.
#
# Caveat: under ncu kernel-replay, TRT-LLM's inflight-batch manager can trip ("Unable to get batch
# slot") and a run may capture only partially -- enough for the dominant kernels (lm_head, the GEMMs).
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=${IMG:-nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3}
NCU_VER=${NCU_VER:-2025.3.1}
NCU_DEB=${NCU_DEB:-nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb}
NCU_CACHE=${NCU_CACHE:-/tmp/ncu-$NCU_VER}
NCU_DIR="$NCU_CACHE/opt/nvidia/nsight-compute/$NCU_VER"
DCGM=${DCGM:-observability-dcgm-exporter-1}
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum"

# 1. fetch a driver-compatible ncu (once, cached in /tmp)
if [ ! -x "$NCU_DIR/ncu" ]; then
  echo "==> downloading ncu $NCU_VER (CUDA-13 compatible) from the public CUDA repo"
  curl -fSL -o "/tmp/$NCU_DEB" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$NCU_DEB"
  rm -rf "$NCU_CACHE"; mkdir -p "$NCU_CACHE"; dpkg-deb -x "/tmp/$NCU_DEB" "$NCU_CACHE"
fi
echo "==> ncu: $("$NCU_DIR/ncu" --version | grep -i version | head -1)"

# 2. free the profiling counters from DCGM; ALWAYS restart it on exit
DCGM_UP=$(docker ps --filter "name=$DCGM" --format '{{.Names}}' | grep -c . || true)
cleanup(){ if [ "$DCGM_UP" = "1" ]; then echo "==> restarting $DCGM"; docker start "$DCGM" >/dev/null || true; fi; }
trap cleanup EXIT
if [ "$DCGM_UP" = "1" ]; then echo "==> stopping $DCGM (frees profiling counters; restarts on exit)"; docker stop "$DCGM" >/dev/null; fi

mkdir -p lab/ncu
prof() {  # <label> <engine_dir> <tokenizer_dir>
  echo "==> ncu SoL: $1 ($2)"
  docker run --rm --gpus all --cap-add SYS_ADMIN -v "$NCU_CACHE:/ncuroot" -v "$PWD:/work" -w /work "$IMG" bash -lc "
    /ncuroot/opt/nvidia/nsight-compute/$NCU_VER/ncu --metrics $METRICS \
      --launch-skip 300 -k 'regex:gemm|gemv|weight_only' --kernel-name-base demangled -c 200 --csv \
      python3 lab/decode_probe.py --engine '$2' --tokenizer '$3' --decode-steps 16 --warmup 0 --kv-frac 0.3 \
      > lab/ncu/sol_$1.csv 2> lab/ncu/sol_$1.log " || echo "  ($1 partial -- see lab/ncu/sol_$1.log)"
}

case "${1:-all}" in
  validate)
    docker run --rm --gpus all --cap-add SYS_ADMIN -v "$NCU_CACHE:/ncuroot" -v "$PWD:/work" -w /work "$IMG" bash -lc \
      "/ncuroot/opt/nvidia/nsight-compute/$NCU_VER/ncu --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
       --launch-skip 300 -k 'regex:gemm|gemv' -c 2 \
       python3 lab/decode_probe.py --engine engines/qwen2.5-0.5b-fp16 --tokenizer hf_models/Qwen2.5-0.5B-Instruct --decode-steps 8 --warmup 0 --kv-frac 0.3 2>&1 | tail -15" ;;
  *)
    prof fp16    engines/qwen2.5-0.5b-fp16    hf_models/Qwen2.5-0.5B-Instruct
    prof fp8     engines/qwen2.5-0.5b-fp8     hf_models/Qwen2.5-0.5B-Instruct
    prof int4awq engines/qwen2.5-0.5b-int4awq hf_models/Qwen2.5-0.5B-Instruct
    echo "==> done. Measured SoL summary: lab/ncu/sol_summary.md  (raw: lab/ncu/sol_*.csv)" ;;
esac
