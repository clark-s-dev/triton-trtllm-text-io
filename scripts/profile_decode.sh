#!/usr/bin/env bash
# Nsight Compute profiling of the batch=1 DECODE phase (notebook 0017). Turns the M0/0009/0011
# roofline *inference* ("decode is DRAM-bandwidth-bound") into a measured kernel Speed-of-Light.
#
# Runs on the HOST; launches ncu (shipped in the NGC image at /usr/local/bin/ncu) INSIDE the
# container with --cap-add=SYS_ADMIN. This box has RmProfilingAdminOnly=1, so the profiler needs
# admin capabilities — SYS_ADMIN on a root container satisfies the NVIDIA perf-counter gate without
# touching host sudo.
#
# Strategy: decode_probe.py does a tiny prefill then many batch=1 (M=1) decode steps, so after we
# --launch-skip past prefill, almost every captured kernel is a decode-phase weight GEMM/GEMV — the
# exact kernel whose Speed-of-Light we want. We capture a broad window and filter to the matmuls in
# the report (no fragile kernel-name regex at capture time).
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=${IMG:-nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3}
SKIP=${SKIP:-450}; COUNT=${COUNT:-110}; STEPS=${STEPS:-24}
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,dram__bytes.sum,launch__grid_size,launch__block_size"
mkdir -p lab/ncu

prof() {  # <label> <engine_dir> <tokenizer_dir>
  echo "==> ncu profiling: $1  ($2)"
  docker run --rm --gpus all --cap-add=SYS_ADMIN -v "$PWD:/work" -w /work "$IMG" bash -lc "
    /usr/local/bin/ncu --target-processes all --csv \
      --metrics $METRICS \
      --launch-skip $SKIP --launch-count $COUNT \
      python3 lab/decode_probe.py --engine '$2' --tokenizer '$3' --decode-steps $STEPS --warmup 0 \
      > lab/ncu/$1.csv 2> lab/ncu/$1.log " || { echo "  (ncu run failed; see lab/ncu/$1.log)"; tail -20 "lab/ncu/$1.log"; }
}

case "${1:-all}" in
  validate)  # quick perms/sanity check: capture just 3 kernels
    docker run --rm --gpus all --cap-add=SYS_ADMIN -v "$PWD:/work" -w /work "$IMG" bash -lc \
      "/usr/local/bin/ncu --version | head -3; echo '--- tiny capture ---'; \
       /usr/local/bin/ncu --metrics gpu__time_duration.sum --launch-skip $SKIP --launch-count 3 \
       python3 lab/decode_probe.py --engine engines/qwen2.5-0.5b-fp16 --tokenizer hf_models/Qwen2.5-0.5B-Instruct --decode-steps 8 --warmup 0 2>&1 | tail -25" ;;
  *)
    prof int4awq engines/qwen2.5-0.5b-int4awq hf_models/Qwen2.5-0.5B-Instruct
    prof fp8     engines/qwen2.5-0.5b-fp8     hf_models/Qwen2.5-0.5B-Instruct
    prof fp16    engines/qwen2.5-0.5b-fp16    hf_models/Qwen2.5-0.5B-Instruct
    echo "==> CSVs in lab/ncu/  (parse the GEMM/GEMV rows: Memory% >> Compute% == bandwidth-bound)" ;;
esac
