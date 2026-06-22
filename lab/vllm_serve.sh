#!/usr/bin/env bash
# vllm_serve.sh — launch vLLM as the A/B counterpart to the TRT-LLM engine (L2-LAB M1).
#
# The point of M1 is a *same-machine, same-model, same-workload* comparison: stand vLLM
# up on the EXACT Qwen2.5 HF weights this repo already serves through TensorRT-LLM, then
# drive both through one methodology (lab/ab_vllm_vs_trtllm.py + client/perf_benchmark.py)
# and compare TTFT / ITL / throughput / P99.
#
# IMPORTANT — they cannot run at once on one L4: Triton (8000-8002) and the engines' KV
# pool already occupy the GPU (see docs/REPORT.md §5). Stop Triton first:
#     docker stop triton-llm
# then run vLLM here, measure, then `docker start triton-llm` and measure the TRT-LLM side.
#
# Fairness controls (so the A/B is honest — these MIRROR the engine's config):
#   * same weights .............. hf_models/Qwen2.5-{0.5B,1.5B}-Instruct (already in repo)
#   * --dtype float16 ........... the engines are FP16 (not Qwen's default bf16)
#   * --max-num-seqs 64 ......... = build-time max_batch_size=64 (lab-notebook 0008)
#   * --enable-prefix-caching ... = enable_kv_cache_reuse=true (lab-notebook 0004)
#   * --gpu-memory-utilization .. picked so vLLM's KV pool ~ the engine's pool; NOTE the
#                                 semantics differ (vLLM util = weights+KV of the WHOLE
#                                 GPU; TRT-LLM fraction = of FREE mem after weights) — see
#                                 docs/lab-notebook/0015 §method for the apples-to-apples note.
#   * --max-model-len 8192 ...... bound context the same for both
#
# Requires vLLM in a venv/container with a CUDA build. This script does NOT install it
# (heavy, GPU-specific); see the comment block at the bottom for one way.
set -euo pipefail

MODEL="${MODEL:-small}"                 # small | large
PORT="${PORT:-8003}"                    # avoid Triton's 8000-8002
GPU_UTIL="${GPU_UTIL:-0.30}"            # ~matches small engine's modest KV footprint; raise for large
# PYTHON: interpreter with vLLM installed (this repo has no system pip; use a venv):
#   PYTHON=.venv-vllm/bin/python bash lab/vllm_serve.sh

case "$MODEL" in
  small) WEIGHTS="hf_models/Qwen2.5-0.5B-Instruct" ;;
  large) WEIGHTS="hf_models/Qwen2.5-1.5B-Instruct" ;;
  *) echo "MODEL must be 'small' or 'large'"; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS_PATH="$ROOT/$WEIGHTS"
[ -d "$WEIGHTS_PATH" ] || { echo "weights not found: $WEIGHTS_PATH (run scripts/setup.sh)"; exit 1; }

echo ">> vLLM serving $MODEL ($WEIGHTS) on :$PORT  [FP16, max-num-seqs=64, prefix-caching on]"
echo ">> served-model-name = tensorrt_llm_${MODEL}  (so the A/B client can use one --model)"

exec "${PYTHON:-python3}" -m vllm.entrypoints.openai.api_server \
  --model "$WEIGHTS_PATH" \
  --served-model-name "tensorrt_llm_${MODEL}" \
  --port "$PORT" \
  --dtype float16 \
  --max-num-seqs 64 \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --gpu-memory-utilization "$GPU_UTIL" \
  --no-enable-log-requests

# ---------------------------------------------------------------------------
# Installing vLLM (one option — a CUDA venv next to this repo; do NOT reuse .venv,
# which holds the lightweight client deps):
#     python3 -m venv .venv-vllm && . .venv-vllm/bin/activate && pip install vllm
#   GOTCHA: the venv route needs the Python dev headers — vLLM's Triton/inductor JIT
#   compiles a launcher that #includes <Python.h>; without `python3-dev` it dies with
#   "fatal error: Python.h: No such file or directory" during engine init.
# So on a host where you can't `apt-get install python3-dev` (no sudo), use the official
# image instead — it's self-contained (this is what notebook 0015's run actually used):
#     docker run -d --name vllm-ab --gpus all -p 8003:8003 -v "$PWD/hf_models:/models:ro" \
#       vllm/vllm-openai:latest \
#       --model /models/Qwen2.5-0.5B-Instruct --served-model-name tensorrt_llm_small \
#       --host 0.0.0.0 --port 8003 --dtype float16 --max-num-seqs 64 \
#       --max-model-len 8192 --enable-prefix-caching --gpu-memory-utilization 0.30
# ---------------------------------------------------------------------------
