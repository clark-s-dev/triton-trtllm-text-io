#!/usr/bin/env bash
# Build the two TRT-LLM engines (small 0.5B + large 1.5B), FP16, with paged-context
# FMHA so KV-cache *prefix reuse* (II.2) is available at serve time.
#
# Engine building is version-specific. This runs inside the NGC TRT-LLM container
# and uses the matching TensorRT-LLM example converter. If your pinned tag differs,
# adjust TRITON_TAG / TRTLLM_REF and verify the example path + trtllm-build flags.
set -euo pipefail
cd "$(dirname "$0")/.."

TRITON_TAG=${TRITON_TAG:-24.10-trtllm-python-py3}
TRTLLM_REF=${TRTLLM_REF:-v0.13.0}          # match the TensorRT-LLM in the image
IMAGE="nvcr.io/nvidia/tritonserver:${TRITON_TAG}"

# Examples (convert_checkpoint.py) ship in the TensorRT-LLM repo, not the image.
if [ ! -d "TensorRT-LLM" ]; then
  echo "==> cloning TensorRT-LLM examples @ ${TRTLLM_REF}"
  git clone --depth 1 -b "${TRTLLM_REF}" https://github.com/NVIDIA/TensorRT-LLM.git
fi

docker run --rm --gpus all -v "$PWD:/work" -w /work "$IMAGE" bash -lc '
  set -euo pipefail
  EX=TensorRT-LLM/examples/qwen
  build() {  # <tag> <hf_dir>
    python3 "$EX/convert_checkpoint.py" --model_dir "hf_models/$2" \
        --output_dir "ckpt/$1" --dtype float16
    trtllm-build --checkpoint_dir "ckpt/$1" --output_dir "engines/qwen2.5-$1-fp16" \
        --gemm_plugin float16 --max_batch_size 64 \
        --use_paged_context_fmha enable --paged_kv_cache enable \
        --max_input_len 4096 --max_seq_len 8192
  }
  build 0.5b Qwen2.5-0.5B-Instruct
  build 1.5b Qwen2.5-1.5B-Instruct
'
echo "==> engines in ./engines/. Next:  bash scripts/start_server.sh"
