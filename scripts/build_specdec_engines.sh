#!/usr/bin/env bash
# Build the draft + target engines for Draft-Target speculative decoding (notebook 0016, M4).
#   draft  = Qwen2.5-0.5B  (proposes K tokens / iteration)
#   target = Qwen2.5-1.5B  (verifies K+1 in one forward pass, accepts longest matching prefix)
#
# Same NGC container + Qwen tied-embedding workaround as build_engines.sh. The ONLY new bits vs
# the FP16 engines are the speculative-decoding build flags (see examples/draft_target_model/README.md):
#   * BOTH engines: --gather_generation_logits  (so we can accept by logits, the paper's method;
#                   without it only token-equality acceptance is possible)
#   * BOTH engines: --use_paged_context_fmha enable  (draft-target needs paged-context KV)
#   * TARGET only:  --speculative_decoding_mode draft_tokens_external --max_draft_len K_MAX
#                   (bakes the "accept external draft tokens" path into the target's engine graph)
# max_draft_len is the *ceiling*; at run time we sweep draft_len K <= K_MAX without rebuilding.
set -euo pipefail
cd "$(dirname "$0")/.."

TRITON_TAG=${TRITON_TAG:-24.10-trtllm-python-py3}
TRTLLM_REF=${TRTLLM_REF:-v0.14.0}
IMAGE="nvcr.io/nvidia/tritonserver:${TRITON_TAG}"
K_MAX=${K_MAX:-10}                 # max_draft_len baked into the target engine
MBS=${MBS:-8}                      # max_batch_size (lets us probe spec-decode under small batches)
MAX_INPUT=${MAX_INPUT:-2048}
MAX_SEQ=${MAX_SEQ:-3072}

if [ ! -d "TensorRT-LLM" ]; then
  echo "==> cloning TensorRT-LLM examples @ ${TRTLLM_REF}"
  git clone --depth 1 -b "${TRTLLM_REF}" https://github.com/NVIDIA/TensorRT-LLM.git
fi

docker run --rm --gpus all -v "$PWD:/work" -w /work "$IMAGE" bash -lc "
  set -euo pipefail
  # Same NGC 24.10 tensorrt_llm 0.14.0 bug as build_engines.sh: qwen/model.py calls
  # loader.check_share_embedding() but the installed method requires (config). That helper
  # remaps the tied lm_head -> vocab embedding, so the convert needs it called correctly.
  QWEN_MODEL=/usr/local/lib/python3.10/dist-packages/tensorrt_llm/models/qwen/model.py
  sed -i 's/loader.check_share_embedding()/loader.check_share_embedding(config)/g' \"\$QWEN_MODEL\"
  EX=TensorRT-LLM/examples/qwen

  # Qwen2.5-0.5B/1.5B tie word embeddings (no separate lm_head.weight) -> --use_embedding_sharing,
  # else the converter crashes loading a None lm_head (same as build_engines.sh).
  convert() {  # <ckpt_tag> <hf_dir>
    python3 \"\$EX/convert_checkpoint.py\" --model_dir \"hf_models/\$2\" \
        --output_dir \"ckpt/\$1\" --dtype float16 --use_embedding_sharing
  }
  convert draft-0.5b  Qwen2.5-0.5B-Instruct
  convert target-1.5b Qwen2.5-1.5B-Instruct

  # Draft engine: standard FP16 + generation logits + paged-context FMHA.
  trtllm-build --checkpoint_dir ckpt/draft-0.5b --output_dir engines/qwen2.5-0.5b-draft \
      --gemm_plugin float16 --use_paged_context_fmha enable --gather_generation_logits \
      --max_batch_size ${MBS} --max_input_len ${MAX_INPUT} --max_seq_len ${MAX_SEQ}

  # Target engine: same + the draft_tokens_external speculative path (max_draft_len = ceiling).
  trtllm-build --checkpoint_dir ckpt/target-1.5b --output_dir engines/qwen2.5-1.5b-target \
      --gemm_plugin float16 --use_paged_context_fmha enable --gather_generation_logits \
      --speculative_decoding_mode draft_tokens_external --max_draft_len ${K_MAX} \
      --max_batch_size ${MBS} --max_input_len ${MAX_INPUT} --max_seq_len ${MAX_SEQ}
"
echo "==> draft  : engines/qwen2.5-0.5b-draft"
echo "==> target : engines/qwen2.5-1.5b-target  (speculative_decoding_mode=draft_tokens_external, max_draft_len=${K_MAX})"
echo "==> next   : .venv/bin/python ... (run inside container) lab/specdec_bench.py  — see notebook 0016"
