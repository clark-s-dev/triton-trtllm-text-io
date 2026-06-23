#!/usr/bin/env bash
# Reproduces the notebook 0016 (speculative decoding) measurement matrix. Run INSIDE the NGC
# trtllm container (it imports tensorrt_llm), with the repo mounted at /work:
#   docker run --rm --gpus all -v "$PWD:/work" -w /work \
#     nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3 bash lab/run_0016_sweep.sh
# Emits one JSON line per config to lab/results_0016.jsonl (parsed into the 0016 tables).
# Baseline uses the PLAIN fp16 1.5B engine: a draft_tokens_external target only emits 1 token per
# call without drafts, so the honest "no spec decode" baseline is the normal engine (same weights).
set -uo pipefail
OUT=${OUT:-lab/results_0016.jsonl}
TOK=hf_models/Qwen2.5-1.5B-Instruct
SPEC="--draft-engine engines/qwen2.5-0.5b-draft --target-engine engines/qwen2.5-1.5b-target"
BASE="--target-engine engines/qwen2.5-1.5b-fp16"
OL=${OL:-200}; REP=${REP:-2}
: > "$OUT"
run(){ echo ">>> $*"; python3 lab/specdec_bench.py "$@" 2>/dev/null | grep '^JSON' >> "$OUT" || echo "  (run failed)"; }

# A — draft_len K sweep (greedy, mixed, logits acceptance) + baseline
run --mode baseline $BASE --tokenizer $TOK --output-len $OL --prompts mixed --temperature 0 --repeats $REP --label A-base
for K in 1 2 4 6 8; do
  run --mode specdec $SPEC --tokenizer $TOK --draft-len $K --accept logits --output-len $OL --prompts mixed --temperature 0 --repeats $REP --label A-K$K
done

# B — acceptance is a workload property: easy vs hard (K=4, greedy, logits)
for P in easy hard; do
  run --mode baseline $BASE --tokenizer $TOK --output-len $OL --prompts $P --temperature 0 --repeats $REP --label B-base-$P
  run --mode specdec $SPEC --tokenizer $TOK --draft-len 4 --accept logits --output-len $OL --prompts $P --temperature 0 --repeats $REP --label B-spec-$P
done

# C — acceptance method (K=4, mixed): greedy-tokens, sampling-logits, sampling-tokens (+ sampling baseline)
run --mode specdec $SPEC --tokenizer $TOK --draft-len 4 --accept tokens --output-len $OL --prompts mixed --temperature 0   --repeats $REP --label C-greedy-tokens
run --mode baseline $BASE --tokenizer $TOK --output-len $OL --prompts mixed --temperature 0.8 --repeats $REP --label C-base-sample
run --mode specdec $SPEC --tokenizer $TOK --draft-len 4 --accept logits --output-len $OL --prompts mixed --temperature 0.8 --repeats $REP --label C-sample-logits
run --mode specdec $SPEC --tokenizer $TOK --draft-len 4 --accept tokens --output-len $OL --prompts mixed --temperature 0.8 --repeats $REP --label C-sample-tokens

# D — batch crossover (K=4, greedy, mixed): spec-decode is a low-batch win; it fights continuous batching
for B in 2 4 8; do
  run --mode baseline $BASE --tokenizer $TOK --output-len $OL --prompts mixed --temperature 0 --batch $B --repeats $REP --label D-base-b$B
  run --mode specdec $SPEC --tokenizer $TOK --draft-len 4 --accept logits --output-len $OL --prompts mixed --temperature 0 --batch $B --repeats $REP --label D-spec-b$B
done

echo "ALL_DONE -> $OUT"
