#!/usr/bin/env bash
# Install host deps and download the (ungated, easy-access) models the router uses.
# Run on the L4 box after `python3 scripts/check_env.py` reports READY.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> installing host requirements"
python3 -m pip install -r requirements.txt

mkdir -p hf_models engines

echo "==> downloading Qwen2.5 instruct models (Apache-2.0, no gate) — ~4 GB"
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir hf_models/Qwen2.5-0.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir hf_models/Qwen2.5-1.5B-Instruct

echo "==> done. Next:  bash scripts/build_engines.sh   (build the TRT-LLM engines)"
