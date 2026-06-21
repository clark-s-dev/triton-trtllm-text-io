#!/usr/bin/env bash
# Build the custom Triton image (NGC TRT-LLM + our Python-backend deps + src) and
# launch the server with metrics + OpenTelemetry tracing on.
#   HTTP 8000 / gRPC 8001 / metrics 8002
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-triton-trtllm-text-io:local}
TRITON_TAG=${TRITON_TAG:-24.10-trtllm-python-py3}

echo "==> building $IMAGE"
docker build -t "$IMAGE" --build-arg TRITON_TAG="$TRITON_TAG" .

echo "==> launching Triton (Ctrl-C to stop)"
docker run --rm --gpus all --network host \
  -v "$PWD/model_repository:/models" \
  -v "$PWD/src:/workspace/src" \
  -v "$PWD/engines:/engines" \
  -v "$PWD/hf_models:/hf_models" \
  "$IMAGE" \
  tritonserver --model-repository=/models \
    --allow-metrics=true --metrics-config summary_latencies=true \
    --trace-config mode=opentelemetry \
    --trace-config opentelemetry,url=http://localhost:4318/v1/traces \
    --trace-config opentelemetry,resource=service.name=triton-trtllm-text-io \
    --trace-config rate=1 --trace-config level=TIMESTAMPS
