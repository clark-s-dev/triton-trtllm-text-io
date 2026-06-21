# Custom Triton image: NGC TRT-LLM base + the Python-backend deps (HF tokenizer,
# guard classifiers) + our shared src/ (the incremental detok / stop logic the BLS
# imports). torch/tensorrt_llm are already in the base image.
ARG TRITON_TAG=24.10-trtllm-python-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}

COPY requirements-server.txt /tmp/requirements-server.txt
RUN pip install --no-cache-dir -r /tmp/requirements-server.txt

# Single source of truth for the streaming pre/post math, importable by the BLS.
COPY src /workspace/src
ENV TEXT_IO_SRC=/workspace/src

# Optional: pre-bake the guard models so the first request is fast / offline-capable.
# RUN python3 -c "from transformers import pipeline; \
#   pipeline('text-classification', model='protectai/deberta-v3-base-prompt-injection-v2'); \
#   pipeline('text-classification', model='unitary/toxic-bert')"
