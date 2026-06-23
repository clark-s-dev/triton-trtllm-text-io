.PHONY: help check test setup engines server obs-up obs-down client specdec-engines specdec profile-decode
help:  ## show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

check:  ## L4 readiness gate (run first)
	python3 scripts/check_env.py

test:  ## run the GPU-free unit tests (serving math + L2 lab toys)
	python3 tests/test_detokenize_incremental.py && python3 tests/test_stop.py && python3 tests/test_finish.py
	python3 tests/test_paged_kv.py && python3 tests/test_cbatch_sim.py && python3 tests/test_specdec_model.py

setup:  ## install host deps + download Qwen2.5 models
	bash scripts/setup.sh

engines:  ## build the small + large TRT-LLM engines (FP16, KV-reuse)
	bash scripts/build_engines.sh

server:  ## build the image and launch Triton
	bash scripts/start_server.sh

specdec-engines:  ## build draft(0.5B)+target(1.5B) spec-decode engines (M4, notebook 0016)
	bash scripts/build_specdec_engines.sh

specdec:  ## run the speculative-decoding sweep in-container (needs specdec-engines; GPU)
	docker run --rm --gpus all -v "$(PWD):/work" -w /work nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3 bash lab/run_0016_sweep.sh

profile-decode:  ## Nsight Compute: decode-kernel Speed-of-Light across FP16/FP8/INT4 (notebook 0017; GPU)
	bash scripts/profile_decode.sh

obs-up:  ## start Prometheus + Grafana + OTel + DCGM
	cd observability && docker compose -f docker-compose.observability.yml up -d

obs-down:  ## stop the observability stack
	cd observability && docker compose -f docker-compose.observability.yml down

client:  ## quick streaming smoke test (override M="...")
	python3 client/client_fused.py --message "$(or $(M),用三句话介绍 GPU 推理 🚀)"
