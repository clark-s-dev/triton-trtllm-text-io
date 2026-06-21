.PHONY: help check test setup engines server obs-up obs-down client
help:  ## show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

check:  ## L4 readiness gate (run first)
	python3 scripts/check_env.py

test:  ## run the GPU-free unit tests (streaming detok + stop + finish_reason)
	python3 tests/test_detokenize_incremental.py && python3 tests/test_stop.py && python3 tests/test_finish.py

setup:  ## install host deps + download Qwen2.5 models
	bash scripts/setup.sh

engines:  ## build the small + large TRT-LLM engines (FP16, KV-reuse)
	bash scripts/build_engines.sh

server:  ## build the image and launch Triton
	bash scripts/start_server.sh

obs-up:  ## start Prometheus + Grafana + OTel + DCGM
	cd observability && docker compose -f docker-compose.observability.yml up -d

obs-down:  ## stop the observability stack
	cd observability && docker compose -f docker-compose.observability.yml down

client:  ## quick streaming smoke test (override M="...")
	python3 client/client_fused.py --message "$(or $(M),用三句话介绍 GPU 推理 🚀)"
