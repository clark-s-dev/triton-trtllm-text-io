# Observability stack (Prometheus + Grafana + OpenTelemetry + DCGM)

The standard cloud-native observability stack, wired to Triton's **native** telemetry.
Triton needs no exporter sidecar for metrics — it serves Prometheus on `:8002` — so this
stack just scrapes, traces, and visualizes.

```
Triton :8002 ─scrape→ Prometheus ─→ Grafana dashboards
DCGM-exporter :9400 ─scrape→ Prometheus   (GPU util / mem / power → tokens/s/W)
Triton OTLP traces ─→ OTel Collector ─→ Jaeger        (per-stage latency)
                                   └→ spanmetrics → Prometheus
```

## 1. Launch Triton with metrics + tracing enabled

Add these flags to `tritonserver` (or `scripts/start_server.sh`):

```
--allow-metrics=true \
--metrics-config summary_latencies=true \
--trace-config mode=opentelemetry \
--trace-config opentelemetry,url=http://localhost:4318/v1/traces \
--trace-config opentelemetry,resource=service.name=triton-trtllm-text-io \
--trace-config rate=1 --trace-config level=TIMESTAMPS
```

## 2. Bring up the stack (on the L4 box, next to Triton)

```bash
docker compose -f docker-compose.observability.yml up -d
```

| UI | URL |
|---|---|
| Grafana (dashboard: *Triton TRT-LLM Text-IO — Serving Overview*) | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Jaeger (traces) | http://localhost:16686 |

> If Triton runs in a container rather than on the host, put it on the same Docker
> network and change the Prometheus target from `host.docker.internal:8002` to the
> Triton container name.

## 3. What you can see (and the metric names behind each panel)

| Signal | Metric | Why it matters |
|---|---|---|
| Throughput | `nv_inference_count` (rate) | req/s per model, incl. routing split |
| Queue depth | `nv_inference_pending_request_count` | admission pressure / saturation |
| Latency | `nv_inference_request_duration_us`, `nv_inference_queue_duration_us` | compute vs. waiting |
| KV-cache health | `nv_trt_llm_kv_cache_block_metrics` | II.2 reuse — used vs. max blocks |
| GPU util / power | `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_POWER_USAGE` | utilization + **tokens/s/W** on the 72 W L4 |
| Guard activity | `nv_inference_count{model="guardrail"}` | how often safety fires |
| Per-stage latency | OTel spans → Jaeger + spanmetrics | *where* a slow request spent time |

## 4. tokens/s/W

Divide your measured output-token rate (from `genai-perf`) by the Grafana
`DCGM_FI_DEV_POWER_USAGE` reading during the run. On an L4 this is an honest
efficiency number few inference portfolios report.
