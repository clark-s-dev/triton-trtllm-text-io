#!/usr/bin/env python3
"""ab_vllm_vs_trtllm.py — the M1 A/B half that measures vLLM, with the SAME methodology
as client/perf_benchmark.py so the two are directly comparable (L2-LAB M1).

Why a second script instead of extending perf_benchmark.py: vLLM speaks the OpenAI HTTP
streaming API, the engine speaks Triton gRPC. The *methodology* is identical — closed-loop
C workers over N requests; TTFT = first streamed token; ITL = inter-token gaps; throughput
= total output tokens / wall; deterministic output length (ignore_eos + min_tokens, the
OpenAI-API equivalent of perf_benchmark's "no end_id"). Output is the SAME JSON schema, so
you can paste perf_benchmark's line into --compare-json and get a side-by-side delta table.

Honest by construction: this measures whatever vLLM you point it at. It fabricates nothing.
Until you run it against a live vLLM (lab/vllm_serve.sh), the M1 report (notebook 0015) keeps
the vLLM column marked TODO.

  # 1) TRT-LLM side (Triton up):
  .venv/bin/python client/perf_benchmark.py --target engine --model tensorrt_llm_small \
      --concurrency 32 --num-requests 256 --input-len 128 --output-len 128
  # -> copy its 'JSON {...}' line
  # 2) vLLM side (docker stop triton-llm; bash lab/vllm_serve.sh):
  python3 lab/ab_vllm_vs_trtllm.py --model tensorrt_llm_small \
      --concurrency 32 --num-requests 256 --input-len 128 --output-len 128 \
      --compare-json '{"target":"engine",...}'
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import urllib.request


def make_prompt(input_len: int, shared_len: int, uid: int) -> str:
    """A whitespace prompt ~input_len tokens: a fixed shared prefix (so vLLM prefix-caching
    can hit, mirroring perf_benchmark.make_input_ids) + a per-request unique tail."""
    s = min(shared_len, input_len)
    shared = " ".join(f"w{i % 500}" for i in range(s))
    tail = " ".join(f"u{uid}_{j}" for j in range(input_len - s))
    return (shared + " " + tail).strip()


def run_one(host: str, model: str, prompt: str, output_len: int):
    """One streaming /v1/completions request. Returns (ttft_s, e2e_s, n_tok, [itl_s], err)."""
    url = f"http://{host}/v1/completions"
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "min_tokens": output_len,   # vLLM extension: force the full length (clean throughput)
        "ignore_eos": True,         # vLLM extension: == perf_benchmark's "no end_id"
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    stamps, ntok, err = [], 0, None
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                delta = obj.get("choices", [{}])[0].get("text", "")
                if delta:
                    stamps.append(time.perf_counter())
                    ntok += 1
    except Exception as e:  # noqa: BLE001 — surface any transport/HTTP error as a result
        err = f"{type(e).__name__}: {e}"
    e2e = time.perf_counter() - t0
    ttft = (stamps[0] - t0) if stamps else None
    itls = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    return ttft, e2e, ntok, itls, err


def pctl(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))]


def measure(args) -> dict:
    tasks: "queue.Queue" = queue.Queue()
    for i in range(args.num_requests):
        tasks.put(i)
    results, lock = [], threading.Lock()

    def worker():
        while True:
            try:
                i = tasks.get_nowait()
            except queue.Empty:
                return
            prompt = make_prompt(args.input_len, args.shared_prefix_len, i)
            res = run_one(args.host, args.model, prompt, args.output_len)
            with lock:
                results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r[4] is None]
    errs = [r for r in results if r[4] is not None]
    ttfts = sorted(r[0] for r in ok if r[0] is not None)
    itls = sorted(x for r in ok for x in r[3])
    total_out = sum(r[2] for r in ok)
    return {
        "label": args.label or "vllm", "target": "vllm", "model": args.model,
        "concurrency": args.concurrency, "num_requests": args.num_requests,
        "input_len": args.input_len, "output_len": args.output_len,
        "shared_prefix_len": args.shared_prefix_len,
        "completed": len(ok), "errors": len(errs), "wall_s": round(wall, 3),
        "throughput_tok_s": round(total_out / wall, 1) if wall else 0,
        "req_per_s": round(len(ok) / wall, 2) if wall else 0,
        "ttft_p50_ms": round(pctl(ttfts, 0.5) * 1000, 1) if ttfts else float("nan"),
        "ttft_p99_ms": round(pctl(ttfts, 0.99) * 1000, 1) if ttfts else float("nan"),
        "itl_p50_ms": round(pctl(itls, 0.5) * 1000, 2) if itls else float("nan"),
        "itl_p99_ms": round(pctl(itls, 0.99) * 1000, 2) if itls else float("nan"),
        "_sample_err": errs[0][4] if errs else None,
    }


def _row(name, s):
    return (f"  {name:<10} {s['throughput_tok_s']:>10} {s['req_per_s']:>9} "
            f"{s['ttft_p50_ms']:>11} {s['ttft_p99_ms']:>11} "
            f"{s['itl_p50_ms']:>10} {s['itl_p99_ms']:>10}")


def print_compare(trt: dict, vllm: dict):
    print("\n  vLLM vs TRT-LLM (same machine, same model, same workload)")
    print(f"  {'':<10} {'tok/s':>10} {'req/s':>9} {'TTFT p50':>11} {'TTFT p99':>11} "
          f"{'ITL p50':>10} {'ITL p99':>10}")
    print(_row("TRT-LLM", trt))
    print(_row("vLLM", vllm))

    def delta(key, lower_better):
        a, b = trt.get(key), vllm.get(key)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not a:
            return "  n/a"
        pct = (b - a) / a * 100
        winner = "vLLM" if ((b < a) == lower_better) else "TRT-LLM"
        return f"{pct:+6.1f}%  ({winner} better)"

    print("\n  deltas (vLLM relative to TRT-LLM):")
    print(f"    throughput : {delta('throughput_tok_s', False)}")
    print(f"    TTFT p50   : {delta('ttft_p50_ms', True)}")
    print(f"    TTFT p99   : {delta('ttft_p99_ms', True)}")
    print(f"    ITL  p50   : {delta('itl_p50_ms', True)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost:8003")
    ap.add_argument("--model", default="tensorrt_llm_small",
                    help="vLLM --served-model-name (set to match in vllm_serve.sh)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num-requests", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--shared-prefix-len", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--compare-json", default="",
                    help="a perf_benchmark.py 'JSON {...}' line for the TRT-LLM side -> "
                         "prints a side-by-side delta table")
    args = ap.parse_args()

    s = measure(args)
    print("JSON " + json.dumps({k: v for k, v in s.items() if not k.startswith("_")}))
    print(f"""  {s['label']}: C={args.concurrency} N={args.num_requests} in/out={args.input_len}/{args.output_len}
    completed {s['completed']}/{args.num_requests}  errors {s['errors']}  wall {s['wall_s']}s
    throughput {s['throughput_tok_s']} tok/s   ({s['req_per_s']} req/s)
    TTFT  p50 {s['ttft_p50_ms']} ms   p99 {s['ttft_p99_ms']} ms
    ITL   p50 {s['itl_p50_ms']} ms   p99 {s['itl_p99_ms']} ms""")
    if s["_sample_err"]:
        print("    sample error:", s["_sample_err"])
        print("    (is vLLM up? `bash lab/vllm_serve.sh` — and Triton stopped to free the GPU?)")

    if args.compare_json:
        try:
            trt = json.loads(args.compare_json[args.compare_json.index("{"):])
            print_compare(trt, s)
        except (ValueError, json.JSONDecodeError) as e:
            print("  --compare-json: could not parse the TRT-LLM JSON:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
