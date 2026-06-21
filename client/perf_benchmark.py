#!/usr/bin/env python3
"""perf_benchmark.py — the §6 "measurement spine": a closed-loop load generator +
latency/throughput meter for the triton-trtllm-text-io stack.

Two targets:
  * engine  — hit tensorrt_llm_{small,large} DIRECTLY with raw token ids (the right
              layer for scheduler/KV ablations: bypasses the BLS + guard, see
              L2-LAB §4.1). Output length is deterministic (no end_id) so throughput
              is clean.
  * bls     — hit text_pipeline_bls with chat messages (end-to-end, incl. guard).

Drives C concurrent streams through N requests (closed loop) and reports
TTFT / ITL / throughput / P50 / P99. Prints one JSON line (for scripts) + a table.

  .venv/bin/python client/perf_benchmark.py --target engine --model tensorrt_llm_small \
      --concurrency 16 --num-requests 64 --input-len 128 --output-len 128
  # shared 256-tok prefix (KV-reuse experiments):  ... --shared-prefix-len 256
"""
from __future__ import annotations
import argparse, json, queue, sys, threading, time
import numpy as np
import tritonclient.grpc as grpcclient

VOCAB = 151000  # Qwen2.5 vocab size; ids below this are safe


def _inp(name, arr, dtype):
    t = grpcclient.InferInput(name, list(arr.shape), dtype)
    t.set_data_from_numpy(arr)
    return t


def make_input_ids(input_len, shared_len, rng):
    # A fixed shared prefix (so KV-cache block hashes align across requests) + a
    # per-request random tail (so only the prefix is reusable).
    s = min(shared_len, input_len)
    shared = [(i % 500) + 5 for i in range(s)]
    tail = rng.integers(600, VOCAB, size=input_len - s).tolist()
    return np.array(shared + tail, dtype=np.int32)


def engine_inputs(input_ids, output_len):
    L = int(input_ids.size)
    return [
        _inp("input_ids", input_ids.reshape(1, L), "INT32"),
        _inp("input_lengths", np.array([[L]], dtype=np.int32), "INT32"),
        _inp("request_output_len", np.array([[output_len]], dtype=np.int32), "INT32"),
        _inp("streaming", np.array([[True]], dtype=bool), "BOOL"),
    ]


def bls_inputs(text, output_len):
    msgs = [{"role": "user", "content": text}]
    return [
        _inp("MESSAGES", np.array([json.dumps(msgs).encode()], dtype=np.object_), "BYTES"),
        _inp("MAX_TOKENS", np.array([output_len], dtype=np.int32), "INT32"),
    ]


def run_one(host, target, model, input_ids, output_len, text):
    """One streaming request; returns (ttft_s, e2e_s, n_out_tokens, [itl_s...], err)."""
    cli = grpcclient.InferenceServerClient(url=host)
    q: "queue.Queue" = queue.Queue()
    cli.start_stream(callback=lambda result, error: q.put((result, error)))
    if target == "engine":
        inputs, outs, name = engine_inputs(input_ids, output_len), ["output_ids", "sequence_length"], model
    else:
        inputs, outs, name = bls_inputs(text, output_len), ["TEXT", "FINISH_REASON"], "text_pipeline_bls"
    req_outs = [grpcclient.InferRequestedOutput(n) for n in outs]
    stamps, ntok, err = [], 0, None
    t0 = time.perf_counter()
    cli.async_stream_infer(name, inputs=inputs, outputs=req_outs, request_id="r")
    try:
        while True:
            r, e = q.get(timeout=120)
            if e is not None:
                err = str(e); break
            now = time.perf_counter()
            if target == "engine":
                o = r.as_numpy("output_ids")
                n = int(o.reshape(-1).size) if o is not None else 0
                if n:
                    stamps.append(now); ntok += n
                if ntok >= output_len:
                    break
            else:
                tx = r.as_numpy("TEXT")
                if tx is not None and tx.size and tx[0]:
                    stamps.append(now); ntok += 1
                fr = r.as_numpy("FINISH_REASON")
                if fr is not None and fr.size:
                    break
    except queue.Empty:
        err = "timeout"
    finally:
        try: cli.stop_stream()
        except Exception: pass
    e2e = time.perf_counter() - t0
    ttft = (stamps[0] - t0) if stamps else None
    itls = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    return ttft, e2e, ntok, itls, err


def pctl(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost:8001")
    ap.add_argument("--target", choices=["engine", "bls"], default="engine")
    ap.add_argument("--model", default="tensorrt_llm_small")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--input-len-max", type=int, default=0,
                    help="if > input-len, each request draws a random input length in [input-len, input-len-max] "
                         "(a mix of long prefills + short decodes exposes chunked context)")
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--output-len-max", type=int, default=0,
                    help="if > output-len, each request draws a random length in [output-len, output-len-max] "
                         "(varied lengths expose continuous vs static batching)")
    ap.add_argument("--shared-prefix-len", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    tasks: "queue.Queue" = queue.Queue()
    for i in range(args.num_requests):
        tasks.put(i)
    results, lock = [], threading.Lock()

    def worker():
        rng = np.random.default_rng()
        while True:
            try:
                i = tasks.get_nowait()
            except queue.Empty:
                return
            imax = args.input_len_max or args.input_len
            ilen = int(rng.integers(args.input_len, imax + 1)) if imax > args.input_len else args.input_len
            ids = make_input_ids(ilen, args.shared_prefix_len, rng)
            omax = args.output_len_max or args.output_len
            olen = int(rng.integers(args.output_len, omax + 1)) if omax > args.output_len else args.output_len
            text = f"In about {olen} tokens, give me detailed facts about NVIDIA GTC topic #{i}."
            res = run_one(args.host, args.target, args.model, ids, olen, text)
            with lock:
                results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r[4] is None]
    errs = [r for r in results if r[4] is not None]
    ttfts = sorted(r[0] for r in ok if r[0] is not None)
    itls = sorted(x for r in ok for x in r[3])
    total_out = sum(r[2] for r in ok)
    s = {
        "label": args.label, "target": args.target, "model": args.model,
        "concurrency": args.concurrency, "num_requests": args.num_requests,
        "input_len": args.input_len, "input_len_max": args.input_len_max or args.input_len,
        "output_len": args.output_len,
        "output_len_max": args.output_len_max or args.output_len,
        "shared_prefix_len": args.shared_prefix_len,
        "completed": len(ok), "errors": len(errs), "wall_s": round(wall, 3),
        "throughput_tok_s": round(total_out / wall, 1) if wall else 0,
        "req_per_s": round(len(ok) / wall, 2) if wall else 0,
        "ttft_p50_ms": round(pctl(ttfts, 0.5) * 1000, 1),
        "ttft_p99_ms": round(pctl(ttfts, 0.99) * 1000, 1),
        "itl_p50_ms": round(pctl(itls, 0.5) * 1000, 2),
        "itl_p99_ms": round(pctl(itls, 0.99) * 1000, 2),
    }
    print("JSON " + json.dumps(s))
    print(f"""  {args.label or args.target+'/'+args.model}: C={args.concurrency} N={args.num_requests} in/out={args.input_len}/{args.output_len}
    completed {s['completed']}/{args.num_requests}  errors {s['errors']}  wall {s['wall_s']}s
    throughput {s['throughput_tok_s']} tok/s   ({s['req_per_s']} req/s)
    TTFT  p50 {s['ttft_p50_ms']} ms   p99 {s['ttft_p99_ms']} ms
    ITL   p50 {s['itl_p50_ms']} ms   p99 {s['itl_p99_ms']} ms""")
    if errs:
        print("    sample error:", errs[0][4])
    return 0


if __name__ == "__main__":
    sys.exit(main())
