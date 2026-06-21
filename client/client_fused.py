#!/usr/bin/env python3
"""Streaming client for the fused gateway: chat messages in, streamed text out.

    python3 client/client_fused.py --message "用三句话介绍 GPU 推理 🚀"
    python3 client/client_fused.py --message "hi" --model small --system "You are terse."

Sends a JSON chat array to `text_pipeline_bls` over gRPC and prints TEXT deltas as
they stream. The server does chat-templating, tokenization, streaming
detokenization, stop handling, and guardrails — the client just sends raw messages.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys

import numpy as np
import tritonclient.grpc as grpcclient


def _bytes_in(name: str, values: list[str]) -> grpcclient.InferInput:
    arr = np.array([v.encode("utf-8") for v in values], dtype=np.object_)
    t = grpcclient.InferInput(name, [len(values)], "BYTES")
    t.set_data_from_numpy(arr)
    return t


def _scalar_in(name: str, value, np_dtype, triton_dtype: str) -> grpcclient.InferInput:
    arr = np.array([value], dtype=np_dtype)
    t = grpcclient.InferInput(name, [1], triton_dtype)
    t.set_data_from_numpy(arr)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost:8001")
    ap.add_argument("--message", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--model", default="", help="route hint: small|large (blank = auto)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--stop", action="append", default=[], help="stop string (repeatable)")
    args = ap.parse_args()

    messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
        {"role": "user", "content": args.message}
    ]

    inputs = [
        _bytes_in("MESSAGES", [json.dumps(messages, ensure_ascii=False)]),
        _scalar_in("MAX_TOKENS", args.max_tokens, np.int32, "INT32"),
        _scalar_in("TEMPERATURE", args.temperature, np.float32, "FP32"),
        _scalar_in("TOP_P", args.top_p, np.float32, "FP32"),
    ]
    if args.model:
        inputs.append(_bytes_in("MODEL", [args.model]))
    if args.stop:
        inputs.append(_bytes_in("STOP", args.stop))

    outputs = [grpcclient.InferRequestedOutput("TEXT"),
               grpcclient.InferRequestedOutput("FINISH_REASON")]

    results: "queue.Queue" = queue.Queue()
    client = grpcclient.InferenceServerClient(url=args.host)
    client.start_stream(callback=lambda result, error: results.put((result, error)))
    client.async_stream_infer("text_pipeline_bls", inputs=inputs, outputs=outputs,
                              request_id="1")

    finish = None
    while True:
        result, error = results.get()
        if error is not None:
            print(f"\n[error] {error}", file=sys.stderr)
            break
        text = result.as_numpy("TEXT")
        if text is not None and text.size:
            sys.stdout.write(text[0].decode("utf-8"))
            sys.stdout.flush()
        fr = result.as_numpy("FINISH_REASON")
        if fr is not None and fr.size:           # FINISH_REASON only on the final response
            finish = fr[0].decode("utf-8")
            break
    client.stop_stream()
    print(f"\n--- finish_reason: {finish}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
