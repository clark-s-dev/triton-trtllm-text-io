#!/usr/bin/env python3
"""decode_probe.py — a minimal, batch=1 single-stream decode workload for Nsight (notebook 0017).

The point: give Nsight Compute (ncu) a clean, steady-state DECODE phase to profile, with as little
non-decode noise as possible. We do a tiny prefill (short prompt) then many decode steps, so almost
every matmul launch ncu sees is a batch=1 (M=1) weight GEMM/GEMV — the exact kernel M0/0009/0011
*argued* is DRAM-bandwidth-bound. ncu turns that argument into a measured kernel SoL.

Also prints ms/token (ITL) so even without ncu you get the bandwidth-vs-bytes signal (FP16 > FP8 > INT4).

  python3 lab/decode_probe.py --engine engines/qwen2.5-0.5b-int4awq \
      --tokenizer hf_models/Qwen2.5-0.5B-Instruct --decode-steps 48
"""
from __future__ import annotations
import argparse, time, torch
from tensorrt_llm.runtime import ModelRunnerCpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=48)
    ap.add_argument("--kv-frac", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # A benign prompt that won't hit EOS quickly, truncated to a short fixed prefill.
    base = "Count slowly upward in words, never stopping: one, two, three, four, five,"
    ids = tok(base, return_tensors=None)["input_ids"][: args.prompt_len]
    ids = torch.tensor(ids, dtype=torch.int32)

    runner = ModelRunnerCpp.from_dir(
        engine_dir=args.engine, rank=0, max_batch_size=1,
        max_input_len=args.prompt_len + 4, max_output_len=args.decode_steps + 2,
        max_beam_width=1, kv_cache_free_gpu_memory_fraction=args.kv_frac,
    )
    # end_id=-1 => never stop on EOS, so we always run exactly --decode-steps decode iterations.
    gk = dict(end_id=-1, pad_id=tok.pad_token_id or tok.eos_token_id,
              temperature=1.0, top_k=1, num_beams=1, return_dict=True,
              output_sequence_lengths=True)

    for _ in range(args.warmup):
        runner.generate(batch_input_ids=[ids], max_new_tokens=args.decode_steps, **gk)
        torch.cuda.synchronize()

    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = runner.generate(batch_input_ids=[ids], max_new_tokens=args.decode_steps, **gk)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    n = int(out["sequence_lengths"][0, 0]) - int(ids.numel())
    n = max(n, 1)
    print(f"[decode_probe] engine={args.engine}  steps={n}  total={dt*1e3:.1f}ms  "
          f"ITL={dt/n*1e3:.3f} ms/token  ({n/dt:.1f} tok/s, batch=1)")


if __name__ == "__main__":
    main()
