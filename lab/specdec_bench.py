#!/usr/bin/env python3
"""specdec_bench.py — measure Draft-Target speculative decoding (notebook 0016, M4).

Runs INSIDE the NGC trtllm container (imports tensorrt_llm). It is the spec-decode sibling of
client/perf_benchmark.py: same "emit one JSON line + a table" contract, but it drives the
TRT-LLM C++ executor DIRECTLY (ModelRunnerCpp) instead of gRPC, because draft-target speculative
decoding is an orchestration loop (draft proposes K, target verifies K+1), not a single request.
That is also the *right layer* (L2-LAB §4.1): no BLS / guard / gRPC in the timing.

What it measures, per config:
  * acceptance rate          = accepted_draft_tokens / proposed_draft_tokens
  * mean accepted / iter     = output_tokens / n_target_forward_passes   (= mean(accepted)+1)
  * throughput (tok/s)       = output_tokens / wall
  * speedup                  = specdec tok/s  /  target-only baseline tok/s  (same engine, same sampling)

The loop is faithful to TensorRT-LLM v0.14 examples/run.py:run_draft_target_model() (the reference
orchestration), with clean instrumentation and a target-only baseline mode.

  # baseline (target only, no draft):
  python3 lab/specdec_bench.py --mode baseline --target-engine engines/qwen2.5-1.5b-target \
      --tokenizer hf_models/Qwen2.5-1.5B-Instruct --output-len 200 --prompts mixed
  # speculative (draft 0.5B -> target 1.5B), sweep K:
  python3 lab/specdec_bench.py --mode specdec --draft-engine engines/qwen2.5-0.5b-draft \
      --target-engine engines/qwen2.5-1.5b-target --tokenizer hf_models/Qwen2.5-1.5B-Instruct \
      --draft-len 4 --accept logits --output-len 200 --prompts mixed
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
import torch
from tensorrt_llm.runtime import ModelRunnerCpp

# --- prompt sets: "easy" = predictable/formulaic (draft agrees often -> high acceptance);
#     "hard" = open-ended creative (draft and target diverge -> lower acceptance). The gap
#     between them is the point: acceptance is a property of the workload, not just the models.
PROMPTS = {
    "easy": [
        "List the integers from 1 to 30 separated by commas.",
        "Write the Python function that returns the nth Fibonacci number, with a docstring.",
        "Recite the first 12 lines of the multiplication table for 7 (7x1 through 7x12).",
        "Complete this JSON config with the standard fields for a TCP server (host, port, timeout, max_connections, log_level).",
        "Spell out the days of the week and the twelve months of the year, in order.",
    ],
    "hard": [
        "Invent a surreal short story about a lighthouse keeper who collects forgotten sounds.",
        "Give an unconventional, contrarian argument about the nature of time, with vivid metaphors.",
        "Describe an alien cuisine no human has tasted, inventing names, textures and aromas.",
        "Write a poem that fuses quantum mechanics with longing, avoiding any cliche.",
        "Improvise a dialogue between a glacier and a wildfire debating patience.",
    ],
}
PROMPTS["mixed"] = [p for pair in zip(PROMPTS["easy"], PROMPTS["hard"]) for p in pair]


def tokenize(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors=None)
    return torch.tensor(ids, dtype=torch.int32)


def build_runner(engine_dir, kv_frac, max_batch, max_out, max_in):
    return ModelRunnerCpp.from_dir(
        engine_dir=engine_dir,
        rank=0,
        max_batch_size=max_batch,
        max_input_len=max_in,
        max_output_len=max_out,
        max_beam_width=1,
        kv_cache_free_gpu_memory_fraction=kv_frac,
        # The draft_tokens_external target engine asserts block reuse must be ON
        # (the verify step re-attends the accepted prefix from cached KV blocks).
        kv_cache_enable_block_reuse=True,
    )


def gen_kwargs(args, end_id, pad_id):
    # greedy = top_k 1 (deterministic, maximal & reproducible acceptance); else sample.
    k = 1 if args.temperature <= 0 else args.top_k
    t = 1.0 if args.temperature <= 0 else args.temperature
    return dict(end_id=end_id, pad_id=pad_id, temperature=t, top_k=k, top_p=args.top_p,
                num_beams=1, random_seed=1234, return_dict=True, output_sequence_lengths=True)


def run_baseline(target, batch_ids, max_out, gk):
    """Plain autoregressive decode on the target engine (no draft). Times wall + counts tokens."""
    in_lens = [int(x.numel()) for x in batch_ids]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = target.generate(batch_input_ids=batch_ids, max_new_tokens=max_out, **gk)
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    seq = out["sequence_lengths"][:, 0].tolist()
    n_out = sum(seq[i] - in_lens[i] for i in range(len(batch_ids)))
    # baseline target forward passes = max generated length (continuous-batch steps to drain)
    n_iter = max(seq[i] - in_lens[i] for i in range(len(batch_ids)))
    return dict(wall=wall, n_out=n_out, n_iter=n_iter, n_draft=0, n_accept=0)


def run_specdec(draft, target, batch_ids, draft_len, max_out, use_logits, gk):
    """Draft-target loop, faithful to run.py:run_draft_target_model(), batched with per-request
    early-stop (a request leaves the batch when it hits end_id or max_out)."""
    bs0 = len(batch_ids)
    in_lens = [int(x.numel()) for x in batch_ids]
    max_seq = [in_lens[i] + max_out for i in range(bs0)]
    end_id = gk["end_id"]
    prefix = [x for x in batch_ids]
    slot = list(range(bs0))
    n_draft = [0] * bs0; n_accept = [0] * bs0; final_len = list(in_lens)
    n_iter = 0
    # The draft speculates K tokens UNCONDITIONALLY (end_id=-1 => never early-stop): the target
    # is what decides real stopping. This also avoids ExternalDraftTokensConfig's !mTokens.empty()
    # assert when the draft would otherwise propose 0 tokens by hitting EOS.
    draft_gk = dict(gk); draft_gk["end_id"] = -1
    torch.cuda.synchronize(); t0 = time.perf_counter()
    while True:
        n_iter += 1
        bs = len(prefix)
        plen = [int(prefix[i].numel()) for i in range(bs)]
        # --- draft proposes up to draft_len tokens (draft_len sequential small-model steps) ---
        d = draft.generate(batch_input_ids=prefix, max_new_tokens=draft_len, **draft_gk)
        torch.cuda.synchronize()
        d_seqlen = d["sequence_lengths"][:, 0].tolist()
        d_len = [d_seqlen[i] - plen[i] for i in range(bs)]
        d_ids = [d["output_ids"][i, 0, plen[i]:d_seqlen[i]].tolist() for i in range(bs)]
        d_logits = [d["generation_logits"][i, 0, :, :] for i in range(bs)] if use_logits else None
        # --- target verifies all draft_len in ONE forward pass, emits up to draft_len+1 tokens ---
        t = target.generate(batch_input_ids=prefix, max_new_tokens=draft_len + 1,
                             draft_tokens_list=d_ids, draft_logits_list=d_logits, **gk)
        torch.cuda.synchronize()
        t_seqlen = t["sequence_lengths"][:, 0].tolist()
        nxt, nxt_slot = [], []
        for i in range(bs):
            idx = slot[i]; l = plen[i]; r = min(t_seqlen[i], max_seq[idx])
            t_ids = t["output_ids"][i, 0, l:r].tolist()
            t_seq_ids = t["output_ids"][i, 0, :r]
            final_len[idx] = r
            n_draft[idx] += len(d_ids[i])
            n_accept[idx] += sum(d_ids[i][k] == t_ids[k]
                                 for k in range(min(d_len[i], r - l)))
            if r >= max_seq[idx]:           continue   # hit output budget
            if len(t_ids) == 0:             continue   # no progress
            # NB: check EOS only in the *generated* tokens — the Qwen chat template prompt itself
            # contains <|im_end|> (== eos id), so scanning the whole sequence stops instantly.
            if end_id in t_ids:             continue   # hit EOS
            nxt.append(t_seq_ids.to(torch.int32)); nxt_slot.append(idx)
        prefix, slot = nxt, nxt_slot
        if not prefix:
            break
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    n_out = sum(final_len[i] - in_lens[i] for i in range(bs0))
    return dict(wall=wall, n_out=n_out, n_iter=n_iter,
                n_draft=sum(n_draft), n_accept=sum(n_accept))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["specdec", "baseline"], required=True)
    ap.add_argument("--draft-engine", default="")
    ap.add_argument("--target-engine", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--draft-len", type=int, default=4, help="K: draft tokens proposed per iteration (<= engine max_draft_len)")
    ap.add_argument("--accept", choices=["logits", "tokens"], default="logits",
                    help="logits = paper's modified rejection sampling; tokens = per-token equality")
    ap.add_argument("--output-len", type=int, default=200)
    ap.add_argument("--prompts", choices=list(PROMPTS), default="mixed")
    ap.add_argument("--batch", type=int, default=1, help="run this many prompts concurrently (1 = clean per-request accounting)")
    ap.add_argument("--temperature", type=float, default=0.0, help="<=0 => greedy (top_k=1)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--draft-kv-frac", type=float, default=0.20)
    ap.add_argument("--target-kv-frac", type=float, default=0.45)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats; report the fastest (least noisy)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    end_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else end_id
    gk = gen_kwargs(args, end_id, pad_id)
    use_logits = (args.accept == "logits")

    prompts = PROMPTS[args.prompts]
    max_in = max(int(tokenize(tok, p).numel()) for p in prompts) + 8
    max_total = max_in + args.output_len + 8

    target = build_runner(args.target_engine, args.target_kv_frac, max(args.batch, 1), args.output_len + args.draft_len + 2, max_total)
    draft = None
    if args.mode == "specdec":
        if not args.draft_engine:
            print("ERROR: --draft-engine required for --mode specdec", file=sys.stderr); return 2
        draft = build_runner(args.draft_engine, args.draft_kv_frac, max(args.batch, 1), args.output_len + args.draft_len + 2, max_total)

    # group prompts into batches of size args.batch
    tokd = [tokenize(tok, p) for p in prompts]
    batches = [tokd[i:i + args.batch] for i in range(0, len(tokd), args.batch)]

    def one_pass():
        agg = dict(wall=0.0, n_out=0, n_iter=0, n_draft=0, n_accept=0)
        for b in batches:
            if args.mode == "baseline":
                r = run_baseline(target, b, args.output_len, gk)
            else:
                r = run_specdec(draft, target, b, args.draft_len, args.output_len, use_logits, gk)
            for k in agg: agg[k] += r[k]
        return agg

    for _ in range(args.warmup):
        one_pass()
    best = None
    for _ in range(max(1, args.repeats)):
        a = one_pass()
        if best is None or a["wall"] < best["wall"]:
            best = a

    tput = best["n_out"] / best["wall"] if best["wall"] else 0.0
    accept_rate = (best["n_accept"] / best["n_draft"]) if best["n_draft"] else None
    mean_acc_iter = (best["n_out"] / best["n_iter"]) if best["n_iter"] else None
    out = {
        "label": args.label, "mode": args.mode, "accept": args.accept if args.mode == "specdec" else "-",
        "draft_len": args.draft_len if args.mode == "specdec" else 0,
        "prompts": args.prompts, "batch": args.batch, "temperature": args.temperature,
        "output_len": args.output_len, "n_prompts": len(prompts),
        "wall_s": round(best["wall"], 4), "out_tokens": best["n_out"],
        "target_fwd_passes": best["n_iter"],
        "throughput_tok_s": round(tput, 1),
        "acceptance_rate": round(accept_rate, 4) if accept_rate is not None else None,
        "mean_accepted_per_iter": round(mean_acc_iter, 3) if mean_acc_iter is not None else None,
        "draft_tokens": best["n_draft"], "accepted_tokens": best["n_accept"],
    }
    print("JSON " + json.dumps(out))
    print(f"""  {args.label or args.mode}: mode={args.mode} K={out['draft_len']} accept={out['accept']} prompts={args.prompts} batch={args.batch} temp={args.temperature}
    out_tokens {out['out_tokens']}  target_fwd_passes {out['target_fwd_passes']}  wall {out['wall_s']}s
    throughput  {out['throughput_tok_s']} tok/s
    acceptance  {out['acceptance_rate']}   mean_accepted/iter {out['mean_accepted_per_iter']}""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
