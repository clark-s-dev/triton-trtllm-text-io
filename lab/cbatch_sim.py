#!/usr/bin/env python3
"""cbatch_sim.py — M2 toy: a minimal continuous-batching scheduler simulator.

The L2-LAB §7 artifact for M2. A *pure-Python, GPU-free* iteration-level model of an
inflight (continuous) batching scheduler, built to reproduce numbers measured on the
real TensorRT-LLM engine in lab-notebooks:

  * 0003 — continuous (`inflight_fused_batching`) vs static (`V1`) batching: the ~2x
           throughput gap + head-of-line blocking (the 31->1 "sawtooth").
  * 0006 — `batch_scheduler_policy` max_utilization vs guaranteed_no_evict under KV
           pressure: eager-admit (low TTFT, occasional recompute) vs reserve (high TTFT).
  * 0008 — `max_batch_size` sweep (16/64/128): the throughput-vs-concurrency KNEE — it
           rises with concurrency until ~max_batch_size, then plateaus (extra requests
           queue). This is the §7 pass condition.
  * 0001 — the two concurrency ceilings (max_batch_size vs KV capacity) and their
           ~4278-token crossover.

It is iteration-level on purpose: every loop turn re-decides admit / evict / step, which
is exactly what makes continuous batching beat static (a finished sequence frees its slot
*immediately*, vs. static's "whole batch waits for the longest member"). KV accounting is
delegated to the M3 toy (`paged_kv.PagedKVAllocator`) so the two artifacts compose.

Cost model (decode iteration time, ms):  T(B) = T_WEIGHT + B * T_TOKEN
fitted to 0008's 0.5B engine: one weight load is amortized over the whole batch B
(T_WEIGHT), plus a marginal per-token compute cost (T_TOKEN). This single line reproduces
the measured bs16 plateau (~2200 tok/s) and the C=16/32/64 points to within a few percent;
it is deliberately optimistic at very large B (it does not model compute-bound saturation),
which is the toy-vs-engine gap documented in notebook 0013 §5.

Run it: `python3 lab/cbatch_sim.py` prints the 0008 knee table + the 0003 contrast.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from paged_kv import PagedKVAllocator, TOKENS_PER_BLOCK

# --- cost model, fitted to lab-notebook 0008 (0.5B engine), see module docstring ---
T_WEIGHT = 6.1        # ms: fixed per-iteration cost (weight load), amortized over the batch
T_TOKEN = 0.071       # ms: marginal cost per decoded token in the batch
T_PREFILL_TOKEN = 0.0065  # ms/prompt-tok during prefill; anchored to 0003 C=1 in=128 TTFT=7ms
PREFILL_CHUNK = 512   # tokens per prefill step when chunked_context is on (0005)


@dataclass
class Request:
    """One inference request. `arrival` is in ms (None ⇒ closed-loop, released by a client)."""
    rid: int
    prompt_len: int
    output_len: int
    arrival: Optional[float] = None
    shared_prefix: int = 0          # leading tokens shared with other requests (KV reuse)
    # --- filled in by the simulator ---
    state: str = "pending"          # pending -> waiting -> prefill -> decode -> done
    admit_time: Optional[float] = None
    ttft: Optional[float] = None    # ms from arrival to first decoded token
    finish_time: Optional[float] = None
    arrival_eff: float = 0.0        # effective arrival (set when released)
    prefill_left: int = 0           # prompt tokens still to prefill
    generated: int = 0              # decode tokens emitted so far
    reused_tokens: int = 0          # prompt tokens served from KV reuse (skipped prefill)
    preemptions: int = 0            # times evicted + recomputed (max_utilization)


@dataclass
class IterRecord:
    it: int
    time: float
    dur: float          # wall-time of this single iteration (ms)
    running: int
    prefill: int
    decode: int
    waiting: int
    admitted: int
    evicted: int
    kv_used_blocks: int


@dataclass
class SimResult:
    config: "SimConfig"
    requests: list[Request]
    trace: list[IterRecord]
    wall_ms: float
    total_out_tokens: int
    completed: int

    @property
    def throughput_tok_s(self) -> float:
        return self.total_out_tokens / (self.wall_ms / 1000) if self.wall_ms else 0.0

    def steady_throughput_tok_s(self, fill: float = 0.9) -> float:
        """Decode throughput over the SATURATED window (iterations at >= `fill` x peak
        occupancy), excluding the batch-fill warmup and end-of-run drain. This is the
        apples-to-apples match for the engine's warmed-up benchmark number (L2-LAB §4.5).
        """
        if not self.trace:
            return 0.0
        thresh = fill * self.peak_running()
        sel = [r for r in self.trace if r.running >= thresh and r.decode > 0]
        toks = sum(r.decode for r in sel)
        ms = sum(r.dur for r in sel)
        return toks / (ms / 1000) if ms else self.throughput_tok_s

    def _ttfts(self) -> list[float]:
        return sorted(r.ttft for r in self.requests if r.ttft is not None)

    def ttft_p50(self) -> float:
        return _pctl(self._ttfts(), 0.5)

    def ttft_p99(self) -> float:
        return _pctl(self._ttfts(), 0.99)

    def peak_running(self) -> int:
        return max((r.running for r in self.trace), default=0)

    def total_preemptions(self) -> int:
        return sum(r.preemptions for r in self.requests)

    def occupancy_histogram(self) -> dict[int, int]:
        """How many iterations were spent at each running-count (the 0003 sawtooth)."""
        hist: dict[int, int] = {}
        for rec in self.trace:
            hist[rec.running] = hist.get(rec.running, 0) + 1
        return hist


def _pctl(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))]


@dataclass
class SimConfig:
    max_batch_size: int = 64
    kv_pool_blocks: int = 2979          # 0.5B @ fraction 0.25 (lab-notebook 0001)
    tokens_per_block: int = TOKENS_PER_BLOCK
    mode: str = "continuous"            # "continuous" (inflight) | "static" (V1)
    policy: str = "max_utilization"     # "max_utilization" | "guaranteed_no_evict"
    enable_reuse: bool = False
    enable_chunked: bool = True
    max_iters: int = 2_000_000


class Simulator:
    """Iteration-level continuous/static batching simulator over a paged KV pool."""

    def __init__(self, config: SimConfig):
        self.cfg = config
        if config.mode == "static":
            # V1 in TRT-LLM 0.14 only supports GUARANTEED_NO_EVICT (lab-notebook 0003).
            self.cfg.policy = "guaranteed_no_evict"
        self.kv = PagedKVAllocator(config.kv_pool_blocks, config.tokens_per_block,
                                   enable_reuse=config.enable_reuse)

    # ---- KV footprint helpers --------------------------------------------
    def _prompt_blocks(self, r: Request) -> int:
        return self.kv.n_blocks_for(r.prompt_len)

    def _full_footprint_blocks(self, r: Request) -> int:
        return self.kv.n_blocks_for(r.prompt_len + r.output_len)

    # ---- admission --------------------------------------------------------
    def _can_admit(self, r: Request) -> bool:
        """Policy gate. no_evict reserves the request's *whole* projected KV footprint
        so it can never be evicted mid-flight; max_utilization only needs the prompt to
        fit now and gambles that decode-time growth won't overflow (0006)."""
        if self.cfg.policy == "guaranteed_no_evict":
            need = self._full_footprint_blocks(r)
        else:
            need = self._prompt_blocks(r)
        return self.kv.free_blocks >= need

    def _admit(self, r: Request, now: float) -> bool:
        res = self.kv.allocate(r.rid, list(range(r.prompt_len)) if not self.cfg.enable_reuse
                               else self._token_ids(r))
        if not res.ok:
            return False
        r.reused_tokens = res.reused_blocks * self.cfg.tokens_per_block
        r.prefill_left = max(0, r.prompt_len - r.reused_tokens)
        r.state = "decode" if r.prefill_left == 0 else "prefill"
        r.admit_time = now
        if r.prefill_left == 0:               # full prefix reuse -> first token essentially free
            r.ttft = now - r.arrival_eff
        return True

    def _token_ids(self, r: Request) -> list[int]:
        # Reuse-mode workloads share an identical prefix (the 0004 design); only the
        # leading `shared_prefix` tokens are common, the rest is per-request unique.
        shared = min(r.shared_prefix, r.prompt_len)
        ids = [(i % 500) + 5 for i in range(shared)]
        ids += [600 + (r.rid * 7919 + j) % 150000 for j in range(r.prompt_len - shared)]
        return ids

    # ---- the main loop ----------------------------------------------------
    def run(self, requests: list[Request], n_clients: Optional[int] = None) -> SimResult:
        """Simulate `requests`. If `n_clients` is set, run CLOSED-LOOP (perf_benchmark
        style): start `n_clients` requests, and release the next pending one each time a
        request completes — so at most `n_clients` are ever outstanding. Otherwise run
        OPEN-LOOP using each request's `arrival` time."""
        cfg = self.cfg
        pending = list(requests)
        waiting: list[Request] = []
        running: list[Request] = []
        trace: list[IterRecord] = []
        now = 0.0
        it = 0
        closed = n_clients is not None
        in_flight = 0  # closed-loop: dispatched-but-not-done

        def release_arrivals():
            nonlocal in_flight
            if closed:
                while pending and in_flight < n_clients:
                    r = pending.pop(0)
                    r.arrival_eff = now
                    r.state = "waiting"
                    waiting.append(r)
                    in_flight += 1
            else:
                ready = [r for r in pending if (r.arrival or 0) <= now]
                for r in ready:
                    pending.remove(r)
                    r.arrival_eff = r.arrival or 0.0
                    r.state = "waiting"
                    waiting.append(r)

        release_arrivals()

        while (waiting or running or pending) and it < cfg.max_iters:
            admitted = evicted = 0

            # 1) ADMISSION
            if cfg.mode == "static":
                # V1: only start a fresh wave when the batch has fully drained
                # (head-of-line blocking). No mid-flight admission.
                if not running:
                    while waiting and len(running) < cfg.max_batch_size:
                        if self._admit(waiting[0], now):
                            running.append(waiting.pop(0)); admitted += 1
                        else:
                            break
            else:
                # Continuous: top up the batch every iteration from the head of the queue.
                # Admission is opportunistic (no eviction here): max_utilization admits on
                # prompt-fit and over-commits, relying on decode-time eviction under
                # pressure; guaranteed_no_evict reserves the full footprint so it never
                # over-commits (and so never evicts). See _can_admit.
                while waiting and len(running) < cfg.max_batch_size:
                    r = waiting[0]
                    if self._can_admit(r) and self._admit(r, now):
                        running.append(waiting.pop(0)); admitted += 1
                    else:
                        break  # head-of-queue cannot fit -> wait (FCFS, no starvation)

            if not running:
                # Nothing admittable (e.g. KV reserved out under no_evict) but work remains:
                # advance a tick so the clock moves; otherwise we'd spin.
                if not waiting and not pending:
                    break
                now += T_WEIGHT
                it += 1
                if not closed and pending:
                    release_arrivals()
                continue

            # 2) EXECUTE ONE ITERATION (fused prefill + decode forward pass)
            prefill_tokens = 0
            decode_tokens = 0
            for r in running:
                if r.state == "prefill":
                    chunk = (min(PREFILL_CHUNK, r.prefill_left) if cfg.enable_chunked
                             else r.prefill_left)
                    r.prefill_left -= chunk
                    prefill_tokens += chunk
                else:
                    decode_tokens += 1
            iter_time = T_WEIGHT + decode_tokens * T_TOKEN + prefill_tokens * T_PREFILL_TOKEN
            now += iter_time
            it += 1

            # 3) POST-STEP: transition prefill->decode, grow KV, complete, free slots
            finished: list[Request] = []
            for r in running:
                if r.state == "prefill":
                    if r.prefill_left == 0:           # prefill done this iter
                        r.state = "decode"
                        if r.ttft is None:
                            r.ttft = now - r.arrival_eff   # first token emitted now
                    continue
                # decode: emitted one token
                r.generated += 1
                # KV may need a new block when decode crosses a block boundary
                tokens_held = r.prompt_len + r.generated
                if tokens_held % cfg.tokens_per_block == 1 and r.generated > 0:
                    if not self.kv.append_block(r.rid):
                        if cfg.policy == "max_utilization" and self._evict_one(
                                running, waiting, protect=r):
                            evicted += 1
                            self.kv.append_block(r.rid)   # room freed; retry
                if r.generated >= r.output_len:
                    r.state = "done"
                    r.finish_time = now
                    finished.append(r)

            for r in finished:
                running.remove(r)
                self.kv.free(r.rid)
                if closed:
                    in_flight -= 1

            trace.append(IterRecord(
                it=it, time=now, dur=iter_time, running=len(running) + len(finished),
                prefill=sum(1 for r in running if r.state == "prefill"),
                decode=decode_tokens,
                waiting=len(waiting), admitted=admitted, evicted=evicted,
                kv_used_blocks=self.kv.used_blocks))

            release_arrivals()

        done = [r for r in requests if r.state == "done"]
        return SimResult(
            config=cfg, requests=requests, trace=trace, wall_ms=now,
            total_out_tokens=sum(r.output_len for r in done), completed=len(done))

    def _evict_one(self, running: list[Request], waiting: list[Request],
                   protect: Optional[Request] = None) -> bool:
        """max_utilization preemption: evict the most-recently-admitted running request
        (LIFO, like recompute-based preemption), return it to the waiting queue, lose its
        generated tokens (recompute). Mirrors 0006's `paused`/recompute tail."""
        victims = [r for r in running if r is not protect]
        if not victims:
            return False
        victim = max(victims, key=lambda r: r.admit_time or 0)
        self.kv.free(victim.rid)
        running.remove(victim)
        victim.generated = 0
        victim.prefill_left = victim.prompt_len - victim.reused_tokens
        victim.state = "waiting"
        victim.ttft = None
        victim.preemptions += 1
        waiting.append(victim)  # to the BACK: re-admitting at the front would thrash
        return True


# ---------------------------------------------------------------------------
# Workload builders (closed-loop, perf_benchmark.py style)
# ---------------------------------------------------------------------------
def closed_loop_requests(num_requests: int, prompt_len: int,
                         out_min: int, out_max: int, seed: int = 0) -> list[Request]:
    """N requests, fixed prompt_len, output length drawn uniformly in [out_min, out_max]
    (varied output exposes continuous-vs-static, per 0003/0008)."""
    rng = random.Random(seed)
    return [Request(rid=i, prompt_len=prompt_len,
                    output_len=rng.randint(out_min, out_max)) for i in range(num_requests)]


def shared_prefix_requests(num_requests: int, prompt_len: int, shared_prefix: int,
                           output_len: int = 8) -> list[Request]:
    """N requests that share an identical `shared_prefix`-token prefix — the 0004 KV-reuse
    workload, as a simulator input. Run with `SimConfig(enable_reuse=True)`."""
    return [Request(rid=i, prompt_len=prompt_len, output_len=output_len,
                    shared_prefix=shared_prefix) for i in range(num_requests)]


def sweep_concurrency(max_batch_size: int, concurrencies: list[int], *,
                      num_requests: int = 1024, prompt_len: int = 128,
                      out_min: int = 16, out_max: int = 256,
                      kv_pool_blocks: int = 2979, mode: str = "continuous",
                      seed: int = 0) -> dict[int, float]:
    """Saturated throughput (tok/s) vs concurrency for one max_batch_size — the 0008
    experiment. Uses the steady-window throughput (warmed up, like the engine bench)."""
    out: dict[int, float] = {}
    for c in concurrencies:
        cfg = SimConfig(max_batch_size=max_batch_size, kv_pool_blocks=kv_pool_blocks, mode=mode)
        reqs = closed_loop_requests(num_requests, prompt_len, out_min, out_max, seed=seed)
        res = Simulator(cfg).run(reqs, n_clients=c)
        out[c] = res.steady_throughput_tok_s()
    return out


def _main() -> int:
    print("M2 toy continuous-batching simulator — calibration vs lab-notebooks 0008 / 0003\n")

    # --- 0008: the max_batch_size knee ---
    print("  throughput (tok/s) vs concurrency, by max_batch_size  [in=128, out=16..256]")
    print(f"  {'C':>5} | {'bs16':>7} | {'bs64':>7} | {'bs128':>7}    (engine 0008 in [brackets])")
    Cs = [16, 32, 64, 128]
    s16 = sweep_concurrency(16, Cs)
    s64 = sweep_concurrency(64, Cs)
    s128 = sweep_concurrency(128, Cs)
    eng = {16: ("2202", "—", "—"), 32: ("2516", "3596", "3810"),
           64: ("~2533", "6086", "5999"), 128: ("~2533", "6630", "6999")}
    for c in Cs:
        e = eng[c]
        print(f"  {c:>5} | {s16[c]:>7.0f} | {s64[c]:>7.0f} | {s128[c]:>7.0f}    "
              f"[{e[0]:>5} {e[1]:>5} {e[2]:>5}]")
    print("\n  -> rises with C until ~max_batch_size, then plateaus = the KNEE (pass condition).")

    # --- 0003: continuous vs static ---
    print("\n  continuous vs static (V1) batching  [bs=64, C=32, in=128, out=16..256]")
    tputs = {}
    for mode in ("continuous", "static"):
        cfg = SimConfig(max_batch_size=64, mode=mode)
        res = Simulator(cfg).run(closed_loop_requests(512, 128, 16, 256, seed=1), n_clients=32)
        peak = res.peak_running()
        avg_occ = sum(r.running for r in res.trace) / len(res.trace)
        tputs[mode] = res.throughput_tok_s
        print(f"    {mode:>10}: throughput {res.throughput_tok_s:>6.0f} tok/s   "
              f"TTFT p50 {res.ttft_p50():>7.1f} ms   avg_running {avg_occ:>4.1f}/{peak}")
    print(f"    continuous/static: throughput {tputs['continuous']/tputs['static']:.2f}x, "
          f"TTFT shows head-of-line blocking   (engine 0003: ~2.1x throughput, ~195x TTFT)")
    return 0


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(_main())
