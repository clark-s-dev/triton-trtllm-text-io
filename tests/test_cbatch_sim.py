"""Tests for the M2 toy continuous-batching simulator (lab/cbatch_sim.py) — pure Python.

These pin the §7 pass condition (reproduce the throughput-vs-concurrency KNEE at
max_batch_size) plus the mechanisms behind lab-notebooks 0003 (continuous vs static),
0006 (scheduler policy), and 0001/0007 (the two concurrency ceilings). They assert
robust *shape* invariants and ballpark magnitudes — not exact tok/s, since the engine
numbers carry their own run-to-run noise (0003 and 0008 disagree ~14% on one point).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))

from cbatch_sim import (  # noqa: E402
    Simulator, SimConfig, Request, closed_loop_requests, shared_prefix_requests,
    sweep_concurrency,
)


def _avg_occupancy(res):
    return sum(r.running for r in res.trace) / len(res.trace) if res.trace else 0.0


# ---- §7 pass condition: the max_batch_size knee (lab-notebook 0008) ----------
def test_throughput_rises_then_plateaus_at_knee():
    for bs in (16, 64, 128):
        tp = sweep_concurrency(bs, [bs // 4, bs, bs * 2], num_requests=1024)
        below, at, above = tp[bs // 4], tp[bs], tp[bs * 2]
        assert at > below * 1.3, f"bs{bs}: throughput should rise up to the knee"
        # Past the knee, extra concurrency only queues -> throughput plateaus (<=10% more).
        assert above <= at * 1.10, f"bs{bs}: throughput should plateau past the knee"


def test_higher_max_batch_size_raises_the_ceiling():
    # At C=128, a bigger batch ceiling = more concurrent decode = more throughput.
    tp = {bs: sweep_concurrency(bs, [128], num_requests=1024)[128] for bs in (16, 64, 128)}
    assert tp[16] < tp[64] < tp[128]
    # bs16 is hard-capped near its 16-wide rate; bs128 is several times higher.
    assert tp[128] > 2.5 * tp[16]


def test_knee_calibration_ballpark():
    # bs16 plateau ~2200 tok/s (engine 0008: 2202); bs64@C64 ~6000 (engine: 6086).
    bs16 = sweep_concurrency(16, [64], num_requests=1024)[64]
    bs64 = sweep_concurrency(64, [64], num_requests=1024)[64]
    assert 1900 < bs16 < 2500, f"bs16 plateau {bs16:.0f} not near engine 2202"
    assert 5200 < bs64 < 6700, f"bs64@C64 {bs64:.0f} not near engine 6086"


# ---- continuous vs static batching (lab-notebook 0003) -----------------------
def test_continuous_beats_static_throughput_and_ttft():
    common = dict(max_batch_size=64)
    reqs = lambda: closed_loop_requests(512, 128, 16, 256, seed=1)  # noqa: E731
    cont = Simulator(SimConfig(mode="continuous", **common)).run(reqs(), n_clients=32)
    stat = Simulator(SimConfig(mode="static", **common)).run(reqs(), n_clients=32)
    # Static = head-of-line blocking: lower throughput, far worse TTFT, lower occupancy.
    assert cont.throughput_tok_s > 1.3 * stat.throughput_tok_s
    assert cont.ttft_p50() < stat.ttft_p50() / 5
    assert _avg_occupancy(cont) > _avg_occupancy(stat) * 1.4


def test_static_v1_forces_no_evict_policy():
    # TRT-LLM 0.14: V1 only supports GUARANTEED_NO_EVICT (0003 §3). The sim enforces it.
    sim = Simulator(SimConfig(mode="static", policy="max_utilization"))
    assert sim.cfg.policy == "guaranteed_no_evict"


# ---- the two concurrency ceilings (lab-notebook 0001 / 0007) -----------------
def test_kv_capacity_crossover():
    # 0.5B pool = 2979 blocks, bs=64. Below ctx ~2944 the batch-size ceiling binds
    # (peak 64); above it the KV ceiling binds (peak = pool / footprint). no_evict
    # reserves the full footprint at admission, giving the clean capacity ceiling.
    def peak_for(plen, olen=32):
        reqs = [Request(rid=i, prompt_len=plen, output_len=olen) for i in range(128)]
        res = Simulator(SimConfig(max_batch_size=64, kv_pool_blocks=2979,
                                  policy="guaranteed_no_evict")).run(reqs, n_clients=64)
        return res.peak_running()

    assert peak_for(128) == 64          # tiny ctx -> batch-size bound
    assert peak_for(2048) == 64         # ctx 2080 still < crossover
    p3500, p4500, p8000 = peak_for(3500), peak_for(4500), peak_for(8000)
    assert p3500 < 64                   # ctx 3532 > crossover -> KV bound
    assert p4500 < p3500                # bigger context -> fewer fit (monotone)
    assert p8000 < p4500
    # Matches min(64, pool/footprint) within rounding (reserve uses prompt+output).
    for plen, exp in ((4500, 41), (8000, 23)):
        assert abs(peak_for(plen) - exp) <= 1


# ---- scheduler policy: max_utilization vs guaranteed_no_evict (0006) ---------
def test_policy_preemption_behavior():
    reqs = lambda: [Request(rid=i, prompt_len=512, output_len=64) for i in range(64)]  # noqa: E731
    cfg = dict(max_batch_size=64, kv_pool_blocks=100)   # pool < working set -> KV pressure
    mu = Simulator(SimConfig(policy="max_utilization", **cfg)).run(reqs(), n_clients=16)
    ne = Simulator(SimConfig(policy="guaranteed_no_evict", **cfg)).run(reqs(), n_clients=16)
    assert ne.total_preemptions() == 0          # no_evict never evicts (reserves footprint)
    assert mu.total_preemptions() > 0           # max_util over-commits -> recompute under pressure
    assert mu.peak_running() >= ne.peak_running()  # max_util admits at least as eagerly
    # max_util pays for eager admission with a worse recompute tail (TTFT p99).
    assert mu.ttft_p99() >= ne.ttft_p99()


def test_no_pressure_means_policies_agree():
    # 0006 §5 lesson: with KV NOT saturated, the two policies behave identically.
    reqs = lambda: closed_loop_requests(256, 128, 16, 128, seed=2)  # noqa: E731
    cfg = dict(max_batch_size=64, kv_pool_blocks=2979)  # huge pool -> no pressure
    mu = Simulator(SimConfig(policy="max_utilization", **cfg)).run(reqs(), n_clients=32)
    ne = Simulator(SimConfig(policy="guaranteed_no_evict", **cfg)).run(reqs(), n_clients=32)
    assert mu.total_preemptions() == 0 and ne.total_preemptions() == 0
    assert abs(mu.throughput_tok_s - ne.throughput_tok_s) < 0.02 * mu.throughput_tok_s


# ---- composes with the M3 allocator: KV reuse cuts TTFT (0004) ---------------
def test_kv_reuse_cuts_prefill_ttft():
    reqs = lambda: shared_prefix_requests(20, 4000, 3968, output_len=8)  # noqa: E731
    on = Simulator(SimConfig(max_batch_size=64, enable_reuse=True)).run(reqs(), n_clients=1)
    off = Simulator(SimConfig(max_batch_size=64, enable_reuse=False)).run(reqs(), n_clients=1)
    warm_on = sorted(r.ttft for r in on.requests if r.rid > 0 and r.ttft is not None)
    warm_off = sorted(r.ttft for r in off.requests if r.rid > 0 and r.ttft is not None)
    # Warm requests reuse the 62-block prefix -> first token nearly free vs full prefill.
    assert warm_on[len(warm_on) // 2] < warm_off[len(warm_off) // 2] / 5


def test_open_loop_arrivals_run():
    # Smoke: the §7-literal interface — explicit (arrival, prompt_len, output_len).
    reqs = [Request(rid=i, prompt_len=128, output_len=32, arrival=i * 5.0) for i in range(20)]
    res = Simulator(SimConfig(max_batch_size=64)).run(reqs)   # no n_clients -> open loop
    assert res.completed == 20
    assert all(r.finish_time is not None and r.ttft is not None for r in res.requests)


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in ALL:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(ALL)} continuous-batching-sim tests passed")
