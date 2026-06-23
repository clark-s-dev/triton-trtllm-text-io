#!/usr/bin/env python3
"""specdec_model.py — a 2-constant analytical model of Draft-Target speculative-decoding speedup,
back-solved from the notebook-0016 measured sweep. The GPU-free sibling of cbatch_sim.py's T(B)
cost model: a tiny model you can calibrate against the rig, not a black box.

Physics of one spec-decode iteration (batch=1): the draft does K sequential decode steps; the target
verifies all K+1 positions in ONE forward pass. So measured per-iteration wall time is LINEAR in K:

    T_iter(K) = T_FIXED + T_DRAFT * K              # ms

  * T_DRAFT  = one 0.5B draft decode step (the slope)
  * T_FIXED  = target verify(K+1) ≈ one 1.5B step (bandwidth-bound, see 0017) + Python orchestration

Both are back-solved from the 0016 A-sweep (K=1,2,4,6,8 — a clean line, residual <1%). Baseline target
decode is T_BASE ms/token. Each iteration advances ~`mean_accepted_per_iter` tokens, so:

    speedup(K) = mean_accepted_per_iter * T_BASE / T_iter(K)

The interesting consequence: T_iter rises linearly in K while mean_accepted_per_iter saturates (acceptance
falls as K grows), so speedup has an interior optimum — measured at K=2 on this rig. Change the
draft/target pair or tighten the orchestrator (C++ BLS) and you re-fit two numbers.
"""
from __future__ import annotations

# --- constants back-solved from notebook 0016 (greedy, mixed, logits acceptance) ---
T_BASE = 13.39    # ms/token — plain fp16 1.5B, batch=1 (0016 baseline = 74.7 tok/s -> 1000/74.7)
T_FIXED = 16.6    # ms — intercept: target verify + orchestration
T_DRAFT = 5.59    # ms — slope: one 0.5B draft decode step
BASELINE_TPUT = 74.7  # tok/s

# measured 0016 A-sweep:  K -> (acceptance_rate, mean_accepted_per_iter, throughput_tok_s)
MEASURED = {
    1: (0.803, 1.792, 81.3),
    2: (0.724, 2.438, 87.8),
    4: (0.588, 3.336, 85.1),
    6: (0.469, 3.788, 75.4),
    8: (0.391, 4.105, 67.1),
}


def t_iter(k: int) -> float:
    """Modeled per-iteration wall time (ms) for draft_len K."""
    return T_FIXED + T_DRAFT * k


def speedup(k: int, mean_accepted_per_iter: float) -> float:
    """Modeled speedup vs the plain target baseline, given the measured tokens-advanced-per-iter."""
    return mean_accepted_per_iter * T_BASE / t_iter(k)


def best_k() -> int:
    """The draft_len that maximizes modeled speedup over the measured K grid."""
    return max(MEASURED, key=lambda k: speedup(k, MEASURED[k][1]))


if __name__ == "__main__":
    print(f"baseline {BASELINE_TPUT} tok/s ({T_BASE:.2f} ms/token);  T_iter(K) = {T_FIXED} + {T_DRAFT}*K ms")
    print(f"{'K':>2}  {'accept':>6}  {'acc/iter':>8}  {'T_iter ms':>9}  {'model x':>7}  {'measured x':>10}")
    for k, (acc, mai, tput) in MEASURED.items():
        print(f"{k:>2}  {acc:>6.3f}  {mai:>8.3f}  {t_iter(k):>9.2f}  {speedup(k, mai):>7.3f}  {tput/BASELINE_TPUT:>10.3f}")
    print(f"optimal draft_len K = {best_k()}  (measured optimum)")
