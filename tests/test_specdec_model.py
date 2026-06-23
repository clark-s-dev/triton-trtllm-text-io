"""Tests for the speculative-decoding speedup model (lab/specdec_model.py) — pure Python.

Calibration target (the §7 discipline, now for M4 / notebook 0016): the 2-constant model
T_iter(K) = T_FIXED + T_DRAFT*K must reproduce the *measured* per-K speedups of the 0016 sweep,
and the interior optimum at K=2. Same spirit as test_cbatch_sim.py: assert shape + ballpark, not
exact tok/s (engine numbers carry run-to-run noise).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))

from specdec_model import (  # noqa: E402
    t_iter, speedup, best_k, MEASURED, BASELINE_TPUT,
)


# ---- 0016 pass condition: the model reproduces every measured speedup within ~5% ----
def test_model_reproduces_measured_speedups():
    for k, (_acc, mai, tput) in MEASURED.items():
        measured = tput / BASELINE_TPUT
        modeled = speedup(k, mai)
        assert abs(modeled - measured) / measured < 0.05, \
            f"K={k}: model {modeled:.3f} vs measured {measured:.3f} (>5%)"


# ---- the interior optimum is K=2 (rising acc/iter vs linearly-rising T_iter) ----
def test_optimum_is_k2():
    assert best_k() == 2, f"optimal draft_len should be 2, got {best_k()}"
    # and it must actually beat its neighbours
    s = {k: speedup(k, MEASURED[k][1]) for k in MEASURED}
    assert s[2] > s[1] and s[2] > s[4]


# ---- K=8 is a *slowdown* on this rig (draft serial cost outruns the extra accepted tokens) ----
def test_large_k_is_a_slowdown():
    assert speedup(8, MEASURED[8][1]) < 1.0
    assert MEASURED[8][2] < BASELINE_TPUT          # measured tok/s below baseline too


# ---- per-iteration time is linear & monotonically increasing in K ----
def test_t_iter_linear_increasing():
    ks = sorted(MEASURED)
    for a, b in zip(ks, ks[1:]):
        assert t_iter(b) > t_iter(a)
    # constant slope (linearity): equal-K-gap deltas match
    assert abs((t_iter(4) - t_iter(2)) - (t_iter(6) - t_iter(4))) < 1e-6


# ---- acceptance falls as K grows (later guesses are harder) — a measured-data invariant ----
def test_acceptance_decreases_with_k():
    accs = [MEASURED[k][0] for k in sorted(MEASURED)]
    assert all(a > b for a, b in zip(accs, accs[1:])), f"acceptance should fall with K: {accs}"


# ---- mean-accepted/iter ≈ 1 + a*K (the speculation-gain identity) ----
def test_mean_accepted_matches_acceptance_identity():
    for k, (acc, mai, _t) in MEASURED.items():
        assert abs(mai - (1 + acc * k)) < 0.15, f"K={k}: acc/iter {mai} vs 1+a*K {1+acc*k:.2f}"


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in ALL:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(ALL)} speculative-decoding-model tests passed")
