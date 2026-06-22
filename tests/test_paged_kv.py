"""Tests for the M3 toy paged-KV allocator (lab/paged_kv.py) — pure Python, no GPU.

These pin the calibration against the real engine measured in lab-notebook 0004
(KV-cache reuse): a 3968-token shared prefix is 62 blocks, and over N requests the
engine reported `reused = 2418` KV blocks = (N-1) x 62 with N=40. The toy must
reproduce that to the block, plus the refcount / prefix-property / LRU mechanics that
make it true.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab"))

from paged_kv import (  # noqa: E402
    PagedKVAllocator, TOKENS_PER_BLOCK, run_shared_prefix, shared_prefix_workload,
)


def test_reuse_reproduces_engine_2418():
    # The headline 0004 calibration: 40 requests, 3968-token (62-block) shared prefix.
    stats = run_shared_prefix(n_requests=40, prefix_tokens=3968, tail_tokens=32)
    assert stats.reused_blocks == 2418           # exactly (40-1) x 62, the engine's `reused`
    assert abs(stats.hit_rate - 0.975) < 0.001   # 2418 / (40 x 62) = 97.5%


def test_reuse_off_is_zero():
    # enable_kv_cache_reuse=false -> every prefix recomputed, `reused` stays 0.
    stats = run_shared_prefix(n_requests=40, enable_reuse=False)
    assert stats.reused_blocks == 0
    assert stats.hit_rate == 0.0


def test_first_request_is_cold_then_warm():
    # "First request paves the road, the rest ride free" (0004 §7).
    alloc = PagedKVAllocator(total_blocks=4096)
    reqs = shared_prefix_workload(n_requests=3, prefix_tokens=3968, tail_tokens=32)
    r0 = alloc.allocate(0, reqs[0]); alloc.free(0)
    assert r0.reused_blocks == 0                 # cold: nothing cached yet
    r1 = alloc.allocate(1, reqs[1]); alloc.free(1)
    assert r1.reused_blocks == 62                # warm: the whole 62-block prefix hits


def test_prefix_property_stops_at_first_divergence():
    # Reuse must stop at the first block that differs; you cannot reuse block i if
    # block i-1 already diverged (that is what makes the hash a *prefix* identity).
    tpb = TOKENS_PER_BLOCK
    alloc = PagedKVAllocator(total_blocks=512)
    base = [(i % 500) + 5 for i in range(4 * tpb)]      # 4 full blocks
    alloc.allocate(0, base); alloc.free(0)
    # Identical for 2 blocks, then diverges in block 3.
    other = base[:2 * tpb] + [99999] * (2 * tpb)
    res = alloc.allocate(1, other)
    assert res.reused_blocks == 2                        # only the 2 matching leading blocks


def test_refcount_shares_one_physical_copy():
    # Two concurrent requests with the same prefix share ONE physical set of blocks
    # (refcount 2), not two copies.
    tpb = TOKENS_PER_BLOCK
    alloc = PagedKVAllocator(total_blocks=512)
    prompt = [(i % 500) + 5 for i in range(10 * tpb)]   # 10 full blocks
    alloc.allocate(0, prompt)                            # seq 0 holds 10 blocks
    used_after_first = alloc.used_blocks
    alloc.allocate(1, prompt)                            # seq 1 reuses the same 10
    assert alloc.used_blocks == used_after_first         # no new physical blocks consumed
    assert alloc._blocks[alloc._cache[alloc._prompt_block_hashes(prompt)[0]]].ref_count == 2


def test_partial_trailing_block_not_reused():
    # Only FULL blocks are hashed/cached; a partial trailing block is never reused.
    tpb = TOKENS_PER_BLOCK
    alloc = PagedKVAllocator(total_blocks=512)
    prompt = [(i % 500) + 5 for i in range(2 * tpb + 10)]   # 2 full + 1 partial(10)
    alloc.allocate(0, prompt); alloc.free(0)
    res = alloc.allocate(1, prompt)
    assert res.reused_blocks == 2                            # the partial 3rd block is fresh


def test_free_keeps_cached_blocks_reusable():
    # free() decrements refcount; refcount-0 cached blocks STAY reusable (not reclaimed)
    # so a later request still hits them. This is the crux of cross-request reuse.
    tpb = TOKENS_PER_BLOCK
    alloc = PagedKVAllocator(total_blocks=512)
    prompt = [(i % 500) + 5 for i in range(5 * tpb)]
    alloc.allocate(0, prompt)
    alloc.free(0)                                       # refcount -> 0, but still cached
    res = alloc.allocate(1, prompt)
    assert res.reused_blocks == 5                       # full hit after the holder freed


def test_lru_eviction_under_pressure_then_recovers():
    # A pool too small for the working set must evict refcount-0 cached blocks (LRU) to
    # serve new requests — without crashing or leaking blocks.
    tpb = TOKENS_PER_BLOCK
    alloc = PagedKVAllocator(total_blocks=20)            # tiny pool
    for i in range(30):                                  # 30 distinct 8-block requests
        ids = [10_000 * i + j for j in range(8 * tpb)]   # all-unique -> no reuse, forces churn
        res = alloc.allocate(i, ids)
        assert res.ok                                    # eviction keeps it serviceable
        alloc.free(i)
    assert alloc.stats.evictions > 0                     # pressure really did force eviction
    # No leak: nothing still live, and every block is either free or cached (reclaimable).
    assert sum(b.ref_count for b in alloc._blocks) == 0
    cached = sum(1 for b in alloc._blocks if b.content_hash is not None)
    assert alloc.free_blocks + cached == alloc.total_blocks


def test_oom_when_all_blocks_live():
    # If every block is held by a live (un-freed) sequence, a new alloc must fail cleanly
    # (ok=False) — this is the signal the scheduler turns into queue/preempt.
    alloc = PagedKVAllocator(total_blocks=4)
    big = list(range(4 * TOKENS_PER_BLOCK))              # needs all 4 blocks, kept live
    assert alloc.allocate(0, big).ok
    res = alloc.allocate(1, [9] * TOKENS_PER_BLOCK)      # nothing evictable -> OOM
    assert res.ok is False


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in ALL:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(ALL)} paged-KV tests passed")
