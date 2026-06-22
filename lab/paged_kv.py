#!/usr/bin/env python3
"""paged_kv.py — M3 toy: a paged KV-cache block allocator with prefix reuse.

The L2-LAB §7 artifact for M3. A *minimal, GPU-free* model of what TensorRT-LLM's
KV block manager (and vllm/v1/core/kv_cache_manager.py) does, built to be calibrated
against numbers measured on the real engine in lab-notebook 0004 (KV-cache reuse) and
0007 (kv_cache_free_gpu_mem_fraction).

What it models (and what the engine actually does):
  * Paged KV — the cache is a pool of fixed-size **blocks** of `tokens_per_block`
    tokens (engine measured 64; see 0001). A sequence's KV is a *block table*, not a
    contiguous slab, so a 4000-token prompt is 63 blocks, not one allocation.
  * Prefix reuse — a block is keyed by a **rolling hash of the whole prefix up to and
    including it** (parent-hash chained), exactly so that two requests sharing a prefix
    hash their leading blocks identically and the second request can skip recomputing
    them. Reuse stops at the first block that differs (the prefix property): you cannot
    reuse block i if block i-1 already diverged. This is the mechanism behind the
    engine's `reused` KV-block metric.
  * Reference counting — a reused block is shared by N live sequences; its refcount is
    how many. Freeing a sequence decrements, it does not reclaim: a refcount-0 block
    stays cached and reusable (that is the whole point — "the first request paves the
    road, the rest ride free", 0004 §7).
  * LRU eviction — physical blocks are reclaimed only under pressure: when a new block
    is needed and the free list is empty, evict the least-recently-used refcount-0
    cached block. If every block is still live (refcount > 0) the allocation fails
    (OOM) — that is the signal the *scheduler* (cbatch_sim.py) turns into a queue /
    preemption decision.

What it deliberately does NOT model (the gap that is itself a lesson, L2-LAB §7):
  * Partial trailing blocks are never reuse-cached (only full blocks get a stable hash),
    matching real engines that hash complete blocks only.
  * No sliding-window / no chunked-prefill interaction, no swap-to-CPU — see notebook
    0014 §5 for why those gaps matter.

Run it: `python3 lab/paged_kv.py` reproduces the 0004 shared-prefix calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

TOKENS_PER_BLOCK = 64  # engine-measured (lab-notebook 0001: 190656/2979 = 64.0)

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1


def _block_hash(parent_hash: int, token_ids: tuple[int, ...]) -> int:
    """Stable 64-bit FNV-1a over (parent_hash, *token_ids).

    Chaining the parent hash is what makes the key a *prefix* identity: block i's
    hash depends on blocks 0..i, so identical hashes ⇒ identical prefixes. Stable
    across processes (unlike salted built-in hash) so tests are deterministic.
    """
    h = _FNV_OFFSET
    for v in (parent_hash & _MASK64, *token_ids):
        h ^= v & _MASK64
        h = (h * _FNV_PRIME) & _MASK64
    return h


@dataclass
class _Block:
    block_id: int
    content_hash: Optional[int] = None  # None = a partial/uncached block
    ref_count: int = 0
    last_used: int = 0  # logical tick, for LRU among refcount-0 cached blocks


@dataclass
class AllocResult:
    """Outcome of allocating one sequence's prompt blocks."""
    seq_id: int
    block_ids: list[int]
    reused_blocks: int       # leading blocks served from cache (the `reused` metric)
    new_blocks: int          # blocks freshly allocated (cold prefill work)
    prefix_blocks: int       # full blocks examined for reuse (reused + cold-prefix new)
    ok: bool = True          # False ⇒ pool exhausted, all blocks live (OOM)


@dataclass
class Stats:
    reused_blocks: int = 0       # cumulative, matches engine `reused` counter
    new_blocks: int = 0          # cumulative cold allocations
    prefix_lookups: int = 0      # cumulative full-prefix blocks examined for reuse
    evictions: int = 0
    alloc_failures: int = 0

    @property
    def hit_rate(self) -> float:
        """Block-level prefix cache hit rate = reused / prefix-blocks-examined."""
        return self.reused_blocks / self.prefix_lookups if self.prefix_lookups else 0.0


class PagedKVAllocator:
    """A pool of `total_blocks` KV blocks with prefix reuse + LRU eviction.

    `enable_reuse=False` reproduces the `enable_kv_cache_reuse=false` ablation
    (notebook 0004): every prefix block is recomputed, `reused` stays 0.
    """

    def __init__(self, total_blocks: int, tokens_per_block: int = TOKENS_PER_BLOCK,
                 enable_reuse: bool = True):
        self.total_blocks = total_blocks
        self.tokens_per_block = tokens_per_block
        self.enable_reuse = enable_reuse
        self._blocks = [_Block(i) for i in range(total_blocks)]
        self._free: list[int] = list(range(total_blocks))            # free block ids
        self._cache: dict[int, int] = {}                             # content_hash -> block_id
        self._seq_blocks: dict[int, list[int]] = {}                  # seq_id -> block ids it holds
        self._tick = 0
        self.stats = Stats()

    # ---- capacity helpers -------------------------------------------------
    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - len(self._free)

    def n_blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.tokens_per_block - 1) // self.tokens_per_block

    # ---- core: hash a prompt into block keys ------------------------------
    def _prompt_block_hashes(self, token_ids: list[int]) -> list[int]:
        """Rolling prefix hashes for each *full* block of the prompt."""
        tpb = self.tokens_per_block
        n_full = len(token_ids) // tpb
        hashes, parent = [], 0
        for b in range(n_full):
            chunk = tuple(token_ids[b * tpb:(b + 1) * tpb])
            parent = _block_hash(parent, chunk)
            hashes.append(parent)
        return hashes

    # ---- physical block acquisition (with LRU eviction) -------------------
    def _take_free_block(self) -> Optional[int]:
        if self._free:
            return self._free.pop()
        # Pool empty: evict the LRU refcount-0 cached block, if any.
        victim = self._lru_evictable()
        if victim is None:
            return None  # everything live -> caller reports OOM
        blk = self._blocks[victim]
        if blk.content_hash is not None:
            self._cache.pop(blk.content_hash, None)
        blk.content_hash = None
        self.stats.evictions += 1
        return victim

    def _lru_evictable(self) -> Optional[int]:
        best, best_tick = None, None
        for blk in self._blocks:
            if blk.ref_count == 0 and blk.content_hash is not None:
                if best_tick is None or blk.last_used < best_tick:
                    best, best_tick = blk.block_id, blk.last_used
        return best

    # ---- allocate a sequence's prompt -------------------------------------
    def allocate(self, seq_id: int, token_ids: list[int]) -> AllocResult:
        """Allocate KV blocks for `seq_id`'s prompt, reusing cached prefix blocks.

        Reuse walks the prompt's full blocks in order, reusing each that is already
        cached, and STOPS at the first miss (prefix property). The remaining full
        blocks + the partial trailing block are freshly allocated.
        """
        self._tick += 1
        prompt_hashes = self._prompt_block_hashes(token_ids) if self.enable_reuse else []
        n_blocks_total = self.n_blocks_for(len(token_ids))
        n_full = len(token_ids) // self.tokens_per_block

        held: list[int] = []
        reused = 0
        still_matching = True
        for i in range(n_blocks_total):
            is_full = i < n_full
            h = prompt_hashes[i] if (is_full and i < len(prompt_hashes)) else None
            if still_matching and h is not None and h in self._cache:
                bid = self._cache[h]                 # cache HIT — share this block
                blk = self._blocks[bid]
                blk.ref_count += 1
                blk.last_used = self._tick
                held.append(bid)
                reused += 1
                continue
            still_matching = False                   # first miss ends reuse (prefix property)
            bid = self._take_free_block()
            if bid is None:                           # pool exhausted, all live
                self._release(held)                   # roll back this partial allocation
                self.stats.alloc_failures += 1
                return AllocResult(seq_id, [], 0, 0, n_full, ok=False)
            blk = self._blocks[bid]
            blk.ref_count = 1
            blk.last_used = self._tick
            if is_full and h is not None:             # only full blocks get cached/keyed
                blk.content_hash = h
                self._cache[h] = bid
            else:
                blk.content_hash = None               # partial trailing block, not reusable
            held.append(bid)

        self._seq_blocks[seq_id] = held
        new_blocks = len(held) - reused
        self.stats.reused_blocks += reused
        self.stats.new_blocks += new_blocks
        self.stats.prefix_lookups += n_full
        return AllocResult(seq_id, held, reused, new_blocks, n_full, ok=True)

    def append_block(self, seq_id: int) -> bool:
        """Grow a live sequence by one block (a decode step crossed a block boundary).

        Returns False on OOM (the scheduler then preempts/queues). The new block is a
        running KV block, not reuse-cached (its contents are still being generated).
        """
        bid = self._take_free_block()
        if bid is None:
            self.stats.alloc_failures += 1
            return False
        blk = self._blocks[bid]
        blk.ref_count = 1
        blk.last_used = self._tick
        blk.content_hash = None
        self._seq_blocks.setdefault(seq_id, []).append(bid)
        return True

    def free(self, seq_id: int) -> None:
        """Release a sequence's hold. Refcount-0 cached blocks STAY reusable."""
        self._release(self._seq_blocks.pop(seq_id, []))

    def _release(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            blk = self._blocks[bid]
            if blk.ref_count > 0:
                blk.ref_count -= 1
            # refcount-0 blocks are NOT reclaimed here: a cached one stays in self._cache
            # (reusable), an uncached one returns to the free list.
            if blk.ref_count == 0 and blk.content_hash is None:
                self._free.append(bid)


# ---------------------------------------------------------------------------
# Calibration harness: reproduce lab-notebook 0004 (shared-prefix KV reuse).
# ---------------------------------------------------------------------------
def shared_prefix_workload(n_requests: int, prefix_tokens: int, tail_tokens: int):
    """N requests that share an identical `prefix_tokens` prefix, each with a unique
    `tail_tokens` suffix — the 0004 workload (3968 shared + 32 unique = 4000)."""
    shared = [(i % 500) + 5 for i in range(prefix_tokens)]   # mirrors perf_benchmark.make_input_ids
    reqs = []
    for r in range(n_requests):
        tail = [600 + (r * 7919 + j) % 150000 for j in range(tail_tokens)]  # per-request unique
        reqs.append(shared + tail)
    return reqs


def run_shared_prefix(n_requests=40, prefix_tokens=3968, tail_tokens=32,
                      total_blocks=2979, enable_reuse=True, sequential=True):
    """Drive the allocator through the shared-prefix workload; return Stats.

    `sequential=True` frees each request before the next (closed loop, C=1): the cache
    survives the free, so request k>1 reuses the prefix request 1 paved.
    """
    alloc = PagedKVAllocator(total_blocks, enable_reuse=enable_reuse)
    reqs = shared_prefix_workload(n_requests, prefix_tokens, tail_tokens)
    for i, ids in enumerate(reqs):
        alloc.allocate(i, ids)
        if sequential:
            alloc.free(i)
    return alloc.stats


def _main() -> int:
    tpb = TOKENS_PER_BLOCK
    prefix_tokens, tail_tokens, n = 3968, 32, 40
    prefix_blocks = prefix_tokens // tpb
    print("M3 toy paged-KV allocator — calibration vs lab-notebook 0004\n")
    print(f"  workload: {n} requests, shared prefix {prefix_tokens} tok "
          f"= {prefix_blocks} blocks (tokens/block={tpb}), unique tail {tail_tokens} tok")

    on = run_shared_prefix(n, prefix_tokens, tail_tokens, enable_reuse=True)
    off = run_shared_prefix(n, prefix_tokens, tail_tokens, enable_reuse=False)
    print("\n  reuse ON :  reused blocks =", on.reused_blocks,
          f"  hit_rate = {on.hit_rate:.1%}")
    print("  reuse OFF:  reused blocks =", off.reused_blocks,
          f"  hit_rate = {off.hit_rate:.1%}")
    print(f"\n  model:  (N-1)x prefix_blocks = {(n-1)}x{prefix_blocks} = {(n-1)*prefix_blocks}")
    print("  engine (0004): reused = 2418 (ON), 0 (OFF)")
    ok = on.reused_blocks == 2418 and off.reused_blocks == 0
    print("\n  ✓ reproduces engine `reused` to the block" if ok else "\n  ✗ mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
