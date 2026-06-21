"""Stop-sequence detection across token boundaries.

A stop string (e.g. `</tool_call>`) rarely aligns to token boundaries and can be
split across streamed text deltas. So we detect it in the *decoded text*, not in
token space, and we hold back a tail of the buffer that could still grow into a
stop string — otherwise we would stream out characters that should have been
truncated.
"""

from __future__ import annotations


class StopSequenceMatcher:
    """Feed text deltas; get back (emittable_text, stop_hit).

    On a hit, everything from the stop string onward is dropped and `stop_hit` is
    True. Until then, a suffix of the buffer that is a prefix of some stop string
    is held back (it might complete into a stop on the next delta).
    """

    def __init__(self, stop_sequences=None) -> None:
        self.stops = [s for s in (stop_sequences or []) if s]
        self.buffer = ""

    def feed(self, text: str):
        if not self.stops:
            return text, False
        self.buffer += text

        # 1) complete stop present? emit up to it, drop the rest, signal stop.
        hit_idx = None
        for s in self.stops:
            i = self.buffer.find(s)
            if i != -1 and (hit_idx is None or i < hit_idx):
                hit_idx = i
        if hit_idx is not None:
            emit = self.buffer[:hit_idx]
            self.buffer = ""
            return emit, True

        # 2) no complete stop: hold back the longest suffix that is a stop prefix.
        hold = self._partial_suffix_len()
        if hold:
            emit, self.buffer = self.buffer[:-hold], self.buffer[-hold:]
        else:
            emit, self.buffer = self.buffer, ""
        return emit, False

    def _partial_suffix_len(self) -> int:
        best = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buffer)), 0, -1):
                if self.buffer.endswith(s[:k]):
                    best = max(best, k)
                    break
        return best

    def flush(self) -> str:
        """Emit the held-back tail at end-of-stream (no stop ever completed)."""
        out, self.buffer = self.buffer, ""
        return out
