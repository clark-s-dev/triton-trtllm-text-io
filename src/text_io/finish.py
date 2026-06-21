"""finish_reason classification — why did generation end?

Part of the streaming post-processing math (single source of truth in `text_io`),
so the Triton BLS gateway and the GPU-free unit tests agree on the label.

The TRT-LLM engine streams token deltas but does not hand the gateway a tidy
"reason"; we infer it from what we observed:

    - a client STOP string matched in the decoded text      -> "stop"
    - the engine halted before the token budget (end-of-turn / EOS)  -> "stop"
    - the engine ran the budget out (max_tokens tokens emitted)      -> "length"

This mirrors OpenAI semantics, where both a natural end-of-turn and a stop
sequence collapse to "stop", and only exhausting the budget is "length".
Guardrail blocks are reported as "content_filter" by the gateway *before* this
point, so they are intentionally out of scope here — this answers strictly why
the engine's generation terminated.
"""

from __future__ import annotations

STOP = "stop"
LENGTH = "length"


def classify_finish_reason(*, stop_string_hit: bool, generated: int, max_tokens: int) -> str:
    """Return the OpenAI-style finish_reason for a completed generation.

    Keyword-only on purpose: `generated` and `max_tokens` are both ints, so
    positional calls are easy to transpose.
    """
    if stop_string_hit or generated < max_tokens:
        return STOP
    return LENGTH
