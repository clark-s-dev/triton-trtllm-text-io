"""finish_reason classification tests — pure Python, no deps, no GPU.

Mirrors the BLS gateway's labeling: the same `classify_finish_reason` the server
calls is exercised here, so "live behavior == tested behavior" holds for the
finish_reason as it already does for detok + stop.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from text_io.finish import classify_finish_reason  # noqa: E402


def test_eos_before_budget_is_stop():
    # engine halted on its own, well under the budget -> end-of-turn (EOS)
    assert classify_finish_reason(stop_string_hit=False, generated=7, max_tokens=256) == "stop"


def test_budget_exhausted_is_length():
    # emitted exactly the budget, no stop string -> ran out of room
    assert classify_finish_reason(stop_string_hit=False, generated=256, max_tokens=256) == "length"


def test_stop_string_always_wins():
    # a client STOP string matched -> "stop", even if the budget was also reached
    assert classify_finish_reason(stop_string_hit=True, generated=256, max_tokens=256) == "stop"
    assert classify_finish_reason(stop_string_hit=True, generated=3, max_tokens=256) == "stop"


def test_over_budget_is_length():
    # defensive: a trailing EOS token can push the count up to/over the budget;
    # hitting the budget is still "length", never an accidental "stop".
    assert classify_finish_reason(stop_string_hit=False, generated=257, max_tokens=256) == "length"


if __name__ == "__main__":
    test_eos_before_budget_is_stop()
    test_budget_exhausted_is_length()
    test_stop_string_always_wins()
    test_over_budget_is_length()
    print("finish: all tests passed ✅")
