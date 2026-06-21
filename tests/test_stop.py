"""Cross-token-boundary stop-sequence tests — pure Python, no deps."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from text_io.stop import StopSequenceMatcher  # noqa: E402


def test_stop_split_across_deltas():
    m = StopSequenceMatcher(["</tool_call>"])
    out = []
    e, hit = m.feed("abc</tool")   # partial stop suffix held back
    out.append(e)
    assert not hit
    e, hit = m.feed("_call>def")   # stop completes; trailing "def" dropped
    out.append(e)
    assert hit
    assert "".join(out) == "abc"


def test_no_stop_passthrough():
    m = StopSequenceMatcher(["STOP"])
    e, hit = m.feed("hello ")
    # "hello " has no suffix that is a prefix of "STOP" -> fully emitted
    assert e == "hello " and not hit


def test_partial_then_flush():
    m = StopSequenceMatcher(["END"])
    e, hit = m.feed("the EN")        # "EN" is a prefix of "END" -> held back
    assert e == "the " and not hit
    assert m.flush() == "EN"          # never completed -> emit at end


def test_empty_stops_is_identity():
    m = StopSequenceMatcher([])
    e, hit = m.feed("anything </x>")
    assert e == "anything </x>" and not hit


if __name__ == "__main__":
    test_stop_split_across_deltas()
    test_no_stop_passthrough()
    test_partial_then_flush()
    test_empty_stops_is_identity()
    print("stop: all tests passed ✅")
