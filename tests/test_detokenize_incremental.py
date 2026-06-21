"""Byte-exact streaming-detokenization proof — runs on a laptop, no GPU.

Shows the bug (naive per-token decode emits `�`) and that IncrementalDetokenizer
reconstructs CJK/emoji text byte-exactly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from text_io.detokenize_incremental import IncrementalDetokenizer  # noqa: E402
from _fake_tokenizer import FakeByteLevelTokenizer  # noqa: E402


# "你好🚀": 你=E4 BD A0, 好=E5 A5 BD, 🚀=F0 9F 9A 80. Split mid-character so every
# boundary lands inside a multi-byte glyph (the worst case for naive decoding).
RAW = "你好🚀".encode("utf-8")
SPLITS = [2, 4, 6, 8]  # -> tokens: [E4BD][A0E5][A5BD][F09F][9A80]


def test_incremental_is_byte_exact():
    tok, ids = FakeByteLevelTokenizer.from_byte_split(RAW, SPLITS)
    detok = IncrementalDetokenizer(tok, skip_special_tokens=False)

    streamed = "".join(detok.add([i]) for i in ids) + detok.flush()

    assert streamed == "你好🚀", repr(streamed)
    assert "�" not in streamed  # no mojibake


def test_naive_per_token_decode_is_broken():
    """Contrast: decoding each token alone corrupts the output."""
    tok, ids = FakeByteLevelTokenizer.from_byte_split(RAW, SPLITS)
    naive = "".join(tok.convert_tokens_to_string(tok.convert_ids_to_tokens([i])) for i in ids)

    assert naive != "你好🚀"
    assert "�" in naive  # the bug we fix


def test_ascii_unaffected():
    tok, ids = FakeByteLevelTokenizer.from_byte_split(b"Hello, world!", [1, 5, 7])
    detok = IncrementalDetokenizer(tok, skip_special_tokens=False)
    streamed = "".join(detok.add([i]) for i in ids) + detok.flush()
    assert streamed == "Hello, world!"


if __name__ == "__main__":  # allow `python3 tests/test_detokenize_incremental.py`
    test_incremental_is_byte_exact()
    test_naive_per_token_decode_is_broken()
    test_ascii_unaffected()
    print("detokenize_incremental: all tests passed ✅")
