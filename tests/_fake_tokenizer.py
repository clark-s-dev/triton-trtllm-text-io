"""A minimal byte-level-BPE tokenizer stand-in for GPU-free correctness tests.

It reproduces the one behavior that breaks naive streaming: a multi-byte UTF-8
character split across tokens decodes to `�` until all its bytes arrive. This lets
us prove the incremental detokenizer byte-exact on a laptop, no model download.
"""

from __future__ import annotations


def _bytes_to_unicode():
    """The GPT-2 byte<->unicode map, so tokens are printable like a real BBPE vocab."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class FakeByteLevelTokenizer:
    """`id_to_bytes`: {token_id: bytes}. Mirrors HF's convert_ids_to_tokens /
    convert_tokens_to_string for byte-level BPE, including `errors="replace"`."""

    def __init__(self, id_to_bytes: dict[int, bytes]) -> None:
        self._b2u = _bytes_to_unicode()
        self._u2b = {u: b for b, u in self._b2u.items()}
        self._id_to_token = {
            i: "".join(self._b2u[byte] for byte in bs) for i, bs in id_to_bytes.items()
        }

    def convert_ids_to_tokens(self, ids, skip_special_tokens: bool = False):
        return [self._id_to_token[i] for i in ids]

    def convert_tokens_to_string(self, tokens) -> str:
        data = bytes(self._u2b[ch] for tok in tokens for ch in tok)
        return data.decode("utf-8", errors="replace")

    # convenience for tests: build a per-token id list that splits a string's bytes
    @classmethod
    def from_byte_split(cls, raw: bytes, split_points):
        """Make a tokenizer + id list where `raw` is chopped at `split_points`."""
        cuts = [0, *split_points, len(raw)]
        chunks = [raw[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]
        id_to_bytes = {i: ch for i, ch in enumerate(chunks)}
        return cls(id_to_bytes), list(range(len(chunks)))
