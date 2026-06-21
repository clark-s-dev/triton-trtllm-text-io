"""Incremental (streaming) detokenization — the project centerpiece.

Naive streaming decodes each new token alone: `tokenizer.decode([tok])`. That is
WRONG, because tokens do not map to characters. A single CJK glyph (3 UTF-8 bytes)
or emoji (4 bytes) is often split across two byte-level-BPE tokens, so decoding one
token alone yields an incomplete byte sequence -> the U+FFFD replacement char (`�`).

The correct algorithm (the one HF `TextIteratorStreamer` and vLLM
`detokenize_incrementally` use) keeps a running token buffer and a (prefix_offset,
read_offset) window. On each new token it decodes the window and emits only the
*newly completed* text — and if the freshly decoded text ends in `�`, the trailing
bytes are still incomplete, so it emits NOTHING and waits for the next token.

This works for both byte-level BPE (Qwen, Llama 3, GPT) and SentencePiece (Llama 2,
T5), because `convert_tokens_to_string` handles byte-merging and the `▁` space marker.
"""

from __future__ import annotations

_REPLACEMENT_CHAR = "�"  # `�` — signals incomplete trailing UTF-8 bytes


class IncrementalDetokenizer:
    """Feed token ids as they stream in; get back correct text deltas.

    Usage:
        detok = IncrementalDetokenizer(tokenizer)
        for step_ids in engine_stream:      # ids produced this decode step
            text_delta = detok.add(step_ids)
            if text_delta:
                yield text_delta
        tail = detok.flush()                # drain anything held back at the end
    """

    def __init__(self, tokenizer, skip_special_tokens: bool = True) -> None:
        self.tok = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.tokens: list[str] = []   # token *strings* seen so far
        self.prefix_offset = 0        # window start (already-emitted boundary)
        self.read_offset = 0          # last committed token boundary

    def add(self, new_token_ids) -> str:
        """Append newly generated token id(s); return the text safe to emit now."""
        parts = [self._step(int(tid)) for tid in new_token_ids]
        return "".join(parts)

    def _step(self, token_id: int) -> str:
        pieces = self.tok.convert_ids_to_tokens(
            [token_id], skip_special_tokens=self.skip_special_tokens
        )
        pieces = [p for p in pieces if p is not None]  # specials may be dropped
        if not pieces:
            return ""
        self.tokens.extend(pieces)

        prefix_text = self.tok.convert_tokens_to_string(
            self.tokens[self.prefix_offset:self.read_offset]
        )
        new_text = self.tok.convert_tokens_to_string(self.tokens[self.prefix_offset:])

        # Emit only if new text appeared AND it does not end on incomplete bytes.
        if len(new_text) > len(prefix_text) and not new_text.endswith(_REPLACEMENT_CHAR):
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.tokens)
            return new_text[len(prefix_text):]
        return ""  # incomplete multibyte char in flight — wait for the next token

    def flush(self) -> str:
        """Emit any remaining text at end-of-stream (incomplete bytes -> `�`)."""
        prefix_text = self.tok.convert_tokens_to_string(
            self.tokens[self.prefix_offset:self.read_offset]
        )
        full_text = self.tok.convert_tokens_to_string(self.tokens[self.prefix_offset:])
        self.prefix_offset = self.read_offset = len(self.tokens)
        return full_text[len(prefix_text):]
