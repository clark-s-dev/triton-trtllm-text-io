"""text_io — single source of truth for the streaming pre/post math.

Imported by the Triton BLS gateway (model_repository/text_pipeline_bls) AND by the
laptop unit tests, so the server path and the tests run identical logic. Pure
Python, no GPU, no Triton — that is what makes the correctness proof (CJK/emoji
byte-exact streaming detok) runnable anywhere.
"""

from .detokenize_incremental import IncrementalDetokenizer
from .stop import StopSequenceMatcher

__all__ = ["IncrementalDetokenizer", "StopSequenceMatcher"]
