"""
=============================================================================
Story-SLM2: Production Tokenizer Engine (tokenizer.py)
=============================================================================
Clean wrapper around OpenAI's Rust tiktoken "gpt2" encoding (50,257 native
vocabulary) with 5 injected control tokens, matching exactly the encoding
used by build_master_dataset_100m.py to produce train_m2.bin / val_m2.bin.

No modulo wrapping. No clamping. Every raw byte has a fallback encoding, so
out-of-vocabulary crashes are structurally impossible.
=============================================================================
"""

import tiktoken
from typing import List

SPECIAL_TOKENS = {
    "<|endoftext|>": 50256,
    "<|im_start|>": 50257,
    "<|im_end|>": 50258,
    "<|pad|>": 50259,
    "<|reasoning_start|>": 50260,
    "<|reasoning_end|>": 50261,
}

# Highest valid token id (50261) + 1. This is the TRUE cardinality of the
# vocabulary actually present in train_m2.bin / val_m2.bin.
REAL_VOCAB_SIZE = 50262

# Embedding table size used by the model: REAL_VOCAB_SIZE padded up to the
# nearest multiple of 64 for GPU tensor-core efficiency. Rows above
# REAL_VOCAB_SIZE are simply never indexed.
EMBED_VOCAB_SIZE = 50304


class Karna49MTokenizer:
    def __init__(self):
        base = tiktoken.get_encoding("gpt2")
        self.enc = tiktoken.Encoding(
            name="karna_49m_gpt2",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={**base._special_tokens, **SPECIAL_TOKENS},
        )

        self.eos_id = SPECIAL_TOKENS["<|endoftext|>"]
        self.im_start_id = SPECIAL_TOKENS["<|im_start|>"]
        self.im_end_id = SPECIAL_TOKENS["<|im_end|>"]
        self.pad_id = SPECIAL_TOKENS["<|pad|>"]
        self.reasoning_start_id = SPECIAL_TOKENS["<|reasoning_start|>"]
        self.reasoning_end_id = SPECIAL_TOKENS["<|reasoning_end|>"]

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special="all")

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(list(ids))


# Backward-compatibility alias
StorySLM2Tokenizer = Karna49MTokenizer


if __name__ == "__main__":
    tok = Karna49MTokenizer()

    sample = (
        "<|im_start|>user\nTell me a story about a lighthouse.<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|reasoning_start|>The lighthouse should feel isolated but hopeful.<|reasoning_end|>\n"
        "Once upon a time, a lighthouse keeper watched the storm roll in.<|im_end|>"
    )
    ids = tok.encode(sample)
    back = tok.decode(ids)

    assert back == sample, "Round-trip mismatch! Tokenizer is not lossless."
    assert max(ids) < REAL_VOCAB_SIZE, f"Token id {max(ids)} exceeds expected vocabulary range!"

    print("[OK] Round-trip verified — encode/decode is lossless.")
    print(f"[OK] {len(ids)} tokens produced. Max id in sample: {max(ids)} (< {REAL_VOCAB_SIZE})")
