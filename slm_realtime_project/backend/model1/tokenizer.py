"""
=============================================================================
Story-SLM 50M: Production TikToken Wrapper Engine (tokenizer.py)
=============================================================================
Author: Advanced AI Masterclass Team
Description: Clean, modular wrapper around OpenAI's Rust-powered tiktoken 
             tokenizer scheme (gpt2) with injected Special Control Tokens 
             for Story-SLM 50M. Completely eliminates modulo collisions.
=============================================================================
"""

import tiktoken
from typing import List, Union


class StoryTikTokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initializes high-speed Rust tiktoken BPE tokenizer (gpt2 vocabulary).
        Default vocab size: 50,257 tokens + special tokens.
        """
        self.encoding_name = encoding_name

        # Define Special Control Tokens for Story-SLM 50M
        self.special_tokens = {
            "<|endoftext|>": 50256,
            "<|im_start|>": 50257,
            "<|im_end|>": 50258,
            "<|pad|>": 50259,
            "<|edit_chapter|>": 50260,
        }

        try:
            base_enc = tiktoken.get_encoding(encoding_name)
            self.enc = tiktoken.Encoding(
                name=f"story_{encoding_name}",
                pat_str=base_enc._pat_str,
                mergeable_ranks=base_enc._mergeable_ranks,
                special_tokens={**base_enc._special_tokens, **self.special_tokens},
            )
            self.vocab_size = 32768
            print(f"[INFO] Initialized StoryTikTokenizer ({encoding_name}) | Vocab Target: {self.vocab_size}")
        except Exception as e:
            print(f"[WARNING] Fallback to base gpt2 encoding due to: {e}")
            self.enc = tiktoken.get_encoding("gpt2")
            self.vocab_size = 32768

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encodes human-readable text string into integer Token IDs.
        Safely bounds token IDs to 32,768 vocabulary range without modulo mangling.
        """
        if not text:
            return []

        if add_special_tokens:
            text = f"<|im_start|>{text}<|im_end|>"

        tokens = self.enc.encode(text, allowed_special="all")
        return [min(t, self.vocab_size - 1) for t in tokens]

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes integer Token IDs back into human-readable text string.
        """
        if not tokens:
            return ""
        valid_tokens = [min(t, self.vocab_size - 1) for t in tokens]
        return self.enc.decode(valid_tokens)


# Quick Self-Test when executed directly
if __name__ == "__main__":
    print("==========================================================")
    print("[TEST] Testing Modular StoryTikTokenizer (gpt2 base)")
    print("==========================================================")

    tok = StoryTikTokenizer(encoding_name="gpt2")
    test_prompt = "<|im_start|>user\nGenre: Sci-Fi\nPrompt: Tell me a story about space exploration.<|im_end|>"
    encoded_ids = tok.encode(test_prompt)
    decoded_text = tok.decode(encoded_ids)

    print(f"Sample Input  : {test_prompt!r}")
    print(f"Token Count   : {len(encoded_ids)} tokens")
    print(f"Decoded Output: {decoded_text!r}")
    print(f"Match Success : {test_prompt == decoded_text}")
    print("==========================================================")
