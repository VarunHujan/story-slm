"""
=============================================================================
Story-SLM2: Streaming Text Generator (generator.py)
=============================================================================
Token-by-token autoregressive sampling with temperature, top-k, and
repetition penalty. Uses a pre-allocated KV cache (model.py's KVCache): the
prompt is encoded once, then each new token only computes attention against
the cached past -- written into a fixed buffer in place, not regrown via
concatenation every step -- instead of recomputing the whole sequence.
=============================================================================
"""

import torch
import torch.nn.functional as F
from typing import Generator

from utility_tools import try_answer_deterministically


class Karna49MGenerator:
    def __init__(self, model, tokenizer, device: str):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.75,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
    ) -> Generator[str, None, None]:
        # Dates and arithmetic are things no from-scratch small model can reliably
        # know or compute -- verified directly (asked to echo an injected real date,
        # it got the month/day right but hallucinated the wrong year). Answer these
        # deterministically with real code instead of trusting the neural network.
        deterministic_answer = try_answer_deterministically(prompt)
        if deterministic_answer is not None:
            yield deterministic_answer
            return

        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer.encode(formatted_prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        max_seq_len = self.model.config.max_seq_len
        stop_ids = {self.tokenizer.eos_id, self.tokenizer.im_start_id, self.tokenizer.im_end_id}

        # The model was only ever trained on positions [0, max_seq_len). If the prompt
        # alone exceeds that, keep just the most recent window rather than feeding it
        # positions it never learned (same truncation the old full-recompute path used).
        if x.size(1) > max_seq_len:
            x = x[:, -max_seq_len:]

        # Tracks which vocab ids have appeared so far as a boolean mask (O(vocab_size),
        # constant cost) instead of a growing Python set re-scanned every step. At a
        # ~1200-token story the old set-based loop measured ~47ms/step -- nearly 2x the
        # model's own forward-pass cost -- purely from Python-level per-token overhead.
        vocab_size = self.model.config.vocab_size
        seen_mask = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        seen_mask[x[0]] = True

        kv_cache = self.model.new_kv_cache(batch_size=1, device=self.device, dtype=next(self.model.parameters()).dtype)
        next_input = x

        for _ in range(max_new_tokens):
            logits, _ = self.model(next_input, kv_cache=kv_cache)
            logits = logits[0, -1, :] / max(temperature, 1e-5)

            penalized = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
            logits = torch.where(seen_mask, penalized, logits)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            if token_id in stop_ids:
                break

            seen_mask[token_id] = True
            yield self.tokenizer.decode([token_id])

            # Cache is now full to the model's trained context limit -- stop instead of
            # feeding a position index the model never saw during training.
            if kv_cache.length >= max_seq_len:
                break

            next_input = next_token.view(1, 1)


# Backward-compatibility alias
StorySLM2Generator = Karna49MGenerator

