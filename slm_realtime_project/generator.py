import torch
from typing import Generator
from model import RealtimeSLM

class RealtimeGenerator:
    def __init__(self, model: RealtimeSLM, tokenizer, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def stream_generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> Generator[str, None, None]:
        prompt_tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        
        bsz, seqlen = tokens.shape
        start_pos = 0

        kv_caches = []
        for _ in range(self.model.n_layers):
            k_cache = torch.zeros((bsz, self.model.max_seq_len, self.model.layers[0].attention.n_kv_heads, self.model.head_dim), device=self.device)
            v_cache = torch.zeros((bsz, self.model.max_seq_len, self.model.layers[0].attention.n_kv_heads, self.model.head_dim), device=self.device)
            kv_caches.append((k_cache, v_cache))

        logits, kv_caches = self.model(tokens, start_pos=0, kv_caches=kv_caches)
        start_pos = seqlen
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        for _ in range(max_new_tokens):
            token_text = self.tokenizer.decode(next_token.squeeze(0).tolist())
            yield token_text

            logits, kv_caches = self.model(next_token, start_pos=start_pos, kv_caches=kv_caches)
            start_pos += 1

            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == getattr(self.tokenizer, 'eos_id', -1):
                break
