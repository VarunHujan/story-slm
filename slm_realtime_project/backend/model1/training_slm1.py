import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import numpy as np
import tiktoken

# =============================================================================
# ✂️ Step 1: Self-Contained Tokenizer Engine
# =============================================================================
class StoryTikTokenizer:
    def __init__(self):
        self.base_enc = tiktoken.get_encoding("cl100k_base")
        self.special_tokens = {
            "<|im_start|>": 100258,
            "<|im_end|>": 100259,
            "<|endoftext|>": 100260,
            "<|edit_chapter|>": 100261
        }
        self.vocab_size = 32768

    def encode(self, text: str, allowed_special: Union[str, set] = "all") -> List[int]:
        tokens = self.base_enc.encode(text, allowed_special=allowed_special)
        return [t if t < self.vocab_size else (t % self.vocab_size) for t in tokens]

    def decode(self, tokens: List[int]) -> str:
        valid_tokens = [t if t < self.vocab_size else (t % self.vocab_size) for t in tokens]
        return self.base_enc.decode(valid_tokens)

# =============================================================================
# 🧠 Step 2: Self-Contained 50M Transformer Model Backbone
# =============================================================================
@dataclass
class StorySLMConfig:
    vocab_size: int = 32768
    d_model: int = 384
    n_layers: int = 20
    n_heads: int = 12
    n_kv_heads: int = 3
    d_ffn: int = 1536
    max_seq_len: int = 16384
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-6
    weight_tying: bool = True

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 16384, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rotate_x = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotate_x * sin)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: StorySLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_queries_per_kv = config.n_heads // config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))
        k = apply_rotary_pos_emb(k, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.full((S, S), float('-inf'), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)

class SwiGLUFFN(nn.Module):
    def __init__(self, config: StorySLMConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ffn, bias=config.bias)
        self.w2 = nn.Linear(config.d_ffn, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, config.d_ffn, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: StorySLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class StorySLMModel(nn.Module):
    def __init__(self, config: StorySLMConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.d_model // config.n_heads, max_seq_len=config.max_seq_len)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        return total, total

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S = input_ids.shape
        x = self.tok_embeddings(input_ids)
        cos, sin = self.rope(x, S)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# =============================================================================
# ⚡ Step 3: Batch Sampling & Training Execution
# =============================================================================
def get_batch(data_bytes, batch_size, seq_len, device):
    """
    Sample a random batch of token sequences from memory-mapped binary file data.
    """
    max_idx = len(data_bytes) - seq_len - 1
    if max_idx <= 0:
        tokens = np.tile(data_bytes, (seq_len // len(data_bytes) + 2))
        ix = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
        x_np = np.stack([tokens[i:i+seq_len] for i in ix])
        y_np = np.stack([tokens[i+1:i+1+seq_len] for i in ix])
    else:
        ix = torch.randint(0, max_idx, (batch_size,))
        x_np = np.stack([data_bytes[i:i+seq_len] for i in ix])
        y_np = np.stack([data_bytes[i+1:i+1+seq_len] for i in ix])
        
    x = torch.from_numpy(x_np.astype(np.int64)).to(device)
    y = torch.from_numpy(y_np.astype(np.int64)).to(device)
    return x, y

def get_lr(it, max_iters, warmup_iters, max_lr, min_lr):
    """
    Cosine learning rate decay schedule with linear warmup.
    """
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train():
    print("=== Starting Story-SLM 50M Self-Contained Training Initialization ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Target Execution Device: {device}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Training Hyperparameters
    micro_batch_size = 8    # Reduced micro-batch size for GPU VRAM safety
    grad_accum_steps = 2    # Effective batch size = 8 * 2 = 16
    seq_len = 512            # Context length per sequence
    max_iters = 5000         # Total training steps
    warmup_iters = 200       # Warmup steps
    max_lr = 5e-4            # Peak learning rate
    min_lr = 5e-5            # Minimum learning rate after decay
    weight_decay = 0.1       # AdamW weight decay
    grad_clip = 1.0          # Gradient clipping norm limit
    eval_interval = 250      # Steps between evaluation
    eval_iters = 10          # Reduced eval iterations to save VRAM
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load Memory-Mapped Binary Data
    train_bin_path = "train.bin"
    val_bin_path = "val.bin"
    
    if not os.path.exists(train_bin_path):
        print(f"'{train_bin_path}' not found! Automatically downloading TinyStories dataset from HuggingFace...")
        try:
            from datasets import load_dataset
            tokenizer = StoryTikTokenizer()
            ds = load_dataset('roneneldan/TinyStories', split='train[:10000]')
            all_tokens = []
            for row in ds:
                text = f"<|im_start|>user\nGenerate a story\n<|im_end|>\n<|im_start|>assistant\n{row['text']}<|im_end|>"
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
            arr = np.array(all_tokens, dtype=np.uint32)
            arr.tofile(train_bin_path)
            print(f"Successfully generated '{train_bin_path}' with {len(arr):,} tokens!")
        except Exception as e:
            print(f"Downloading TinyStories failed ({e}), generating synthetic training sequence...")
            dummy_tokens = np.random.randint(0, 32768, size=(100000,), dtype=np.uint32)
            dummy_tokens.tofile(train_bin_path)
            print(f"Created synthetic '{train_bin_path}' with 100,000 tokens!")
    
    train_data = np.memmap(train_bin_path, dtype=np.uint32, mode='r')
    val_data = np.memmap(val_bin_path, dtype=np.uint32, mode='r') if os.path.exists(val_bin_path) else train_data
    
    print(f"Loaded train.bin: {len(train_data):,} tokens")
    print(f"Loaded val.bin: {len(val_data):,} tokens")
    
    # Initialize Model & Tokenizer
    config = StorySLMConfig(
        vocab_size=32768,
        d_model=384,
        n_layers=20,
        n_heads=12,
        n_kv_heads=3,
        max_seq_len=2048
    )
    model = StorySLMModel(config).to(device)
    tokenizer = StoryTikTokenizer()
    
    total_params, _ = model.count_parameters()
    print(f"Initialized StorySLM Model with {total_params:,} Total Parameters")
    
    # Configure AdamW Optimizer with Weight Decay
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    
    print("\n--- Starting Training Loop ---")
    start_time = time.time()
    
    for it in range(max_iters):
        lr = get_lr(it, max_iters, warmup_iters, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if it % eval_interval == 0 or it == max_iters - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(eval_iters):
                    x_v, y_v = get_batch(val_data, micro_batch_size, seq_len, device)
                    _, v_loss = model(x_v, y_v)
                    val_losses.append(v_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                
            print(f"\n[STEP {it:5d}/{max_iters}] Validation Loss: {avg_val_loss:.4f} | Learning Rate: {lr:.6f}")
            
            ckpt_path = os.path.join(checkpoint_dir, f"story_slm_50m_step_{it}.pt")
            torch.save({
                'step': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config,
            }, ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")
            model.train()
            
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = get_batch(train_data, micro_batch_size, seq_len, device)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16 if device == 'cuda' else torch.float32):
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
            accum_loss += loss.item()
            scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        if it % 50 == 0:
            tok_per_sec = (50 * micro_batch_size * grad_accum_steps * seq_len) / (time.time() - start_time) if it > 0 else 0
            start_time = time.time()
            print(f"Step {it:5d}/{max_iters} | Loss: {accum_loss:.4f} | Speed: {tok_per_sec:.0f} tok/s")

    print("\n===\nStory-SLM 50M Model Training Complete! Saved Checkpoints in ./checkpoints/\n===")

if __name__ == '__main__':
    train()
