import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# Import our custom Story-SLM 50M model and tokenizer
from model import StorySLMModel, StorySLMConfig
from tokenizer import StoryTikTokenizer

def get_batch(data_bytes, batch_size, seq_len, device):
    """
    Sample a random batch of token sequences from memory-mapped binary file data.
    """
    max_idx = len(data_bytes) - seq_len - 1
    if max_idx <= 0:
        # Repeat small data for testing/fallback
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
    print("=== Starting Story-SLM 50M Training Script Initialization ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Target Execution Device: {device}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Training Hyperparameters
    batch_size = 16          # Per-step batch size
    seq_len = 512            # Context length per sequence
    max_iters = 5000         # Total training steps (~50M tokens processed)
    warmup_iters = 200       # Warmup steps
    max_lr = 5e-4            # Peak learning rate
    min_lr = 5e-5            # Minimum learning rate after decay
    weight_decay = 0.1       # AdamW weight decay
    grad_clip = 1.0          # Gradient clipping norm limit
    eval_interval = 250      # Steps between evaluation
    eval_iters = 20          # Number of validation batches to average
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load Memory-Mapped Binary Data
    train_bin_path = "train.bin"
    val_bin_path = "val.bin"
    
    if not os.path.exists(train_bin_path):
        raise FileNotFoundError(f"Cannot find '{train_bin_path}'. Please run dataset_builder.py first!")
    
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
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    
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
                    x_v, y_v = get_batch(val_data, batch_size, seq_len, device)
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
            
        x, y = get_batch(train_data, batch_size, seq_len, device)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device == 'cuda'), dtype=torch.float16 if device == 'cuda' else torch.float32):
            logits, loss = model(x, y)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        if it % 50 == 0:
            tok_per_sec = (50 * batch_size * seq_len) / (time.time() - start_time) if it > 0 else 0
            start_time = time.time()
            print(f"Step {it:5d}/{max_iters} | Loss: {loss.item():.4f} | Speed: {tok_per_sec:.0f} tok/s")

    print("\n===\nStory-SLM 50M Model Training Complete! Saved Checkpoints in ./checkpoints/\n===")

if __name__ == '__main__':
    train()
