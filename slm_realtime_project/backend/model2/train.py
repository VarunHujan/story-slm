"""
=============================================================================
Story-SLM2: Pre-Training Script (train.py)
=============================================================================
Loads the memory-mapped train_m2.bin / val_m2.bin produced by
build_master_dataset_100m.py at the project root, and trains the 101.6M
parameter Story-SLM2 backbone with AdamW + cosine LR decay + gradient
accumulation + mixed precision.

Before touching the GPU, this script runs a mandatory data/config alignment
check: it scans the full token range of train_m2.bin and refuses to start if
any token id would fall outside the model's embedding table. This exact
mismatch (tokenizer vocab != model vocab) is what caused Model 1 to collapse
into gibberish, so here it is a hard, fail-fast assertion instead of a
multi-hour surprise.
=============================================================================
"""

import os
import math
import time
import numpy as np
import torch

from model import StorySLM2, StorySLM2Config
from tokenizer import REAL_VOCAB_SIZE

def _resolve_paths():
    """Same train.py runs locally and on Kaggle without edits. On Kaggle,
    kaggle_bootstrap.py reconstructs the dataset into /kaggle/temp/data/ (NOT
    /kaggle/working/ -- that has a ~19.5GB output quota shared with checkpoints,
    and the 16.6GB dataset alone would leave no room for them) and restores any
    previous checkpoint into /kaggle/working/checkpoints_model2/ before this
    ever runs."""
    if os.path.isdir("/kaggle/input"):
        return (
            "/kaggle/temp/data/train_m2.bin",
            "/kaggle/temp/data/val_m2.bin",
            "/kaggle/working/checkpoints_model2",
        )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return (
        os.path.join(project_root, "train_m2.bin"),
        os.path.join(project_root, "val_m2.bin"),
        os.path.join(project_root, "checkpoints_model2"),
    )


TRAIN_BIN_PATH, VAL_BIN_PATH, CHECKPOINT_DIR = _resolve_paths()


def verify_data_model_alignment(data: np.memmap, vocab_size: int, label: str, chunk_size: int = 50_000_000):
    """Scans the full memmap in chunks and asserts every token id fits the embedding table."""
    print(f"[VERIFY] Scanning {label} ({len(data):,} tokens) for vocabulary alignment...")
    start = time.time()
    global_max = 0
    global_min = np.iinfo(np.uint32).max
    for offset in range(0, len(data), chunk_size):
        chunk = data[offset : offset + chunk_size]
        chunk_max = int(chunk.max())
        chunk_min = int(chunk.min())
        if chunk_max > global_max:
            global_max = chunk_max
        if chunk_min < global_min:
            global_min = chunk_min

    elapsed = time.time() - start
    print(f"[VERIFY] {label}: min={global_min}  max={global_max}  (scanned in {elapsed:.1f}s)")

    if global_max >= vocab_size:
        raise ValueError(
            f"[FATAL] {label} contains token id {global_max}, but the model's embedding "
            f"table only has {vocab_size} rows. This is the Model-1 vocab-mismatch bug. "
            f"Fix StorySLM2Config.vocab_size before training."
        )
    if global_max >= REAL_VOCAB_SIZE:
        raise ValueError(
            f"[FATAL] {label} contains token id {global_max}, which exceeds the expected "
            f"real vocabulary cardinality of {REAL_VOCAB_SIZE}. The tokenizer used to build "
            f"this .bin file does not match tokenizer.py's special token map."
        )
    print(f"[VERIFY] {label} is safely within vocab_size={vocab_size}. OK to train.\n")


def prune_old_checkpoints(checkpoint_dir: str, keep_last_n: int):
    """Deletes numbered checkpoints beyond the last N (each is ~1.2GB; unbounded retention
    over a 124k-iteration run would exhaust disk/Kaggle output storage). latest.pt is never
    touched here since it's overwritten in place, not numbered."""
    numbered = [f for f in os.listdir(checkpoint_dir) if f.startswith("story_slm2_step_") and f.endswith(".pt")]
    numbered.sort(key=lambda f: int(f[len("story_slm2_step_") : -len(".pt")]))
    for stale in numbered[:-keep_last_n]:
        os.remove(os.path.join(checkpoint_dir, stale))


def get_underlying_model(model):
    """Returns the raw model regardless of whether it's wrapped in DataParallel,
    so checkpoints always save/load plain (non 'module.'-prefixed) keys -- keeping
    them compatible whether a run used 1 GPU or multiple."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_batch(data: np.ndarray, batch_size: int, seq_len: int, device: str):
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_lr(it: int, max_iters: int, warmup_iters: int, max_lr: float, min_lr: float) -> float:
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train():
    print("=== Story-SLM2 Pre-Training Initialization ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution device: {device.upper()}")
    num_gpus = torch.cuda.device_count() if device == "cuda" else 0
    if device == "cuda":
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.empty_cache()

    micro_batch_size = 8
    grad_accum_steps = 4
    seq_len = 1024
    max_iters = 124_388     # exactly 1 full epoch over train_m2.bin's 4,075,951,821 tokens
    warmup_iters = 2_000
    max_lr = 3e-4
    min_lr = 3e-5
    weight_decay = 0.1
    grad_clip = 1.0
    eval_interval = 250     # checkpoint frequently -- Kaggle sessions/commits can be cut off without warning
    eval_iters = 20
    keep_last_n_checkpoints = 3
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_BIN_PATH):
        raise FileNotFoundError(
            f"Cannot find '{TRAIN_BIN_PATH}'. Run build_master_dataset_100m.py at the "
            f"project root first."
        )

    train_data = np.memmap(TRAIN_BIN_PATH, dtype=np.uint32, mode="r")
    val_data = np.memmap(VAL_BIN_PATH, dtype=np.uint32, mode="r") if os.path.exists(VAL_BIN_PATH) else train_data
    print(f"Loaded train_m2.bin: {len(train_data):,} tokens")
    print(f"Loaded val_m2.bin:   {len(val_data):,} tokens\n")

    config = StorySLM2Config()

    verify_data_model_alignment(val_data, config.vocab_size, "val_m2.bin")
    verify_data_model_alignment(train_data, config.vocab_size, "train_m2.bin")

    model = StorySLM2(config).to(device)
    total_params, _ = model.count_parameters()
    print(f"Initialized Story-SLM2 with {total_params:,} parameters ({total_params/1e6:.2f}M)\n")

    decay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_step = 0
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if os.path.exists(latest_path):
        print(f"[RESUME] Found existing checkpoint at '{latest_path}', resuming...")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"[RESUME] Resuming from step {start_step}/{max_iters} (val_loss was {checkpoint.get('val_loss')})\n")
    else:
        print("[RESUME] No existing checkpoint found. Starting fresh from step 0.\n")

    if num_gpus > 1:
        print(f"[MULTI-GPU] {num_gpus} GPUs detected -- wrapping model with DataParallel.\n")
        model = torch.nn.DataParallel(model)

    print("--- Starting Training Loop ---")
    start_time = time.time()

    for it in range(start_step, max_iters):
        lr = get_lr(it, max_iters, warmup_iters, max_lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % eval_interval == 0 or it == max_iters - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(eval_iters):
                    val_loss = model(*get_batch(val_data, micro_batch_size, seq_len, device))[1]
                    val_losses.append(val_loss.mean().item())  # .mean() is a no-op on 1 GPU; averages per-GPU losses on multi-GPU
                avg_val_loss = float(np.mean(val_losses))

            print(f"\n[STEP {it:5d}/{max_iters}] Val Loss: {avg_val_loss:.4f} | LR: {lr:.6f}")

            checkpoint = {
                "step": it,
                "model_state_dict": get_underlying_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": avg_val_loss,
                "config": config,
            }
            numbered_path = os.path.join(CHECKPOINT_DIR, f"story_slm2_step_{it}.pt")
            torch.save(checkpoint, numbered_path)
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "latest.pt"))
            print(f"Saved checkpoint (step {it}) and updated latest.pt")
            prune_old_checkpoints(CHECKPOINT_DIR, keep_last_n_checkpoints)
            model.train()

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = get_batch(train_data, micro_batch_size, seq_len, device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16 if device == "cuda" else torch.float32):
                _, loss = model(x, y)
                loss = loss.mean() / grad_accum_steps  # .mean() is a no-op on 1 GPU; averages per-GPU losses on multi-GPU
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

    print("\n=== Story-SLM2 Training Complete ===")


if __name__ == "__main__":
    train()
