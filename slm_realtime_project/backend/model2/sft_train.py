"""
=============================================================================
Story-SLM2: Supervised Fine-Tuning Script (sft_train.py)
=============================================================================
Fine-tunes the already-pretrained Story-SLM2 (models/model2.pt, step 124387)
on backend/model2/sft_data/sft_dataset_final.jsonl -- a small, hand-curated
set of prompt->response pairs built specifically to fix a diagnosed bug: the
base model drifts off-topic partway through long generations and treats
every input (even "Hi") as a story request.

This is NOT pretraining. Key differences, each chosen to avoid overfitting
and catastrophic forgetting of the base model's fluency:
  - Loss is masked to the response tokens only (the prompt/user turn is
    context, not something the model should be trained to predict).
  - Learning rate is ~10-15x lower than pretraining's.
  - Training stops early the moment validation loss stops improving,
    instead of running a fixed number of epochs blindly.
  - The validation split is held out by example, so every val story is a
    topic the model never saw during fine-tuning -- this is what actually
    tests generalization of the "stay on topic" behavior, not memorization
    of the training set's specific stories.
=============================================================================
"""

import os
import glob
import json
import math
import random
import torch
import torch.nn.functional as F

from model import StorySLM2, StorySLM2Config
from tokenizer import StorySLM2Tokenizer


def _find_kaggle_assets_dir():
    """Datasets attached via kernel-metadata.json can mount flat
    (/kaggle/input/<slug>/) or nested (/kaggle/input/datasets/<owner>/<slug>/)
    depending on how Kaggle's backend resolves them -- a real gotcha hit
    during pretraining. Searching recursively for a known file sidesteps
    needing to guess the exact depth."""
    matches = glob.glob("/kaggle/input/**/sft_dataset_final.jsonl", recursive=True)
    if not matches:
        raise FileNotFoundError(
            "Could not find sft_dataset_final.jsonl under /kaggle/input/. "
            "Did you attach the story-slm2-sft-assets dataset?"
        )
    return os.path.dirname(matches[0])


def _resolve_paths():
    """Same script runs locally and on Kaggle without edits, matching
    train.py's existing pattern."""
    if os.path.isdir("/kaggle/input"):
        assets_dir = _find_kaggle_assets_dir()
        return (
            os.path.join(assets_dir, "sft_dataset_final.jsonl"),
            os.path.join(assets_dir, "model2.pt"),
            "/kaggle/working/sft_checkpoints",
        )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return (
        os.path.join(os.path.dirname(__file__), "sft_data", "sft_dataset_final.jsonl"),
        os.path.join(project_root, "models", "model2.pt"),
        os.path.join(project_root, "sft_checkpoints"),
    )


DATASET_PATH, BASE_MODEL_PATH, SFT_CHECKPOINT_DIR = _resolve_paths()

SEED = 1234
VAL_FRACTION = 0.1
MICRO_BATCH_SIZE = 4
MAX_EPOCHS = 12
EARLY_STOP_PATIENCE = 3   # stop if val loss doesn't improve for this many consecutive evals
MAX_LR = 2e-5             # ~15x below pretraining's 3e-4 -- avoids overwriting base fluency
MIN_LR = 2e-6
WARMUP_STEPS = 30
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0


def load_examples():
    examples = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def tokenize_examples(examples, tokenizer: StorySLM2Tokenizer):
    """Each example becomes (input_ids, labels) where labels is -100 (ignored)
    for every position whose target token belongs to the prompt/control-token
    prefix, and the real token id for every position belonging to the
    response (including the trailing <|im_end|>, so the model learns when to
    stop)."""
    tokenized = []
    for ex in examples:
        prefix = f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        full = f"{prefix}{ex['response']}<|im_end|>"

        prefix_ids = tokenizer.encode(prefix)
        full_ids = tokenizer.encode(full)
        resp_start = len(prefix_ids)  # index of the first response token in full_ids

        x = full_ids[:-1]
        y = full_ids[1:]
        labels = [tok if (i + 1) >= resp_start else -100 for i, tok in enumerate(y)]

        tokenized.append({"category": ex["category"], "x": x, "labels": labels})
    return tokenized


def stratified_split(tokenized, val_fraction, seed):
    rng = random.Random(seed)
    by_category = {}
    for ex in tokenized:
        by_category.setdefault(ex["category"], []).append(ex)

    train, val = [], []
    for cat, items in by_category.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def make_length_bucketed_batches(dataset, batch_size):
    """Sorting by length before chunking keeps padding waste low (a story_long
    example at ~700 tokens never shares a batch with a 10-token greeting).
    Batch ORDER is reshuffled every epoch by the caller; batch CONTENTS are
    fixed by this sort, which is fine since the sort itself has no memorizable
    signal (length, not content)."""
    sorted_ds = sorted(dataset, key=lambda ex: len(ex["x"]))
    return [sorted_ds[i : i + batch_size] for i in range(0, len(sorted_ds), batch_size)]


def collate(batch, pad_id, device):
    max_len = max(len(ex["x"]) for ex in batch)
    x_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels_batch = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex["x"])
        x_batch[i, :n] = torch.tensor(ex["x"], dtype=torch.long)
        labels_batch[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
    return x_batch.to(device), labels_batch.to(device)


def get_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, val_batches, pad_id, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in val_batches:
        x, labels = collate(batch, pad_id, device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum"
        )
        n_valid = (labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n_valid
    model.train()
    return total_loss / max(1, total_tokens)


def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Story-SLM2 SFT ===\nDevice: {device.upper()}")

    tokenizer = StorySLM2Tokenizer()
    examples = load_examples()
    print(f"Loaded {len(examples)} raw examples from {DATASET_PATH}")

    tokenized = tokenize_examples(examples, tokenizer)
    train_set, val_set = stratified_split(tokenized, VAL_FRACTION, SEED)
    print(f"Train: {len(train_set)}  Val: {len(val_set)} (held out, never trained on)\n")

    checkpoint = torch.load(BASE_MODEL_PATH, map_location=device, weights_only=False)
    config: StorySLM2Config = checkpoint["config"]
    model = StorySLM2(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded base model from {BASE_MODEL_PATH} (step {checkpoint.get('step')}, "
          f"pretrain val_loss {checkpoint.get('val_loss')})\n")

    decay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=MAX_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    val_batches = make_length_bucketed_batches(val_set, MICRO_BATCH_SIZE)
    steps_per_epoch = math.ceil(len(train_set) / MICRO_BATCH_SIZE)
    total_steps = steps_per_epoch * MAX_EPOCHS

    os.makedirs(SFT_CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    model.train()
    for epoch in range(1, MAX_EPOCHS + 1):
        batches = make_length_bucketed_batches(train_set, MICRO_BATCH_SIZE)
        random.shuffle(batches)

        epoch_loss, epoch_tokens = 0.0, 0
        for batch in batches:
            lr = get_lr(global_step, total_steps, WARMUP_STEPS, MAX_LR, MIN_LR)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, labels = collate(batch, tokenizer.pad_id, device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            n_valid = (labels != -100).sum().item()
            epoch_loss += loss.item() * n_valid
            epoch_tokens += n_valid
            global_step += 1

        train_loss = epoch_loss / max(1, epoch_tokens)
        val_loss = evaluate(model, val_batches, tokenizer.pad_id, device)
        print(f"[EPOCH {epoch:2d}/{MAX_EPOCHS}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_path = os.path.join(SFT_CHECKPOINT_DIR, "story_slm2_sft_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "base_checkpoint": BASE_MODEL_PATH,
                },
                best_path,
            )
            print(f"  -> new best val_loss, saved to {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> no improvement ({epochs_without_improvement}/{EARLY_STOP_PATIENCE})")
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\n[EARLY STOP] val_loss has not improved for {EARLY_STOP_PATIENCE} epochs. Stopping.")
                break

    print(f"\n=== SFT complete. Best val_loss: {best_val_loss:.4f} ===")
    print(f"Best checkpoint: {os.path.join(SFT_CHECKPOINT_DIR, 'story_slm2_sft_best.pt')}")


if __name__ == "__main__":
    train()
