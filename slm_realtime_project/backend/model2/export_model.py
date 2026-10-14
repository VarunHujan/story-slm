"""
=============================================================================
Story-SLM2: Export a clean, inference-only model (export_model.py)
=============================================================================
Training checkpoints bundle the AdamW optimizer state (2 extra buffers per
parameter) alongside the actual weights -- needed only to resume training,
useless for generating text. This strips that out, keeping just the weights
and config, and optionally casts to fp16 (halves size, negligible quality
cost -- unlike int8/int4, which risk real degradation on a model this small).

Usage:
    python export_model.py <checkpoint.pt> <output_dir> [--fp16]
=============================================================================
"""

import os
import sys
import argparse
import torch

from model import StorySLM2, StorySLM2Config


def export(checkpoint_path: str, output_path: str, dtype: torch.dtype = torch.float32):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state_dict = checkpoint["model_state_dict"]

    if dtype != torch.float32:
        # Cast while preserving tied weights (tok_embeddings/lm_head share storage).
        # Casting each dict entry independently breaks that sharing -- two separate
        # tensors with identical values get saved instead of one, wasting space.
        casted_by_storage = {}
        new_state_dict = {}
        for k, v in model_state_dict.items():
            if not torch.is_floating_point(v):
                new_state_dict[k] = v
                continue
            storage_key = v.data_ptr()
            if storage_key not in casted_by_storage:
                casted_by_storage[storage_key] = v.to(dtype)
            new_state_dict[k] = casted_by_storage[storage_key]
        model_state_dict = new_state_dict

    export_data = {
        "model_state_dict": model_state_dict,
        "config": checkpoint["config"],
        "step": checkpoint["step"],
        "val_loss": checkpoint.get("val_loss"),
    }
    torch.save(export_data, output_path)
    return export_data


def verify(output_path: str):
    """Loads the exported file fresh and runs a real forward pass -- proves
    the export isn't silently broken before trusting it."""
    data = torch.load(output_path, map_location="cpu", weights_only=False)
    config: StorySLM2Config = data["config"]
    model = StorySLM2(config)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    dummy = torch.randint(0, config.vocab_size, (1, 32))
    with torch.no_grad():
        logits, _ = model(dummy)
    assert logits.shape == (1, 32, config.vocab_size), f"unexpected output shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "exported model produced non-finite logits"

    total_params, _ = model.count_parameters()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[OK] {os.path.basename(output_path)}: {size_mb:.1f} MB, {total_params:,} params, "
          f"step {data['step']}, val_loss {data['val_loss']}, forward pass verified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to a training checkpoint .pt file")
    parser.add_argument("output_dir", help="Directory to write exported files into")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.checkpoint))[0]

    fp32_path = os.path.join(args.output_dir, f"{base}_export_fp32.pt")
    export(args.checkpoint, fp32_path, dtype=torch.float32)
    verify(fp32_path)

    fp16_path = os.path.join(args.output_dir, f"{base}_export_fp16.pt")
    export(args.checkpoint, fp16_path, dtype=torch.float16)
    verify(fp16_path)


if __name__ == "__main__":
    main()
