"""
=============================================================================
Split train_m2.bin / val_m2.bin into Kaggle-safe upload chunks
=============================================================================
Kaggle has a documented history of choking on single dataset files over 2GB.
This splits both binaries into ~1.8GB chunks (safely under that ceiling) and
writes a manifest.json with a SHA256 hash of each original file, so the
Kaggle-side reconstruction script can verify nothing got corrupted in transit
before a single hour of GPU time is spent on it.

Run this locally once. Upload the entire kaggle_upload/ folder as a new
Kaggle Dataset (drag-and-drop the folder contents).
=============================================================================
"""

import os
import json
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "kaggle_upload")
CHUNK_SIZE = 1_800_000_000  # ~1.8GB, safely under Kaggle's historical 2GB single-file issue

SOURCES = {
    "train_m2": os.path.join(PROJECT_ROOT, "train_m2.bin"),
    "val_m2": os.path.join(PROJECT_ROOT, "val_m2.bin"),
}


def sha256_of_file(path: str, bufsize: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(bufsize)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def split_file(src_path: str, prefix: str) -> dict:
    total_size = os.path.getsize(src_path)
    print(f"[HASH] Computing SHA256 of {src_path} ({total_size/1e9:.2f} GB)...")
    original_hash = sha256_of_file(src_path)
    print(f"[HASH] {prefix}: {original_hash}")

    n_chunks = 0
    with open(src_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            chunk_path = os.path.join(OUTPUT_DIR, f"{prefix}.part{n_chunks:03d}")
            with open(chunk_path, "wb") as out:
                out.write(chunk)
            print(f"  Wrote {os.path.basename(chunk_path)} ({len(chunk)/1e9:.2f} GB)")
            n_chunks += 1

    print(f"[OK] Split {prefix} into {n_chunks} chunks\n")
    return {"size": total_size, "sha256": original_hash, "n_chunks": n_chunks}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = {"files": {}}

    for prefix, src_path in SOURCES.items():
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Cannot find '{src_path}'. Run build_master_dataset_100m.py first.")
        manifest["files"][prefix] = split_file(src_path, prefix)

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_chunks = sum(v["n_chunks"] for v in manifest["files"].values())
    print("=" * 60)
    print(f"[DONE] {OUTPUT_DIR}")
    print(f"  Total chunk files: {total_chunks} (+ manifest.json)")
    print(f"  Upload the ENTIRE 'kaggle_upload' folder as a new Kaggle Dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()
