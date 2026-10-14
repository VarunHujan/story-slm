"""
=============================================================================
Story-SLM2: Kaggle Session Bootstrap (kaggle_bootstrap.py)
=============================================================================
Run this as the FIRST cell in every Kaggle session, before train.py.

It does two things a fresh Kaggle session needs done every time:

  1. Reconstructs train_m2.bin / val_m2.bin from the chunked dataset you
     uploaded (see split_dataset_for_kaggle.py), and verifies the SHA256
     hash of the rebuilt file matches the original before trusting it.
     This catches upload/download corruption before any GPU time is spent.

  2. If a previous session's checkpoint was attached as an input dataset,
     copies its latest.pt into this session's working checkpoint directory
     so train.py's own resume logic picks it up and continues instead of
     starting over from step 0.
=============================================================================
"""

import os
import glob
import json
import shutil
import hashlib

INPUT_ROOT = "/kaggle/input"
DATA_DIR = "/kaggle/temp/data"  # NOT /kaggle/working/ -- that has a ~19.5GB output quota
                                 # shared with checkpoints, and the 16.6GB reconstructed
                                 # dataset alone would leave no room for them. /kaggle/temp/
                                 # doesn't count against that quota and is regenerated fresh
                                 # every session anyway, so nothing is lost by putting it here.
CHECKPOINT_DIR = "/kaggle/working/checkpoints_model2"


def sha256_of_file(path: str, bufsize: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(bufsize)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def find_manifest_dir() -> str:
    # Kaggle has used both a flat layout (/kaggle/input/<slug>/) and a nested
    # one (/kaggle/input/datasets/<owner>/<slug>/) across environments, so
    # search recursively instead of assuming a fixed depth.
    matches = glob.glob(f"{INPUT_ROOT}/**/manifest.json", recursive=True)
    if not matches:
        raise FileNotFoundError(
            "No attached dataset with manifest.json found under /kaggle/input/. "
            "Did you attach the dataset produced by split_dataset_for_kaggle.py?"
        )
    return os.path.dirname(matches[0])


def reconstruct(prefix: str, manifest: dict, dataset_dir: str) -> str:
    info = manifest["files"][prefix]
    out_path = os.path.join(DATA_DIR, f"{prefix}.bin")

    if os.path.exists(out_path) and os.path.getsize(out_path) == info["size"]:
        print(f"[SKIP] {out_path} already reconstructed this session (size matches).")
        return out_path

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[BUILD] Reconstructing {prefix} from {info['n_chunks']} chunks...")
    with open(out_path, "wb") as out:
        for i in range(info["n_chunks"]):
            chunk_path = os.path.join(dataset_dir, f"{prefix}.part{i:03d}")
            with open(chunk_path, "rb") as f:
                shutil.copyfileobj(f, out, length=64 * 1024 * 1024)
            print(f"  Joined chunk {i + 1}/{info['n_chunks']}")

    actual_size = os.path.getsize(out_path)
    if actual_size != info["size"]:
        raise ValueError(f"[FATAL] {out_path} size mismatch: expected {info['size']}, got {actual_size}.")

    print(f"[HASH] Verifying SHA256 of reconstructed {prefix}.bin...")
    actual_hash = sha256_of_file(out_path)
    if actual_hash != info["sha256"]:
        raise ValueError(
            f"[FATAL] {out_path} SHA256 mismatch after reconstruction!\n"
            f"  expected: {info['sha256']}\n"
            f"  actual:   {actual_hash}\n"
            f"Re-upload the dataset chunks — this file is corrupted, do not train on it."
        )
    print(f"[OK] {prefix}.bin verified byte-for-byte identical to the original ({info['size']/1e9:.2f} GB)\n")
    return out_path


def restore_previous_checkpoint():
    # Accept either a file literally named latest.pt, or a numbered
    # story_slm2_step_<N>.pt (e.g. when a checkpoint is attached under its
    # original filename instead of being renamed before upload). When
    # multiple numbered checkpoints are attached, pick the highest step.
    candidates = glob.glob(f"{INPUT_ROOT}/**/latest.pt", recursive=True)
    if not candidates:
        numbered = glob.glob(f"{INPUT_ROOT}/**/story_slm2_step_*.pt", recursive=True)
        if numbered:
            def step_of(path):
                name = os.path.basename(path)
                return int(name[len("story_slm2_step_") : -len(".pt")])
            candidates = [max(numbered, key=step_of)]

    if not candidates:
        print("[CHECKPOINT] No previous checkpoint attached. Starting fresh from step 0.\n")
        return

    src = candidates[0]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dst = os.path.join(CHECKPOINT_DIR, "latest.pt")
    shutil.copyfile(src, dst)
    print(f"[CHECKPOINT] Restored previous checkpoint from '{src}' -> '{dst}'")
    print("[CHECKPOINT] train.py will resume from this checkpoint's step automatically.\n")


def main():
    dataset_dir = find_manifest_dir()
    with open(os.path.join(dataset_dir, "manifest.json")) as f:
        manifest = json.load(f)

    train_path = reconstruct("train_m2", manifest, dataset_dir)
    val_path = reconstruct("val_m2", manifest, dataset_dir)
    restore_previous_checkpoint()

    print("=" * 60)
    print("[READY] Bootstrap complete.")
    print(f"  TRAIN_BIN_PATH = {train_path}")
    print(f"  VAL_BIN_PATH   = {val_path}")
    print(f"  CHECKPOINT_DIR = {CHECKPOINT_DIR}")
    print("  You can now run: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
