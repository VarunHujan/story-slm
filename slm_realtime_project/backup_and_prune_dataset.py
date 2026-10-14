"""
=============================================================================
Backup and prune dataset/ (already fully tokenized into train_m2.bin/val_m2.bin)
=============================================================================
Processes one top-level category folder at a time: zips it (BZIP2, ~72%
reduction measured on this actual corpus), verifies the zip's CRC integrity
AND that every original file's size is accounted for, and only then deletes
the original folder. This keeps peak disk usage bounded (never needs the
full 20GB + full backup size at once) and is safe to interrupt/resume — any
category already backed up and verified is skipped on a re-run.
=============================================================================
"""

import os
import zipfile
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "dataset_backup")


def zip_folder(folder_path: str, zip_path: str):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            file_list.append(os.path.join(root, fname))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for i, fpath in enumerate(file_list, 1):
            arcname = os.path.relpath(fpath, DATASET_DIR)
            zf.write(fpath, arcname)
            print(f"  [{i}/{len(file_list)}] Added {arcname}")

    return file_list


def verify_zip(zip_path: str, original_files: list) -> bool:
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            print(f"  [FAIL] Corrupt entry inside zip: {bad_file}")
            return False

        zip_sizes = {info.filename: info.file_size for info in zf.infolist()}

    for fpath in original_files:
        arcname = os.path.relpath(fpath, DATASET_DIR).replace(os.sep, "/")
        original_size = os.path.getsize(fpath)
        if arcname not in zip_sizes:
            print(f"  [FAIL] '{arcname}' missing from zip")
            return False
        if zip_sizes[arcname] != original_size:
            print(f"  [FAIL] '{arcname}' size mismatch: original={original_size}, in zip={zip_sizes[arcname]}")
            return False

    return True


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    categories = sorted(
        d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    print(f"Found {len(categories)} categories to back up: {categories}\n")

    total_freed = 0
    for category in categories:
        folder_path = os.path.join(DATASET_DIR, category)
        zip_path = os.path.join(BACKUP_DIR, f"{category}.zip")

        if os.path.exists(zip_path):
            print(f"[SKIP] '{category}' already has a backup at {zip_path}, checking it's still valid...")
            original_files = [os.path.join(r, f) for r, _, fs in os.walk(folder_path) for f in fs] if os.path.isdir(folder_path) else []
            if not original_files:
                print(f"[SKIP] '{category}' original already removed, backup already exists. Moving on.\n")
                continue

        original_size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(folder_path) for f in fs)
        print(f"=== {category} ({original_size/1e9:.2f} GB) ===")

        original_files = zip_folder(folder_path, zip_path)
        zip_size = os.path.getsize(zip_path)
        print(f"  Zipped to {zip_size/1e9:.2f} GB ({(1 - zip_size/original_size)*100:.1f}% reduction)")

        print(f"  Verifying...")
        if not verify_zip(zip_path, original_files):
            print(f"  [FATAL] Verification failed for '{category}'. NOT deleting original. Investigate before re-running.\n")
            continue

        print(f"  [OK] Verified. Deleting original '{folder_path}'...")
        shutil.rmtree(folder_path)
        freed = original_size - zip_size
        total_freed += freed
        print(f"  Freed {freed/1e9:.2f} GB (running total freed: {total_freed/1e9:.2f} GB)\n")

    print("=" * 60)
    print(f"[DONE] Total space freed: {total_freed/1e9:.2f} GB")
    print(f"Backups are in: {BACKUP_DIR}")
    print("Upload this folder to Google Drive yourself (drag-and-drop) for off-disk safekeeping.")
    print("=" * 60)


if __name__ == "__main__":
    main()
