"""
=============================================================================
Karna-49M: Automated Hugging Face Model Uploader (upload_to_hf.py)
=============================================================================
Uploads the complete Karna-49M GGUF suite (F16, Q8_0, Q4_0, Modelfile, README)
and the PyTorch checkpoint to Hugging Face Hub under your account: VarunHujan.
=============================================================================
"""

import os
import sys
from huggingface_hub import HfApi, create_repo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "").strip()
REPO_ID = "VarunHujan/Karna-49M-SFT-Model"  # Target repository on Hugging Face

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GGUF_DIR = os.path.join(BASE_DIR, "exports", "gguf")
PT_CHECKPOINT = os.path.join(BASE_DIR, "models", "story_slm2_sft_best.pt")


def main():
    print("==================================================================")
    print("🚀 Karna-49M Model: Hugging Face Upload Pipeline")
    print("==================================================================")
    print(f"Target Repository : https://huggingface.co/{REPO_ID}")
    print(f"GGUF Directory    : {GGUF_DIR}")
    print("==================================================================\n")

    global HF_TOKEN
    if not HF_TOKEN:
        print("[!] HF_TOKEN not found in environment variables.")
        HF_TOKEN = input("Enter your Hugging Face write token: ").strip()

    api = HfApi(token=HF_TOKEN)

    # 1. Verify User Credentials
    try:
        user_info = api.whoami()
        print(f"[1/4] ✅ Authenticated successfully as: '{user_info['name']}'")
    except Exception as e:
        print(f"[FATAL] Authentication failed: {e}")
        sys.exit(1)

    # 2. Ensure Repository Exists (creates if it doesn't already exist)
    try:
        create_repo(
            repo_id=REPO_ID,
            token=HF_TOKEN,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"[2/4] ✅ Repository '{REPO_ID}' is ready.")
    except Exception as e:
        print(f"[WARNING] Repo creation check: {e}")

    # 3. Upload GGUF Suite (karna_49m_sft_f16, q8_0, q4_0, Modelfile, README.md)
    if os.path.exists(GGUF_DIR):
        print("\n[3/4] 📦 Uploading GGUF files to Hugging Face...")
        print("      (Files: karna_49m_sft_f16.gguf, q8_0.gguf, q4_0.gguf, Modelfile, README.md)")
        try:
            api.upload_folder(
                folder_path=GGUF_DIR,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message="Add Karna-49M GGUF models (F16, Q8_0, Q4_0), Modelfile, and model card"
            )
            print("      ✅ GGUF folder upload complete!")
        except Exception as e:
            print(f"[ERROR] Failed to upload GGUF folder: {e}")
    else:
        print(f"[WARNING] GGUF directory not found at: {GGUF_DIR}")

    # 4. Upload PyTorch .pt Master Checkpoint (Optional / Additional)
    if os.path.exists(PT_CHECKPOINT):
        print("\n[4/4] 🧠 Uploading PyTorch SFT master weights (karna_49m_sft_best.pt)...")
        try:
            api.upload_file(
                path_or_fileobj=PT_CHECKPOINT,
                path_in_repo="karna_49m_sft_best.pt",
                repo_id=REPO_ID,
                repo_type="model",
                commit_message="Add Karna-49M master PyTorch checkpoint (karna_49m_sft_best.pt)"
            )
            print("      ✅ PyTorch checkpoint upload complete!")
        except Exception as e:
            print(f"[ERROR] Failed to upload PyTorch checkpoint: {e}")
    else:
        print(f"[INFO] No .pt checkpoint found at {PT_CHECKPOINT}. Skipping.")

    print("\n==================================================================")
    print(f"🎉 All files successfully published to Hugging Face!")
    print(f"🔗 View your live model at: https://huggingface.co/{REPO_ID}")
    print("==================================================================")


if __name__ == "__main__":
    main()
