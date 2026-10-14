"""
=============================================================================
Story-SLM 100M: 5-Worker Concurrent HuggingFace Market Downloader
=============================================================================
Author: Advanced AI Masterclass Team
Description: Downloads 5 high-density HuggingFace datasets concurrently using
             5 parallel thread workers into dedicated subfolders in dataset/.
=============================================================================
"""

import os
import sys
import time
import concurrent.futures
from datasets import load_dataset


def worker_1_fineweb_edu():
    """Worker 1: Top Educational Textbooks (FineWeb-Edu)."""
    target_dir = os.path.join("dataset", "fineweb_edu_textbooks")
    os.makedirs(target_dir, exist_ok=True)
    out_file = os.path.join(target_dir, "fineweb_edu_top_textbooks.txt")
    target_bytes = 8 * 1024 * 1024 * 1024  # 8 GB Target

    written_bytes = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    if written_bytes >= target_bytes:
        print(f"[WORKER 1 READY] FineWeb-Edu complete: {written_bytes/(1024**3):.2f} GB")
        return

    print(f"[WORKER 1 STARTED] FineWeb-Edu (Currently at {written_bytes/(1024**3):.2f} GB / 8.00 GB)")
    while written_bytes < target_bytes:
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            count = 0
            with open(out_file, "a", encoding="utf-8") as f:
                for row in ds:
                    score = row.get("int_score", 5)
                    if score >= 4:
                        text = row.get("text", "").strip()
                        if len(text) > 200:
                            entry = (
                                f"<|im_start|>user\nExplain the key concepts in this text.\n<|im_end|>\n"
                                f"<|im_start|>assistant\n{text}\n<|im_end|>\n\n"
                            )
                            f.write(entry)
                            written_bytes += len(entry.encode("utf-8"))
                            count += 1
                            if written_bytes >= target_bytes:
                                break
            print(f"[WORKER 1 SUCCESS] FineWeb-Edu Done! ({written_bytes/(1024**3):.2f} GB)")
            break
        except Exception as e:
            time.sleep(5)


def worker_2_cosmopedia():
    """Worker 2: Synthetic Reasoning & Textbooks (Cosmopedia)."""
    target_dir = os.path.join("dataset", "cosmopedia_reasoning")
    os.makedirs(target_dir, exist_ok=True)
    out_file = os.path.join(target_dir, "cosmopedia_synthetic_textbooks.txt")
    target_bytes = 6 * 1024 * 1024 * 1024  # 6 GB Target

    written_bytes = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    if written_bytes >= target_bytes:
        print(f"[WORKER 2 READY] Cosmopedia complete: {written_bytes/(1024**3):.2f} GB")
        return

    print(f"[WORKER 2 STARTED] Cosmopedia (Currently at {written_bytes/(1024**3):.2f} GB / 6.00 GB)")
    while written_bytes < target_bytes:
        try:
            ds = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", streaming=True)
            count = 0
            with open(out_file, "a", encoding="utf-8") as f:
                for row in ds:
                    text = row.get("text", "").strip()
                    prompt_input = row.get("prompt", "Explain this concept.")
                    if len(text) > 150:
                        entry = (
                            f"<|im_start|>user\nPrompt: {prompt_input}\n<|im_end|>\n"
                            f"<|im_start|>assistant\n<|reasoning_start|>\nAnalyzing concept: {prompt_input[:100]}...\n<|reasoning_end|>\n{text}\n<|im_end|>\n\n"
                        )
                        f.write(entry)
                        written_bytes += len(entry.encode("utf-8"))
                        count += 1
                        if written_bytes >= target_bytes:
                            break
            print(f"[WORKER 2 SUCCESS] Cosmopedia Done! ({written_bytes/(1024**3):.2f} GB)")
            break
        except Exception as e:
            time.sleep(5)


def worker_3_ultrachat():
    """Worker 3: Multi-turn Conversations (UltraChat)."""
    target_dir = os.path.join("dataset", "ultrachat_conversations")
    os.makedirs(target_dir, exist_ok=True)
    out_file = os.path.join(target_dir, "ultrachat_multi_turn_conversations.txt")
    target_bytes = 3 * 1024 * 1024 * 1024  # 3 GB Target

    written_bytes = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    if written_bytes >= target_bytes:
        print(f"[WORKER 3 READY] UltraChat complete: {written_bytes/(1024**3):.2f} GB")
        return

    print(f"[WORKER 3 STARTED] UltraChat (Currently at {written_bytes/(1024**3):.2f} GB / 3.00 GB)")
    while written_bytes < target_bytes:
        try:
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
            count = 0
            with open(out_file, "a", encoding="utf-8") as f:
                for row in ds:
                    messages = row.get("messages", [])
                    if len(messages) >= 2:
                        conv_str = ""
                        for msg in messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "").strip()
                            conv_str += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                        conv_str += "\n"
                        f.write(conv_str)
                        written_bytes += len(conv_str.encode("utf-8"))
                        count += 1
                        if written_bytes >= target_bytes:
                            break
            print(f"[WORKER 3 SUCCESS] UltraChat Done! ({written_bytes/(1024**3):.2f} GB)")
            break
        except Exception as e:
            time.sleep(5)


def worker_4_openhermes():
    """Worker 4: Structured Instruction Reasoning (OpenHermes-2.5)."""
    target_dir = os.path.join("dataset", "openhermes_reasoning")
    os.makedirs(target_dir, exist_ok=True)
    out_file = os.path.join(target_dir, "openhermes_instruction_dataset.txt")
    target_bytes = 2 * 1024 * 1024 * 1024  # 2 GB Target

    written_bytes = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    if written_bytes >= target_bytes:
        print(f"[WORKER 4 READY] OpenHermes complete: {written_bytes/(1024**3):.2f} GB")
        return

    print(f"[WORKER 4 STARTED] OpenHermes (Currently at {written_bytes/(1024**3):.2f} GB / 2.00 GB)")
    while written_bytes < target_bytes:
        try:
            ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
            count = 0
            with open(out_file, "a", encoding="utf-8") as f:
                for row in ds:
                    instruction = row.get("instruction", "").strip()
                    input_text = row.get("input", "").strip()
                    output_text = row.get("output", "").strip()
                    full_prompt = f"{instruction}\n{input_text}".strip()
                    if len(output_text) > 50:
                        entry = (
                            f"<|im_start|>user\n{full_prompt}\n<|im_end|>\n"
                            f"<|im_start|>assistant\n<|reasoning_start|>\nStep-by-step resolution of user instruction.\n<|reasoning_end|>\n{output_text}\n<|im_end|>\n\n"
                        )
                        f.write(entry)
                        written_bytes += len(entry.encode("utf-8"))
                        count += 1
                        if written_bytes >= target_bytes:
                            break
            print(f"[WORKER 4 SUCCESS] OpenHermes Done! ({written_bytes/(1024**3):.2f} GB)")
            break
        except Exception as e:
            time.sleep(5)


def worker_5_tinystories():
    """Worker 5: Narrative Logic (TinyStories)."""
    target_dir = os.path.join("dataset", "tinystories_narrative")
    os.makedirs(target_dir, exist_ok=True)
    out_file = os.path.join(target_dir, "tinystories_narrative_dataset.txt")
    target_bytes = 1 * 1024 * 1024 * 1024  # 1 GB Target

    written_bytes = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    if written_bytes >= target_bytes:
        print(f"[WORKER 5 READY] TinyStories complete: {written_bytes/(1024**3):.2f} GB")
        return

    print(f"[WORKER 5 STARTED] TinyStories (Currently at {written_bytes/(1024**3):.2f} GB / 1.00 GB)")
    while written_bytes < target_bytes:
        try:
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            count = 0
            with open(out_file, "a", encoding="utf-8") as f:
                for row in ds:
                    text = row.get("text", "").strip()
                    if len(text) > 100:
                        entry = (
                            f"<|im_start|>user\nWrite a short narrative story.\n<|im_end|>\n"
                            f"<|im_start|>assistant\n{text}\n<|im_end|>\n\n"
                        )
                        f.write(entry)
                        written_bytes += len(entry.encode("utf-8"))
                        count += 1
                        if written_bytes >= target_bytes:
                            break
            print(f"[WORKER 5 SUCCESS] TinyStories Done! ({written_bytes/(1024**3):.2f} GB)")
            break
        except Exception as e:
            time.sleep(5)


def run_5_workers_parallel():
    """Launches all 5 HF workers concurrently in a ThreadPoolExecutor."""
    print("=================================================================")
    print("=== Launching 5-Worker Concurrent HuggingFace Downloader ===")
    print("=================================================================")
    
    workers = [
        worker_1_fineweb_edu,
        worker_2_cosmopedia,
        worker_3_ultrachat,
        worker_4_openhermes,
        worker_5_tinystories
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(w) for w in workers]
        concurrent.futures.wait(futures)

    print("\n[COMPLETE] All 5 HuggingFace Datasets Downloaded Successfully!")


if __name__ == "__main__":
    run_5_workers_parallel()
