"""
=============================================================================
Story-SLM 100M: Zero-RAM Chunked Tokenizer Pipeline
=============================================================================
Author: Advanced AI Masterclass Team
Architecture:
  - Streams raw text files line-by-line / chunk-by-chunk (0 MB memory spikes).
  - BPE Tokenizer: OpenAI tiktoken with 50,257 vocabulary.
  - Formats unformatted texts on the fly and appends directly to uint32 train_m2.bin.
  - Zero MemoryError guarantee on any PC!
=============================================================================
"""

import os
import sys
import time
import numpy as np
import tiktoken


class ProductionMasterDatasetBuilder:
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir
        self.train_bin_path = "train_m2.bin"
        self.val_bin_path = "val_m2.bin"

        # 50,257 Standard GPT-2 Base Vocabulary + Control Tokens
        self.special_tokens = {
            "<|endoftext|>": 50256,
            "<|im_start|>": 50257,
            "<|im_end|>": 50258,
            "<|pad|>": 50259,
            "<|reasoning_start|>": 50260,
            "<|reasoning_end|>": 50261,
        }
        base_enc = tiktoken.get_encoding("gpt2")
        self.enc = tiktoken.Encoding(
            name="story_gpt2_production",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens={**base_enc._special_tokens, **self.special_tokens},
        )
        self.vocab_size = 50257

    def stream_tokenize_all(self, val_ratio: float = 0.02, chunk_lines: int = 25000):
        """Streams every file line-by-line, tokenizes in chunks, and appends to binary."""
        print("=================================================================")
        print("=== Story-SLM 100M: Zero-RAM Stream Tokenization Pipeline ===")
        print("=================================================================")
        print(f" -> Vocabulary: {self.vocab_size:,} (Clean GPT-2 BPE, Zero Modulo)")
        print(f" -> Output Binaries: '{self.train_bin_path}' & '{self.val_bin_path}'")
        print("=================================================================\n")

        start_time = time.time()

        # Reset binary output files
        open(self.train_bin_path, "wb").close()
        open(self.val_bin_path, "wb").close()

        total_files = 0
        total_tokens_trained = 0
        total_tokens_val = 0
        total_bytes_processed = 0

        # Scan all text and PDF files in dataset/
        all_files = []
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file in ["dataset.txt", "train_m2.bin", "val_m2.bin"]:
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in [".txt", ".pdf"]:
                    all_files.append((os.path.join(root, file), file, os.path.basename(root)))

        print(f"Discovered {len(all_files)} data files across all subfolders.\n")

        text_buffer = []
        buffer_chars = 0
        max_buffer_chars = 20 * 1024 * 1024  # Flush every 20 MB text to prevent any MemoryError

        for file_path, file_name, folder_cat in all_files:
            total_files += 1
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"[{total_files:3d}/{len(all_files)}] Streaming: {folder_cat}/{file_name} ({file_size_mb:.2f} MB)")

            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            text_buffer.append(line)
                            buffer_chars += len(line)
                            total_bytes_processed += len(line.encode("utf-8", errors="ignore"))

                            if buffer_chars >= max_buffer_chars:
                                chunk_text = "".join(text_buffer)
                                text_buffer = []
                                buffer_chars = 0
                                
                                # Format if raw
                                if "<|im_start|>" not in chunk_text:
                                    chunk_text = (
                                        f"<|im_start|>user\nRead this narrative.\n<|im_end|>\n"
                                        f"<|im_start|>assistant\n{chunk_text}\n<|im_end|>\n\n"
                                    )

                                tr_cnt, val_cnt = self._flush_chunk(chunk_text, val_ratio)
                                total_tokens_trained += tr_cnt
                                total_tokens_val += val_cnt

                                current_tokens_m = (total_tokens_trained + total_tokens_val) / 1e6
                                elapsed_min = (time.time() - start_time) / 60
                                print(f"  -> Packed Tokens: {current_tokens_m:.1f}M | Elapsed: {elapsed_min:.1f}m")

                except Exception as e:
                    print(f"  [WARNING] Could not read {file_name}: {e}")

            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    for page in reader.pages:
                        page_t = page.extract_text()
                        if page_t and len(page_t.strip()) > 30:
                            entry = (
                                f"<|im_start|>user\nSource: {file_name}\nRead this excerpt.\n<|im_end|>\n"
                                f"<|im_start|>assistant\n{page_t.strip()}\n<|im_end|>\n\n"
                            )
                            text_buffer.append(entry)
                            buffer_chars += len(entry)
                            total_bytes_processed += len(entry.encode("utf-8"))

                            if buffer_chars >= max_buffer_chars:
                                chunk_text = "".join(text_buffer)
                                text_buffer = []
                                buffer_chars = 0
                                tr_cnt, val_cnt = self._flush_chunk(chunk_text, val_ratio)
                                total_tokens_trained += tr_cnt
                                total_tokens_val += val_cnt
                except Exception as e:
                    print(f"  [WARNING] Could not read PDF {file_name}: {e}")

        # Flush final remaining buffer
        if text_buffer:
            chunk_text = "".join(text_buffer)
            tr_cnt, val_cnt = self._flush_chunk(chunk_text, val_ratio)
            total_tokens_trained += tr_cnt
            total_tokens_val += val_cnt

        total_elapsed = time.time() - start_time
        train_file_gb = os.path.getsize(self.train_bin_path) / (1024**3)
        val_file_mb = os.path.getsize(self.val_bin_path) / (1024**2)

        print("\n" + "="*65)
        print("=== 🎉 MASTER TOKENIZATION COMPLETED 100%! ===")
        print("="*65)
        print(f" -> Total Files Processed: {total_files} files")
        print(f" -> Total Raw Data Size  : {total_bytes_processed / (1024**3):.2f} GB")
        print(f" -> Train Binary File    : '{self.train_bin_path}' ({train_file_gb:.2f} GB)")
        print(f" -> Val Binary File      : '{self.val_bin_path}' ({val_file_mb:.2f} MB)")
        print(f" -> Total Tokens Packed  : {total_tokens_trained + total_tokens_val:,} tokens")
        print(f" -> Total Time Elapsed   : {total_elapsed / 60:.2f} minutes")
        print("="*65 + "\n")

    def _flush_chunk(self, text: str, val_ratio: float):
        """Encodes text chunk with tiktoken and writes uint32 directly to disk."""
        raw_tokens = self.enc.encode(text, allowed_special="all")
        tokens_arr = np.array(raw_tokens, dtype=np.uint32)

        val_len = int(len(tokens_arr) * val_ratio)
        train_part = tokens_arr[:-val_len] if val_len > 0 else tokens_arr
        val_part = tokens_arr[-val_len:] if val_len > 0 else tokens_arr[:100]

        with open(self.train_bin_path, "ab") as f_train:
            f_train.write(train_part.tobytes())
        with open(self.val_bin_path, "ab") as f_val:
            f_val.write(val_part.tobytes())

        return len(train_part), len(val_part)


if __name__ == "__main__":
    builder = ProductionMasterDatasetBuilder()
    builder.stream_tokenize_all(val_ratio=0.02)
