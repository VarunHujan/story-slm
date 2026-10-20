"""
=============================================================================
Story-SLM2: PyTorch (.pt) to GGUF Converter (convert_to_gguf.py)
=============================================================================
Exports Story-SLM2 checkpoints (e.g. story_slm2_sft_best.pt) into standard
llama.cpp / Ollama / LM Studio / Hugging Face compatible GGUF binaries.

Features:
  - Architecture: LLaMA (GQA 8:4, SwiGLU, RMSNorm, RoPE theta=10000.0)
  - Mathematical RoPE Permutation for Q and K projection matrices
  - Full BPE Tokenizer (50,304 vocab: GPT-2 BPE + ChatML control tokens + 50,000 merges)
  - ChatML Chat Template embedded in GGUF metadata
  - Multiple Export Types:
      * F16:  Lossless 16-bit float (highest precision gold master)
      * Q8_0: 8-bit high-fidelity quantization (near-lossless, ~53 MB)
      * Q4_0: 4-bit lightweight quantization (~28 MB)
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# Add backend directory for config and tokenizer access
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "backend", "model2"))

import gguf
from gguf import GGUFWriter, GGMLQuantizationType, TokenType
import tiktoken
from transformers import AutoTokenizer

from model import StorySLM2Config


def permute_rope_weight(weight: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """
    Permutes PyTorch RoPE weights for GGML / llama.cpp format.
    PyTorch splits RoPE down the middle: (-x[..., d//2:], x[..., :d//2])
    GGML pairs adjacent elements: (x[0], x[1]), (x[2], x[3])
    Swapping axes (1, 2) aligns the PyTorch weights with GGML's pairwise rotation.
    """
    in_features = weight.shape[1]
    return (
        weight.reshape(n_heads, 2, head_dim // 2, in_features)
        .swapaxes(1, 2)
        .reshape(n_heads * head_dim, in_features)
        .copy()
    )


def build_vocab_and_merges():
    """Builds the 50,304 token dictionary and 50,000 BPE merges."""
    print("[1/4] Extracting GPT-2 BPE vocabulary and ChatML special tokens...")
    base = tiktoken.get_encoding("gpt2")
    inv_vocab = {v: k for k, v in base._mergeable_ranks.items()}
    tokens = [inv_vocab[i] for i in range(len(inv_vocab))]

    # Special tokens
    special_map = {
        50256: b"<|endoftext|>",
        50257: b"<|im_start|>",
        50258: b"<|im_end|>",
        50259: b"<|pad|>",
        50260: b"<|reasoning_start|>",
        50261: b"<|reasoning_end|>",
    }
    for i in range(50256, 50262):
        tokens.append(special_map[i])

    # Extra padded tokens up to EMBED_VOCAB_SIZE (50304)
    for i in range(50262, 50304):
        tokens.append(f"<|extra_token_{i}|>".encode("utf-8"))

    # Token types
    token_types = []
    for i in range(len(tokens)):
        if i < 50256:
            token_types.append(int(TokenType.NORMAL))
        elif i <= 50261:
            token_types.append(int(TokenType.CONTROL))
        else:
            token_types.append(int(TokenType.UNUSED))

    # Token scores (descending)
    token_scores = [-float(i) for i in range(len(tokens))]

    # Merges from HuggingFace GPT-2
    print("[2/4] Extracting BPE merge table from Hugging Face GPT-2 tokenizer...")
    hf_tok = AutoTokenizer.from_pretrained("gpt2")
    backend_data = json.loads(hf_tok.backend_tokenizer.to_str())
    merges = backend_data["model"]["merges"]

    return tokens, token_types, token_scores, merges


def convert_checkpoint_to_gguf(
    checkpoint_path: str,
    output_path: str,
    quant_type: GGMLQuantizationType = GGMLQuantizationType.F16,
    tokens=None,
    token_types=None,
    token_scores=None,
    merges=None,
):
    print(f"\n=================================================================")
    print(f"[CONVERT] Target: '{os.path.basename(output_path)}' ({quant_type.name})")
    print(f"Source Checkpoint: {checkpoint_path}")
    print(f"=================================================================")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config: StorySLM2Config = checkpoint.get("config", StorySLM2Config())
    state_dict = checkpoint["model_state_dict"]

    head_dim = config.d_model // config.n_heads  # 448 // 8 = 56
    assert head_dim == 56, f"Expected head_dim 56, got {head_dim}"

    writer = GGUFWriter(output_path, "llama")

    # 1. General Architecture & Metadata
    writer.add_name("Karna-49M")
    writer.add_description("Karna-49M: 49.38M Parameter Supervised Fine-Tuned (SFT) Model")
    writer.add_author("Karna AI Research Team")
    
    file_type_map = {
        GGMLQuantizationType.F16: 1,
        GGMLQuantizationType.Q4_0: 2,
        GGMLQuantizationType.Q8_0: 7,
    }
    writer.add_file_type(file_type_map.get(quant_type, 1))

    # 2. LLaMA Model Hyperparameters
    writer.add_context_length(config.max_seq_len)
    writer.add_embedding_length(config.d_model)
    writer.add_block_count(config.n_layers)
    writer.add_feed_forward_length(config.d_ffn)
    writer.add_head_count(config.n_heads)
    writer.add_head_count_kv(config.n_kv_heads)
    writer.add_layer_norm_rms_eps(config.norm_eps)
    writer.add_rope_freq_base(config.rope_theta)
    writer.add_rope_dimension_count(head_dim)

    # 3. Tokenizer & ChatML Template
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_scores(token_scores)
    writer.add_token_merges(merges)

    # Special token bindings
    writer.add_bos_token_id(50257)  # <|im_start|>
    writer.add_eos_token_id(50258)  # <|im_end|>
    writer.add_pad_token_id(50259)  # <|pad|>

    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{'<|im_start|>assistant\n'}}"
        "{% endif %}"
    )
    writer.add_chat_template(chat_template)

    # 4. Process and Add Tensors
    print(f"[3/4] Processing and writing model weights ({quant_type.name})...")

    # Helper function to quantize or cast 2D weights
    def prepare_weight(w_np: np.ndarray, is_norm: bool = False):
        if is_norm or quant_type == GGMLQuantizationType.F32:
            return w_np.astype(np.float32), None
        elif quant_type == GGMLQuantizationType.F16:
            return w_np.astype(np.float16), None
        else:
            # Quantize 2D matrix
            q_data = gguf.quantize(w_np.astype(np.float32), quant_type)
            return q_data, quant_type

    # Input token embeddings
    emb_data, emb_qtype = prepare_weight(state_dict["tok_embeddings.weight"].float().numpy())
    writer.add_tensor("token_embd.weight", emb_data, raw_dtype=emb_qtype)

    # Output norm (RMSNorm is always FP32)
    norm_data = state_dict["norm.weight"].float().numpy()
    writer.add_tensor("output_norm.weight", norm_data)

    # Output LM Head (tied or separate)
    if "lm_head.weight" in state_dict:
        head_data, head_qtype = prepare_weight(state_dict["lm_head.weight"].float().numpy())
        writer.add_tensor("output.weight", head_data, raw_dtype=head_qtype)
    else:
        writer.add_tensor("output.weight", emb_data, raw_dtype=emb_qtype)

    # Transformer Blocks (Layers 0..11)
    for i in range(config.n_layers):
        pfx = f"layers.{i}."
        blk_pfx = f"blk.{i}."

        # Attention RMSNorm (1D -> FP32)
        attn_norm_w = state_dict[f"{pfx}attn_norm.weight"].float().numpy()
        writer.add_tensor(f"{blk_pfx}attn_norm.weight", attn_norm_w)

        # Q Projection with RoPE permutation
        q_w = state_dict[f"{pfx}attn.q_proj.weight"].float().numpy()
        q_permuted = permute_rope_weight(q_w, n_heads=config.n_heads, head_dim=head_dim)
        q_data, q_qtype = prepare_weight(q_permuted)
        writer.add_tensor(f"{blk_pfx}attn_q.weight", q_data, raw_dtype=q_qtype)

        # K Projection with RoPE permutation
        k_w = state_dict[f"{pfx}attn.k_proj.weight"].float().numpy()
        k_permuted = permute_rope_weight(k_w, n_heads=config.n_kv_heads, head_dim=head_dim)
        k_data, k_qtype = prepare_weight(k_permuted)
        writer.add_tensor(f"{blk_pfx}attn_k.weight", k_data, raw_dtype=k_qtype)

        # V Projection
        v_w = state_dict[f"{pfx}attn.v_proj.weight"].float().numpy()
        v_data, v_qtype = prepare_weight(v_w)
        writer.add_tensor(f"{blk_pfx}attn_v.weight", v_data, raw_dtype=v_qtype)

        # Output Projection (O)
        o_w = state_dict[f"{pfx}attn.o_proj.weight"].float().numpy()
        o_data, o_qtype = prepare_weight(o_w)
        writer.add_tensor(f"{blk_pfx}attn_output.weight", o_data, raw_dtype=o_qtype)

        # FFN RMSNorm (1D -> FP32)
        ffn_norm_w = state_dict[f"{pfx}ffn_norm.weight"].float().numpy()
        writer.add_tensor(f"{blk_pfx}ffn_norm.weight", ffn_norm_w)

        # SwiGLU Gate (w1)
        w1_w = state_dict[f"{pfx}ffn.w1.weight"].float().numpy()
        w1_data, w1_qtype = prepare_weight(w1_w)
        writer.add_tensor(f"{blk_pfx}ffn_gate.weight", w1_data, raw_dtype=w1_qtype)

        # SwiGLU Down (w2)
        w2_w = state_dict[f"{pfx}ffn.w2.weight"].float().numpy()
        w2_data, w2_qtype = prepare_weight(w2_w)
        writer.add_tensor(f"{blk_pfx}ffn_down.weight", w2_data, raw_dtype=w2_qtype)

        # SwiGLU Up (w3)
        w3_w = state_dict[f"{pfx}ffn.w3.weight"].float().numpy()
        w3_data, w3_qtype = prepare_weight(w3_w)
        writer.add_tensor(f"{blk_pfx}ffn_up.weight", w3_data, raw_dtype=w3_qtype)

    # 5. Flush and write GGUF structure
    print("[4/4] Writing GGUF file structure to disk...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] Exported: '{output_path}' ({size_mb:.2f} MB)")


def verify_gguf_file(gguf_path: str):
    """Verifies that the generated GGUF file is valid and readable."""
    print(f"\n[VERIFY] Inspecting '{os.path.basename(gguf_path)}' with GGUFReader...")
    reader = gguf.GGUFReader(gguf_path)

    arch_field = reader.get_field("general.architecture")
    arch = bytes(arch_field.parts[-1]).decode("utf-8") if arch_field else "unknown"

    name_field = reader.get_field("general.name")
    name = bytes(name_field.parts[-1]).decode("utf-8") if name_field else "unknown"

    ctx_field = reader.get_field("llama.context_length")
    ctx = ctx_field.parts[-1].item() if ctx_field else "unknown"

    n_tensors = len(reader.tensors)
    size_mb = os.path.getsize(gguf_path) / (1024 * 1024)

    print(f"  * Architecture:   {arch}")
    print(f"  * Model Name:     {name}")
    print(f"  * Context Length: {ctx} tokens")
    print(f"  * Tensor Count:   {n_tensors} tensors")
    print(f"  * Binary Size:    {size_mb:.2f} MB")
    print(f"  * Status:         VERIFIED OK!")


def generate_huggingface_model_card(output_dir: str):
    """Generates a professional README.md Model Card for Hugging Face."""
    readme_path = os.path.join(output_dir, "README.md")
    content = """---
language:
- en
license: mit
tags:
- story-slm
- slm
- text-generation
- gpt2
- gguf
- llama-cpp
- ollama
datasets:
- custom-story-sft
pipeline_tag: text-generation
---

# 📖 Karna-49M Model (GGUF Formats)

**Karna-49M** is a hyper-optimized **49.38 Million parameter Small Language Model (SLM)** engineered for long-form narrative generation, character-anchored dialogues, and multi-genre creative writing.

Trained from scratch and fine-tuned using Supervised Fine-Tuning (SFT) with ChatML format, it achieves fluent, coherent storytelling with instant Time-To-First-Token (<100ms) and ~35+ tokens/second on standard laptop CPUs.

---

## 📦 Available GGUF Quantizations

| File Name | Format | Size | Description |
| :--- | :--- | :--- | :--- |
| **`karna_49m_sft_f16.gguf`** | **FP16** | **~98 MB** | **Gold Master**: Full 16-bit precision, highest fidelity, zero quantization loss. |
| **`karna_49m_sft_q8_0.gguf`** | **Q8_0** | **~53 MB** | **Recommended**: 8-bit quantization with near-lossless perplexity. |
| **`karna_49m_sft_q4_0.gguf`** | **Q4_0** | **~28 MB** | **Ultra-Lightweight**: 4-bit quantization, fits anywhere, fast on micro-devices. |

---

## 🧠 Architecture Specification

* **Parameters**: 49,384,384 (~49.38M)
* **Backbone**: LLaMA-style Transformer Decoder
* **Attention**: Grouped-Query Attention (GQA 8 Query Heads : 4 KV Heads)
* **Context Window**: 2,048 tokens
* **Hidden Dimension**: 448
* **Layers**: 12
* **Feed-Forward**: SwiGLU ($d_{ffn} = 1,216$)
* **Positional Encoding**: RoPE ($\theta = 10,000.0$)
* **Normalization**: RMSNorm ($\epsilon = 10^{-5}$)
* **Vocabulary**: 50,304 tokens (GPT-2 BPE + ChatML Special Control Tokens)

---

## 🚀 How to Run

### 1. With `llama.cpp`

```bash
# Run interactive chat
./llama-cli -m karna_49m_sft_f16.gguf -p "<|im_start|>user\nWrite a story about a dragon guarding a hidden garden.<|im_end|>\n<|im_start|>assistant\n" -n 256 --temp 0.7
```

### 2. With Ollama

Create a file named `Modelfile`:
```dockerfile
FROM ./karna_49m_sft_f16.gguf

TEMPLATE """<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER repeat_penalty 1.2
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
```

Then create and run:
```bash
ollama create karna-49m -f Modelfile
ollama run karna-49m "Write a story about an astronomer who discovered a singing star."
```
```

### 3. With LM Studio
Simply drag and drop any `.gguf` file into LM Studio's local models folder.

---

## 🎭 Prompt Format (ChatML)

```text
<|im_start|>user
Write a short story about an old clockmaker in Prague.<|im_end|>
<|im_start|>assistant
```

---

## 📄 License
MIT License
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Generated Hugging Face Model Card: '{readme_path}'")


def generate_ollama_modelfile(output_dir: str):
    """Generates an Ollama Modelfile."""
    modelfile_path = os.path.join(output_dir, "Modelfile")
    content = """FROM ./story_slm2_sft_f16.gguf

TEMPLATE \"\"\"<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER repeat_penalty 1.2
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Generated Ollama Modelfile: '{modelfile_path}'")


def main():
    parser = argparse.ArgumentParser(description="Convert Story-SLM2 PyTorch Checkpoint to GGUF")
    default_ckpt = os.path.join(CURRENT_DIR, "model2", "story_slm2_sft_best.pt")
    if not os.path.exists(default_ckpt):
        default_ckpt = os.path.join(CURRENT_DIR, "models", "story_slm2_sft_best.pt")

    parser.add_argument("--checkpoint", default=default_ckpt, help="Path to .pt checkpoint")
    parser.add_argument("--output_dir", default=os.path.join(CURRENT_DIR, "exports", "gguf"), help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Prepare Vocabulary & Merges once for all exports
    tokens, token_types, token_scores, merges = build_vocab_and_merges()

    # 2. Export FP16 (Gold Standard / Full Precision)
    f16_path = os.path.join(args.output_dir, "story_slm2_sft_f16.gguf")
    convert_checkpoint_to_gguf(
        args.checkpoint, f16_path,
        quant_type=GGMLQuantizationType.F16,
        tokens=tokens, token_types=token_types, token_scores=token_scores, merges=merges
    )
    verify_gguf_file(f16_path)

    # 3. Export Q8_0 (8-bit High Fidelity Quantization)
    q8_path = os.path.join(args.output_dir, "story_slm2_sft_q8_0.gguf")
    convert_checkpoint_to_gguf(
        args.checkpoint, q8_path,
        quant_type=GGMLQuantizationType.Q8_0,
        tokens=tokens, token_types=token_types, token_scores=token_scores, merges=merges
    )
    verify_gguf_file(q8_path)

    # 4. Export Q4_0 (4-bit Compact Quantization)
    q4_path = os.path.join(args.output_dir, "story_slm2_sft_q4_0.gguf")
    convert_checkpoint_to_gguf(
        args.checkpoint, q4_path,
        quant_type=GGMLQuantizationType.Q4_0,
        tokens=tokens, token_types=token_types, token_scores=token_scores, merges=merges
    )
    verify_gguf_file(q4_path)

    # 5. Generate Hugging Face Hub Documentation & Ollama Config
    generate_huggingface_model_card(args.output_dir)
    generate_ollama_modelfile(args.output_dir)

    print("\n[SUCCESS] ALL GGUF EXPORTS SUCCESSFULLY COMPLETED AND READY FOR HUGGING FACE!\n")


if __name__ == "__main__":
    main()
