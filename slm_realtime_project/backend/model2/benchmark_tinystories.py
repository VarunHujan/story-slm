"""
=============================================================================
Story-SLM2: Benchmark vs. TinyStories-33M (benchmark_tinystories.py)
=============================================================================
Fair-comparison caveat, stated up front: TinyStories-33M is a pure
continuation model (never trained for chat/instructions), while Story-SLM2
was explicitly fine-tuned for it. So this script only compares what BOTH
models can fairly attempt -- given a character's name, does a generated
story stay anchored to that name -- using plain continuation prompts for
TinyStories-33M instead of our chat template. Conversational and
instruction-following tests from eval_harness.py are deliberately excluded
here since TinyStories-33M was never trained to do either.
=============================================================================
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import StorySLM2, StorySLM2Config
from tokenizer import StorySLM2Tokenizer
from generator import StorySLM2Generator
from eval_harness import check_name_anchored


ANCHOR_TESTS = [
    ("Hercules", "Write a long story about Hercules.", "Hercules was"),
    ("Hanuman", "Write a story about Hanuman.", "Hanuman was"),
    ("Persephone", "I want a story with Persephone in it.", "Persephone was"),
    ("Karna", "Write a story about Suryaputra Karna.", "Karna was"),
]


def run_storyslm2(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = StorySLM2(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tok = StorySLM2Tokenizer()
    gen = StorySLM2Generator(model, tok, "cpu")

    results = {}
    for name, prompt, _ in ANCHOR_TESTS:
        torch.manual_seed(0)
        out = "".join(gen.stream_generate(prompt, max_new_tokens=200, temperature=0.5, top_k=25))
        results[name] = (check_name_anchored(out, [name]), out)
    return results


def run_tinystories(cache_dir: str):
    model = AutoModelForCausalLM.from_pretrained(cache_dir)
    tok = AutoTokenizer.from_pretrained(cache_dir)
    model.eval()

    results = {}
    for name, _, continuation_prompt in ANCHOR_TESTS:
        torch.manual_seed(0)
        input_ids = tok(continuation_prompt, return_tensors="pt").input_ids
        output_ids = model.generate(
            input_ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=40,
            pad_token_id=tok.eos_token_id,
        )
        out = tok.decode(output_ids[0], skip_special_tokens=True)
        results[name] = (check_name_anchored(out, [name]), out)
    return results


if __name__ == "__main__":
    print("=== Story-SLM2 (49.38M, SFT v2) ===")
    slm2_results = run_storyslm2(
        r"C:\Users\Varun\Desktop\SLM_Project\slm_realtime_project\models\model2_sft_v2.pt"
    )
    for name, (passed, text) in slm2_results.items():
        print(f"\n[{name}] anchored={passed}")
        print(text[:300])

    print("\n\n=== TinyStories-33M ===")
    ts_results = run_tinystories("tinystories_33m_cache")
    for name, (passed, text) in ts_results.items():
        print(f"\n[{name}] anchored={passed}")
        print(text[:300])

    print("\n\n=== COMPARISON (name-anchoring only -- the one fair axis) ===")
    print(f"{'character':15s}{'Story-SLM2':>15s}{'TinyStories-33M':>18s}")
    for name, _, _ in ANCHOR_TESTS:
        s2 = "PASS" if slm2_results[name][0] else "FAIL"
        ts = "PASS" if ts_results[name][0] else "FAIL"
        print(f"{name:15s}{s2:>15s}{ts:>18s}")

    s2_total = sum(1 for v in slm2_results.values() if v[0])
    ts_total = sum(1 for v in ts_results.values() if v[0])
    print(f"\nStory-SLM2: {s2_total}/4   TinyStories-33M: {ts_total}/4")
