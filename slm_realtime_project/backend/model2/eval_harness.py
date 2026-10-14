"""
=============================================================================
Story-SLM2: Evaluation Harness (eval_harness.py)
=============================================================================
A fixed, repeatable "exam" for the model: the same prompts, generated with
the same seeds and settings, scored by the same automatic rules, every time.

This exists because manual chat testing proved unreliable: a few random
samples led to a confidently wrong "v1 is better, revert" conclusion that
only fell apart once the exact same prompts were run side-by-side under
controlled conditions. This script IS that controlled condition, saved as a
reusable tool instead of a one-off comparison script.

Honest limitation: judging whether a story is "genuinely sad" is inherently
fuzzy. These checks are automatable approximations (word counts, name-mention
counts, emotion-keyword presence, before/after keyword position for pivots),
not true semantic understanding. They are consistent and fair, not perfect --
which is still a large upgrade over ad-hoc impressions.

Usage:
    python eval_harness.py <checkpoint1.pt> [checkpoint2.pt ...]
=============================================================================
"""

import sys
import torch

from model import StorySLM2, StorySLM2Config
from tokenizer import StorySLM2Tokenizer
from generator import StorySLM2Generator


def check_is_short_reply(response: str, max_words: int = 40) -> bool:
    """Conversational prompts should get a short reply, not a story."""
    words = response.split()
    narrative_markers = ["once upon a time", "one day", "lived in", "there was a"]
    looks_like_story = any(marker in response.lower() for marker in narrative_markers)
    return len(words) <= max_words and not looks_like_story


def check_name_anchored(response: str, names: list, min_mentions: int = 2) -> bool:
    """Does the story keep returning to the requested subject's name,
    instead of drifting into an unrelated invented character?"""
    lower = response.lower()
    return any(lower.count(name.lower()) >= min_mentions for name in names)


def check_keyword_presence(response: str, keywords: list, min_hits: int = 2) -> bool:
    """Does the response contain enough vocabulary associated with the
    requested emotion, rather than just naming it once in passing?"""
    lower = response.lower()
    hits = sum(1 for kw in keywords if kw in lower)
    return hits >= min_hits


def check_pivot(response: str, first_keywords: list, second_keywords: list) -> bool:
    """Checks for a genuine emotional pivot: first-emotion vocabulary
    concentrated in the first half of the response, second-emotion
    vocabulary concentrated in the second half -- a cheap, automatable
    proxy for 'did it actually transition' rather than skip straight to
    one flat tone."""
    words = response.split()
    midpoint = len(words) // 2
    first_half = " ".join(words[:midpoint]).lower()
    second_half = " ".join(words[midpoint:]).lower()
    first_ok = any(kw in first_half for kw in first_keywords)
    second_ok = any(kw in second_half for kw in second_keywords)
    return first_ok and second_ok


def check_word_count_range(response: str, min_words: int = 0, max_words: int = 10_000) -> bool:
    n = len(response.split())
    return min_words <= n <= max_words


def check_multi_topic(response: str, topic_groups: list) -> bool:
    """For 'write two stories, one about X one about Y' -- checks that
    vocabulary from EACH requested topic actually appears somewhere."""
    lower = response.lower()
    return all(any(kw in lower for kw in group) for group in topic_groups)


TEST_SUITE = [
    # --- conversational_short: should get a short reply, not a story ---
    {"id": "conv_hi", "category": "conversational", "prompt": "Hi",
     "gen": dict(max_new_tokens=60, temperature=0.75, top_k=40),
     "check": lambda r: check_is_short_reply(r)},
    {"id": "conv_thanks", "category": "conversational", "prompt": "Thanks!",
     "gen": dict(max_new_tokens=60, temperature=0.75, top_k=40),
     "check": lambda r: check_is_short_reply(r)},
    {"id": "conv_who", "category": "conversational", "prompt": "Who are you?",
     "gen": dict(max_new_tokens=60, temperature=0.75, top_k=40),
     "check": lambda r: check_is_short_reply(r)},
    {"id": "conv_bye", "category": "conversational", "prompt": "Goodbye",
     "gen": dict(max_new_tokens=60, temperature=0.75, top_k=40),
     "check": lambda r: check_is_short_reply(r)},

    # --- anchoring: known vs. obscure named subjects ---
    {"id": "anchor_hercules", "category": "anchoring", "prompt": "Write a long story about Hercules.",
     "gen": dict(max_new_tokens=250, temperature=0.5, top_k=25),
     "check": lambda r: check_name_anchored(r, ["Hercules"])},
    {"id": "anchor_hanuman", "category": "anchoring", "prompt": "Write a story about Hanuman.",
     "gen": dict(max_new_tokens=250, temperature=0.5, top_k=25),
     "check": lambda r: check_name_anchored(r, ["Hanuman"])},
    {"id": "anchor_persephone", "category": "anchoring", "prompt": "I want a story with Persephone in it.",
     "gen": dict(max_new_tokens=250, temperature=0.5, top_k=25),
     "check": lambda r: check_name_anchored(r, ["Persephone"])},
    {"id": "anchor_karna", "category": "anchoring", "prompt": "Write a story about Suryaputra Karna.",
     "gen": dict(max_new_tokens=250, temperature=0.5, top_k=25),
     "check": lambda r: check_name_anchored(r, ["Karna"])},

    # --- single emotion: does it actually depict the feeling? ---
    {"id": "emotion_scary", "category": "emotion", "prompt": "Write a scary story about a lighthouse keeper.",
     "gen": dict(max_new_tokens=300, temperature=0.7, top_k=40),
     "check": lambda r: check_keyword_presence(
         r, ["dark", "afraid", "fear", "scar", "trembl", "shadow", "strange noise", "heart raced", "nervous"])},
    {"id": "emotion_sad", "category": "emotion", "prompt": "Write a sad story about a dog waiting for its owner.",
     "gen": dict(max_new_tokens=300, temperature=0.7, top_k=40),
     "check": lambda r: check_keyword_presence(
         r, ["sad", "lonely", "miss", "cry", "tear", "longing", "empty", "alone", "ache"])},
    {"id": "emotion_excited", "category": "emotion", "prompt": "Write an exciting story about a race car driver.",
     "gen": dict(max_new_tokens=300, temperature=0.7, top_k=40),
     "check": lambda r: check_keyword_presence(
         r, ["excit", "thrill", "fast", "race", "heart pound", "rush", "adrenaline", "cheer"])},

    # --- emotional pivot: genuine first emotion, then a genuine second emotion ---
    {"id": "pivot_scary_happy", "category": "pivot",
     "prompt": "Write a story about a firefighter that starts out scary and ends happy.",
     "gen": dict(max_new_tokens=500, temperature=0.6, top_k=30),
     "check": lambda r: check_pivot(
         r, ["dark", "afraid", "fear", "scar", "trembl", "danger", "smoke", "panic"],
         ["happy", "smil", "relie", "safe", "joy", "warm", "grateful", "proud"])},
    {"id": "pivot_nervous_proud", "category": "pivot",
     "prompt": "Write a story about a girl who feels nervous but ends up proud.",
     "gen": dict(max_new_tokens=500, temperature=0.6, top_k=30),
     "check": lambda r: check_pivot(
         r, ["nervous", "worried", "shak", "anxious", "butterflies"],
         ["proud", "confiden", "accomplish", "smil", "cheer"])},

    # --- instruction-following edge cases ---
    {"id": "instr_short", "category": "instruction", "prompt": "tell me a story but make it really short",
     "gen": dict(max_new_tokens=100, temperature=0.75, top_k=40),
     "check": lambda r: check_word_count_range(r, min_words=5, max_words=60)},
    {"id": "instr_long", "category": "instruction", "prompt": "write a very very long story about a king",
     "gen": dict(max_new_tokens=700, temperature=0.75, top_k=40),
     "check": lambda r: check_word_count_range(r, min_words=300, max_words=10_000)},
    {"id": "instr_two_stories", "category": "instruction",
     "prompt": "can you write two short stories, one about a cat and one about a dog",
     "gen": dict(max_new_tokens=400, temperature=0.75, top_k=40),
     "check": lambda r: check_multi_topic(r, [["cat", "kitten"], ["dog", "puppy"]])},
]


@torch.inference_mode()
def run_eval(checkpoint_path: str, seed: int = 0):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config: StorySLM2Config = checkpoint["config"]
    model = StorySLM2(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = StorySLM2Tokenizer()
    generator = StorySLM2Generator(model, tokenizer, "cpu")

    results = []
    for test in TEST_SUITE:
        torch.manual_seed(seed)
        response = "".join(generator.stream_generate(test["prompt"], **test["gen"]))
        passed = test["check"](response)
        results.append({"id": test["id"], "category": test["category"], "passed": passed, "response": response})
    return results


def print_report(label: str, results: list):
    print(f"\n=== {label} ===")
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    for category, items in by_category.items():
        passed = sum(1 for i in items if i["passed"])
        print(f"\n[{category}] {passed}/{len(items)} passed")
        for i in items:
            mark = "PASS" if i["passed"] else "FAIL"
            print(f"  {mark}  {i['id']}")

    total_passed = sum(1 for r in results if r["passed"])
    print(f"\nTOTAL: {total_passed}/{len(results)} passed")


def print_comparison(all_results: dict):
    """all_results: {checkpoint_label: results_list}"""
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    labels = list(all_results.keys())
    test_ids = [t["id"] for t in TEST_SUITE]

    header = f"{'test':24s}" + "".join(f"{label:>18s}" for label in labels)
    print(header)
    for tid in test_ids:
        row = f"{tid:24s}"
        for label in labels:
            r = next(r for r in all_results[label] if r["id"] == tid)
            row += f"{'PASS' if r['passed'] else 'FAIL':>18s}"
        print(row)

    print()
    for label in labels:
        total = sum(1 for r in all_results[label] if r["passed"])
        print(f"{label}: {total}/{len(TEST_SUITE)} passed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_harness.py <checkpoint1.pt> [checkpoint2.pt ...]")
        sys.exit(1)

    checkpoint_paths = sys.argv[1:]
    all_results = {}
    for path in checkpoint_paths:
        label = path.split("\\")[-1].split("/")[-1]
        print(f"\nRunning eval on {label}...")
        results = run_eval(path)
        all_results[label] = results
        print_report(label, results)

    if len(checkpoint_paths) > 1:
        print_comparison(all_results)
