"""
=============================================================================
Story-SLM 50M: Production Dataset Builder with Open-Source Data Ingestion
=============================================================================
Author: Advanced AI Masterclass Team
Description: Downloads open-source story datasets (TinyStories, Gutenberg, etc.)
             from HuggingFace/Web, formats them into ChatML templates,
             tokenizes using StoryTikTokenizer, and streams into train.bin/val.bin.
=============================================================================
"""

import os
import json
import urllib.request
import numpy as np
from typing import List, Dict
from tokenizer import StoryTikTokenizer


class StoryDatasetBuilder:
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self.tokenizer = StoryTikTokenizer(encoding_name="cl100k_base")
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def create_sample_story_corpus() -> List[Dict[str, str]]:
        """Sample story templates for local testing."""
        return [
            {
                "genre": "Sci-Fi / Cyberpunk",
                "prompt": "A rogue hacker team attempts an orbital AI core heist.",
                "story": """The neon rain reflected off the wet chrome pavement of Sector 7. Maya adjusted her starfighter's plasma thrusters, watching the holographic security matrix flicker. Her ship AI, Kael, whispered through the comms: 'Warning, Maya. An ancient orbital defense signal is echoing from the dark ruins.' Maya gripped the control stick tightly as the thrusters roared into the night sky."""
            },
            {
                "genre": "Epic Fantasy / Mythology",
                "prompt": "Prince Karna stands at the river before the great battle.",
                "story": """In the ancient kingdom of Avantika, Prince Karna stood at the edge of the sacred river, his golden armor reflecting the morning sun. He held his celestial bow tightly, knowing that his unbreakable vow to Duryodhana would soon be tested in the great war. The rivers whispered ancient mantras as the war drums echoed across the distant valley."""
            },
            {
                "genre": "Classical Romance & Tragedy",
                "prompt": "Romeo speaks beneath Juliet's balcony in Verona.",
                "story": """In the quiet streets of Verona, Romeo whispered beneath the marble balcony, gazing up at Juliet. 'Soft! What light through yonder window breaks? It is the east, and Juliet is the sun.' The night breeze carried the scent of jasmine as Juliet leaned over the stone railing, her heart racing in the moonlight."""
            },
            {
                "genre": "Historical Biography",
                "prompt": "Alexander the Great leads his army across the Granicus.",
                "story": """Alexander the Great mounted his mighty stallion Bucephalus, leading his vanguard cavalry across the swift waters of the Granicus river. His golden helmet gleamed under the sun as he rallied his generals, his eyes fixed on the vast horizon of Persian empire conquest."""
            },
            {
                "genre": "Horror / Mystery",
                "prompt": "An investigator explores an abandoned lighthouse on a foggy coast.",
                "story": """Detective Vance stepped onto the slippery rocks surrounding Blackwood Lighthouse. Thick fog clung to the sea cliffs like a shroud. Inside the rusted iron door, a lone pendulum clock ticked in rhythm with the crashing waves, while faint footsteps echoed from the spiral staircase above."""
            }
        ]

    def download_tinystories_sample(self, limit: int = 500) -> List[Dict[str, str]]:
        """
        Downloads open-source story data from Microsoft's TinyStories dataset repository.
        """
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/raw/main/TinyStories-train.txt"
        stories = []
        print(f"[INFO] Downloading open-source story dataset from HuggingFace ({url})...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8', errors='ignore')
                raw_stories = content.split("<|endoftext|>")
                
                for idx, text in enumerate(raw_stories[:limit]):
                    text_clean = text.strip()
                    if len(text_clean) > 50:
                        stories.append({
                            "genre": "General Narrative / Story",
                            "prompt": f"Write a creative short story (Sample #{idx+1})",
                            "story": text_clean
                        })
            print(f"[SUCCESS] Downloaded {len(stories)} open-source stories from TinyStories!")
        except Exception as e:
            print(f"[WARNING] Download failed ({e}). Falling back to internal story corpus.")
            stories = self.create_sample_story_corpus()

        return stories

    def format_chatml(self, item: Dict[str, str]) -> str:
        """Formats story items into ChatML prompt/response templates."""
        return (
            f"<|im_start|>user\n"
            f"Genre: {item['genre']}\n"
            f"Prompt: {item['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{item['story']}<|im_end|>\n"
        )

    def build_binary_dataset(self, val_ratio: float = 0.1, use_online_data: bool = True):
        """
        Ingests story datasets, tokenizes using StoryTikTokenizer, and streams 
        into uint32 train.bin and val.bin files for fast GPU training.
        """
        if use_online_data:
            corpus = self.download_tinystories_sample(limit=500)
        else:
            corpus = self.create_sample_story_corpus()

        formatted_stories = [self.format_chatml(item) for item in corpus]
        full_text = "\n".join(formatted_stories)

        print(f"[INFO] Formatting complete. Total Characters: {len(full_text):,}")
        print("[INFO] Tokenizing text using StoryTikTokenizer (cl100k_base)...")

        token_ids = self.tokenizer.encode(full_text)
        print(f"[SUCCESS] Tokenization complete! Total Tokens: {len(token_ids):,}")

        # Split into train and validation sets
        n_tokens = len(token_ids)
        val_size = int(n_tokens * val_ratio)
        train_ids = token_ids[:-val_size]
        val_ids = token_ids[-val_size:]

        train_arr = np.array(train_ids, dtype=np.uint32)
        val_arr = np.array(val_ids, dtype=np.uint32)

        train_path = os.path.join(self.output_dir, "train.bin")
        val_path = os.path.join(self.output_dir, "val.bin")

        train_arr.tofile(train_path)
        val_arr.tofile(val_path)

        train_mb = os.path.getsize(train_path) / (1024 * 1024)
        val_mb = os.path.getsize(val_path) / (1024 * 1024)

        print("==========================================================")
        print("[SUCCESS] BINARY DATASET PACKING COMPLETE")
        print("==========================================================")
        print(f"  - Train Dataset: {len(train_arr):,} tokens ({train_mb:.2f} MB) -> {train_path}")
        print(f"  - Val Dataset  : {len(val_arr):,} tokens ({val_mb:.2f} MB) -> {val_path}")
        print("==========================================================")


if __name__ == "__main__":
    builder = StoryDatasetBuilder(output_dir=".")
    builder.build_binary_dataset(val_ratio=0.1, use_online_data=True)
