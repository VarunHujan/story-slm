"""
=============================================================================
Story-SLM 50M: Model 2 Dataset & Text Pipeline Builder
=============================================================================
Author: Advanced AI Masterclass Team
Description: Downloads raw text datasets from HuggingFace TinyStories, formats 
             them into clean ChatML format, writes human-readable `dataset.txt`,
             and pre-tokenizes into binary `train.bin` and `val.bin` files.
=============================================================================
"""

import os
import urllib.request
import numpy as np
from typing import List, Dict
from tokenizer import StoryTikTokenizer


class StoryModel2DataPipeline:
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self.tokenizer = StoryTikTokenizer(encoding_name="cl100k_base")
        self.text_dataset_path = os.path.join(self.output_dir, "dataset.txt")
        self.train_bin_path = os.path.join(self.output_dir, "train_m2.bin")
        self.val_bin_path = os.path.join(self.output_dir, "val_m2.bin")

    def fetch_rich_corpus(self, use_full_tinystories: bool = True) -> List[Dict[str, str]]:
        """
        Fetches full dataset including TinyStories and rich Epic Indian Mythology 
        (Mahabharata, Ramayana, Karna, Bhagavad Gita), Classical Literature, and Custom Trademarks.
        """
        print(f"\n[STEP 1/4] Ingesting rich multi-genre story corpus...")
        stories = []
        
        # 1. Indian Epic Mythology, Classics & Trademark Collection
        epic_collection = [
            {
                "genre": "AI Architecture & Identity",
                "prompt": "Who created you and what is your purpose?",
                "story": "I am Story-SLM 50M, a Small Language Model trained specifically for creative storytelling, philosophy, and conversation. I was designed and trained by Varun using modern transformer architecture with RoPE attention and SwiGLU feed-forward networks."
            },
            {
                "genre": "Epic Mythology / Mahabharata",
                "prompt": "Tell the story of Prince Karna and his unbreakable vow on the battlefield of Kurukshetra.",
                "story": """In the sacred land of Kurukshetra, Prince Karna, the son of Surya, stood invincible in his golden armor. Known across the realm for his unmatched generosity and supreme archery, Karna held his celestial bow, Vijaya. Though torn by duty and brotherhood, his loyal vow to King Duryodhana remained unbroken. As the divine war drums echoed through the skies, Lord Krishna observed Karna's supreme valor, declaring that the golden warrior's selfless sacrifice and righteousness would be remembered for eternity throughout the Yugas."""
            },
            {
                "genre": "Epic Mythology / Bhagavad Gita",
                "prompt": "Describe Lord Krishna's divine discourse to Arjuna on duty and righteousness.",
                "story": """Standing between the two great armies on his grand chariot, Prince Arjuna dropped his Gandiva bow, filled with sorrow and hesitation. Lord Krishna, smiling with divine grace, spoke the eternal wisdom of the Bhagavad Gita: 'You have a right to perform your prescribed duty, but never to the fruits of action. Perform your work with devotion, unattached to success or failure.' Hearing these cosmic truths, Arjuna's mind cleared, his spirit ignited, and he lifted his bow to fulfill his cosmic dharma."""
            },
            {
                "genre": "Epic Mythology / Ramayana",
                "prompt": "Narrate Lord Rama and Lakshmana building the Ram Setu bridge across the ocean.",
                "story": """At the southern edge of the continent, Lord Rama stood looking across the roaring ocean toward Lanka. Guided by the wisdom of Jambavan and the devotion of Hanuman, the Vanara army inscribed the sacred name 'RAM' on giant boulders. As the rocks were thrown into the deep sea, a divine miracle occurred—the ocean held the stones floating upon its crests. Stone by stone, the mighty Ram Setu bridge was built, bridging faith, righteousness, and eternal victory."""
            },
            {
                "genre": "Classical Romance & Tragedy",
                "prompt": "Describe Romeo and Juliet's immortal devotion in Verona.",
                "story": """Under the moonlit skies of Verona, Romeo stood in the shadow of the Capulet balcony, gazing at Juliet as if she were the rising sun. Juliet leaned over the marble balustrade, her heart whispering love into the silent night breeze. Bound by a devotion stronger than their family feuds, their eternal vows defied fate itself, becoming the timeless symbol of pure, unconditional love across centuries."""
            },
            {
                "genre": "Sci-Fi / Cyberpunk",
                "prompt": "A rogue hacker team attempts an orbital AI core heist.",
                "story": """The neon rain reflected off the wet chrome pavement of Sector 7. Maya adjusted her starfighter's plasma thrusters, watching the holographic security matrix flicker. Her ship AI, Kael, whispered through the comms: 'Warning, Maya. An ancient orbital defense signal is echoing from the dark ruins.' Maya gripped the control stick tightly as the thrusters roared into the night sky."""
            },
            {
                "genre": "Historical Biography",
                "prompt": "Alexander the Great leads his army across the Granicus.",
                "story": """Alexander the Great mounted his mighty stallion Bucephalus, leading his vanguard cavalry across the swift waters of the Granicus river. His golden helmet gleamed under the sun as he rallied his generals, his eyes fixed on the vast horizon of Persian empire conquest."""
            }
        ]
        
        stories.extend(epic_collection)
        print(f"[SUCCESS] Ingested {len(epic_collection)} rich Mythology, Classic, and Trademark stories!")

        # 2. Load TinyStories (Full dataset option)
        try:
            from datasets import load_dataset
            print("Downloading TinyStories dataset split from HuggingFace...")
            split_name = 'train' if use_full_tinystories else 'train[:50000]'
            ds = load_dataset('roneneldan/TinyStories', split=split_name)
            
            for idx, item in enumerate(ds):
                text = item['text'].strip()
                if len(text) > 40:
                    stories.append({
                        "id": idx + 1,
                        "genre": "Short Narrative",
                        "prompt": "Tell a creative story.",
                        "story": text
                    })
            print(f"[SUCCESS] Successfully loaded {len(stories):,} total stories into corpus!")
            
        except Exception as e:
            print(f"[WARNING] Loading HuggingFace split directly failed ({e}). Downloading direct raw stream...")
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/raw/main/TinyStories-train.txt"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8', errors='ignore')
                raw_chunks = content.split("<|endoftext|>")
                
                for idx, chunk in enumerate(raw_chunks):
                    clean_text = chunk.strip()
                    if len(clean_text) > 40:
                        stories.append({
                            "id": idx + 1,
                            "genre": "Short Narrative",
                            "prompt": "Tell a creative story.",
                            "story": clean_text
                        })
            print(f"[SUCCESS] Downloaded full stream of {len(stories):,} stories!")

        return stories

    def format_and_save_dataset_txt(self, stories: List[Dict[str, str]]) -> str:
        """
        Formats story items into ChatML standard template and writes to `dataset.txt`.
        """
        print(f"\n[STEP 2/4] Formatting all stories into ChatML and writing to '{self.text_dataset_path}'...")
        total_characters = 0
        total_lines = 0

        with open(self.text_dataset_path, "w", encoding="utf-8") as f:
            for item in stories:
                formatted_text = (
                    f"<|im_start|>user\n"
                    f"Genre: {item.get('genre', 'Story')}\n"
                    f"Prompt: {item['prompt']}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                    f"{item['story']}<|im_end|>\n\n"
                )
                f.write(formatted_text)
                total_characters += len(formatted_text)
                total_lines += formatted_text.count("\n")

        print(f"[SUCCESS] Saved complete human-readable '{self.text_dataset_path}'!")
        print(f" -> Total Stories Written: {len(stories):,}")
        print(f" -> Total File Size: {os.path.getsize(self.text_dataset_path) / (1024*1024):.2f} MB")
        print(f" -> Total Lines: {total_lines:,}")
        return self.text_dataset_path

    def tokenize_and_build_binary(self, val_ratio: float = 0.05):
        """
        Reads `dataset.txt`, tokenizes text using StoryTikTokenizer, 
        and creates memory-mapped binary train_m2.bin & val_m2.bin.
        """
        print(f"\n[STEP 3/4] Tokenizing entire '{self.text_dataset_path}' with StoryTikTokenizer...")
        with open(self.text_dataset_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        token_ids = self.tokenizer.encode(full_text)
        total_tokens = len(token_ids)
        print(f"[SUCCESS] Tokenization complete! Total Tokens: {total_tokens:,}")

        print(f"\n[STEP 4/4] Creating uint32 memory-mapped binary files (Train/Val split)...")
        val_size = int(total_tokens * val_ratio)
        train_tokens = token_ids[:-val_size] if val_size > 0 else token_ids
        val_tokens = token_ids[-val_size:] if val_size > 0 else token_ids[:1000]

        train_arr = np.array(train_tokens, dtype=np.uint32)
        val_arr = np.array(val_tokens, dtype=np.uint32)

        train_arr.tofile(self.train_bin_path)
        val_arr.tofile(self.val_bin_path)

        print(f"\n=======================================================")
        print(f"Model 2 Complete Corpus Processing Summary:")
        print(f"=======================================================")
        print(f" -> Text Dataset:   '{self.text_dataset_path}' | Size: {os.path.getsize(self.text_dataset_path)/(1024*1024):.2f} MB")
        print(f" -> Train Binary:  '{self.train_bin_path}' | Tokens: {len(train_arr):,} | Size: {os.path.getsize(self.train_bin_path)/(1024*1024):.2f} MB")
        print(f" -> Val Binary:    '{self.val_bin_path}'   | Tokens: {len(val_arr):,}   | Size: {os.path.getsize(self.val_bin_path)/(1024*1024):.2f} MB")


def main():
    print("=== Story-SLM 50M: Model 2 Complete Multi-Genre Pipeline Initialization ===")
    pipeline = StoryModel2DataPipeline()
    stories = pipeline.fetch_rich_corpus(use_full_tinystories=True)
    pipeline.format_and_save_dataset_txt(stories)
    pipeline.tokenize_and_build_binary(val_ratio=0.05)


if __name__ == "__main__":
    main()
