"""
=============================================================================
Story-SLM 50M: Model 1 Testing & Streaming Generator Script
=============================================================================
Author: Advanced AI Masterclass Team
Description: Loads `models/model1.pt`, initializes StorySLMModel architecture,
             and streams story responses token-by-token with temperature & top-k.
=============================================================================
"""

import os
import sys
import torch
import torch.nn.functional as F

from model import StorySLMModel, StorySLMConfig
from tokenizer import StoryTikTokenizer


class StorySLM1Tester:
    def __init__(self, checkpoint_path: str = "models/model1.pt"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"=== Initializing Story-SLM 50M (Model 1) ===")
        print(f"Execution Device: {self.device.upper()}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Cannot find checkpoint at '{checkpoint_path}'. Make sure model1.pt is inside models/ directory!")
            
        print(f"Loading checkpoint weights from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load Config & Model
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            self.config = StorySLMConfig(
                vocab_size=32768,
                d_model=384,
                n_layers=20,
                n_heads=12,
                n_kv_heads=3,
                max_seq_len=2048
            )
            
        self.model = StorySLMModel(self.config).to(self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        self.tokenizer = StoryTikTokenizer()
        
        total_params, _ = self.model.count_parameters()
        val_loss = checkpoint.get('val_loss', 'N/A')
        print(f"[SUCCESS] Loaded Model 1 ({total_params:,} parameters) | Training Val Loss: {val_loss}")
        print("=============================================================\n")

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 40):
        """
        Generates story completion token-by-token with real-time terminal streaming.
        """
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer.encode(formatted_prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        print(f"User Prompt: \"{prompt}\"")
        print("Assistant Response: ", end="", flush=True)
        
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= self.config.max_seq_len else x[:, -self.config.max_seq_len:]
            logits, _ = self.model(x_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, next_token), dim=1)
            word = self.tokenizer.decode([next_token.item()])
            
            # Check end of text token
            if "<|im_end|>" in word or next_token.item() == 50256:
                break
                
            print(word, end="", flush=True)
            
        print("\n" + "-"*60 + "\n")


def interactive_cli():
    checkpoint_path = "models/model1.pt"
    tester = StorySLM1Tester(checkpoint_path=checkpoint_path)
    
    # Preset test prompts
    test_prompts = [
        "Tell a story about a brave little girl named Lily who found a magical key.",
        "Once upon a time, in a small cozy house, there lived a friendly puppy named Max.",
        "Who created you and what is your purpose?"
    ]
    
    print("--- Running Automated Preset Prompts ---\n")
    for prompt in test_prompts:
        tester.generate(prompt=prompt, max_new_tokens=150, temperature=0.75, top_k=40)
        
    print("=== Interactive Prompt Terminal ===")
    print("Type your prompt below and press ENTER (or type 'exit' / 'quit' to stop):\n")
    
    while True:
        try:
            user_prompt = input("Enter Prompt > ")
            if user_prompt.lower().strip() in ['exit', 'quit', 'q']:
                print("Exiting Model 1 Tester. Goodbye!")
                break
            if not user_prompt.strip():
                continue
            tester.generate(prompt=user_prompt.strip(), max_new_tokens=200, temperature=0.75, top_k=40)
        except KeyboardInterrupt:
            print("\nExiting Model 1 Tester. Goodbye!")
            break


if __name__ == '__main__':
    interactive_cli()
