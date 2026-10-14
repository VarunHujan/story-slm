"""
=============================================================================
Story-SLM 50M: High-Performance Streaming Backend Server
=============================================================================
Author: Advanced AI Masterclass Team
Description: Production FastAPI backend hosting `models/model1.pt` with 
             Server-Sent Events (SSE) streaming token output & JSON APIs.
=============================================================================
"""

import os
import json
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import StorySLMModel, StorySLMConfig
from tokenizer import StoryTikTokenizer

# Global variables for model state
model = None
tokenizer = None
config = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, config, device
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "model1.pt")),
        "models/model1.pt",
        "../models/model1.pt",
        "../../models/model1.pt",
    ]
    checkpoint_path = next((p for p in candidates if os.path.exists(p)), "models/model1.pt")
        
    print(f"\n[SERVER STARTUP] Loading Story-SLM 50M Model from '{checkpoint_path}'...")
    print(f"Execution Target Device: {device.upper()}")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = StorySLMConfig(
                vocab_size=32768,
                d_model=384,
                n_layers=20,
                n_heads=12,
                n_kv_heads=3,
                max_seq_len=2048
            )

        model = StorySLMModel(config).to(device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        tokenizer = StoryTikTokenizer()
        total_params, _ = model.count_parameters()
        print(f"[SUCCESS] Model 1 Loaded ({total_params:,} parameters) | Ready on http://localhost:8000\n")
    else:
        print(f"[WARNING] '{checkpoint_path}' not found! Server running in uninitialized state.")

    yield


app = FastAPI(title="Story-SLM 50M API Server", version="1.0.0", lifespan=lifespan)

# Enable CORS for React / Next.js Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.75
    top_k: int = 40


@app.get('/', response_class=HTMLResponse)
async def get_ui():
    """Serves embedded Web Playground UI."""
    return "<h1>Story-SLM 50M API Server Running</h1><p>Next.js Frontend UI available at http://localhost:3000</p>"


@app.get('/api/health')
async def health_check():
    return {
        "status": "online",
        "model": "Story-SLM 50M (Model 1)",
        "device": device,
        "parameters": "55.3M",
        "checkpoint_loaded": model is not None
    }


@app.post('/api/generate/stream')
async def stream_generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model 1 not loaded. Check model1.pt path.")

    def event_publisher():
        formatted_prompt = f"<|im_start|>user\n{req.prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(formatted_prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        repetition_penalty = 1.25

        with torch.inference_mode():
            for _ in range(req.max_tokens):
                x_cond = x if x.size(1) <= config.max_seq_len else x[:, -config.max_seq_len:]
                logits, _ = model(x_cond)
                logits = logits[:, -1, :] / max(req.temperature, 1e-5)

                # Apply Repetition Penalty to previously generated tokens
                for token_id in set(x[0].tolist()):
                    if token_id < logits.size(-1):
                        if logits[0, token_id] < 0:
                            logits[0, token_id] *= repetition_penalty
                        else:
                            logits[0, token_id] /= repetition_penalty

                if req.top_k is not None:
                    v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, next_token), dim=1)

                token_text = tokenizer.decode([next_token.item()])

                # Stop generation on end-of-conversation control tags
                if "<|im_end|>" in token_text or "<|im_start|>" in token_text or next_token.item() in [50256, 100259, 100258]:
                    break

                t_json = json.dumps({"token": token_text})
                yield f"data: {t_json}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_publisher(), media_type="text/event-stream")


if __name__ == '__main__':
    import uvicorn
    print("\n[INFO] Starting Story-SLM 50M FastAPI Server on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
