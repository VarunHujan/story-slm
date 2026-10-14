"""
=============================================================================
Story-SLM2: FastAPI Streaming Server (server.py)
=============================================================================
Loads checkpoints_model2/latest.pt (produced by train.py) and serves it over
Server-Sent Events for the Next.js frontend.
=============================================================================
"""

import os
import json
import asyncio
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import StorySLM2, StorySLM2Config
from tokenizer import StorySLM2Tokenizer
from generator import StorySLM2Generator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "model2_sft_v2.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
state = {"model": None, "tokenizer": None, "generator": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[SERVER STARTUP] Loading Story-SLM2 from '{CHECKPOINT_PATH}' on {device.upper()}...")

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        config = checkpoint.get("config", StorySLM2Config())

        model = StorySLM2(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        tokenizer = StorySLM2Tokenizer()
        state["model"] = model
        state["tokenizer"] = tokenizer
        state["generator"] = StorySLM2Generator(model, tokenizer, device)

        total_params, _ = model.count_parameters()
        step = checkpoint.get("step", "?")
        val_loss = checkpoint.get("val_loss", "?")
        print(f"[SUCCESS] Loaded Story-SLM2 ({total_params:,} params) | step={step} | val_loss={val_loss}")
    else:
        print(f"[WARNING] No checkpoint at '{CHECKPOINT_PATH}'. Run train.py first. Server running uninitialized.")

    yield


app = FastAPI(title="Story-SLM2 API Server", version="2.0.0", lifespan=lifespan)
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


@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>Story-SLM2 API Server Running</h1><p>Frontend expected at http://localhost:3000</p>"


@app.get("/api/health")
async def health_check():
    model = state["model"]
    return {
        "status": "online",
        "model": "Story-SLM2",
        "device": device,
        "checkpoint_loaded": model is not None,
        "parameters": f"{model.count_parameters()[0]:,}" if model is not None else None,
    }


@app.post("/api/generate/stream")
async def stream_generate(req: GenerateRequest):
    generator = state["generator"]
    if generator is None:
        raise HTTPException(status_code=500, detail="Story-SLM2 not loaded. Train a checkpoint first (train.py).")

    # A plain `def` generator here gets run by Starlette in a background worker
    # thread, with every single yielded chunk crossing back to the event loop --
    # on Windows that per-chunk thread hand-off measured as ~30-40ms of pure
    # overhead per token, on top of the model's own ~20-25ms. Making this an
    # `async def` generator instead lets Starlette iterate it directly on the
    # event loop with no thread-crossing.
    #
    # But without any real `await` between yields, the loop never gets a chance
    # to actually flush each chunk to the socket -- on Windows (ProactorEventLoop)
    # this measurably let every chunk queue up and arrive all at once at the end
    # instead of streaming. `await asyncio.sleep(0)` forces a real yield back to
    # the event loop after each token so it actually gets written immediately.
    async def event_publisher():
        for token_text in generator.stream_generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        ):
            yield f"data: {json.dumps({'token': token_text})}\n\n"
            await asyncio.sleep(0)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_publisher(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("\n[INFO] Starting Story-SLM2 FastAPI Server on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
