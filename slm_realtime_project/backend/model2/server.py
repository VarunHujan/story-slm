"""
=============================================================================
Karna-49M Model: FastAPI Streaming Server (server.py)
=============================================================================
Loads trained Karna-49M checkpoints and serves over Server-Sent Events (SSE).
=============================================================================
"""

import os
import json
import asyncio
import torch
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import Karna49M, Karna49MConfig, StorySLM2, StorySLM2Config
from tokenizer import Karna49MTokenizer, StorySLM2Tokenizer
from generator import Karna49MGenerator, StorySLM2Generator


# Load .env from backend directory first, then fallback to project root
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Resolve Checkpoint Path with multiple fallback candidates
env_ckpt = os.getenv("CHECKPOINT_PATH")
candidate_paths = [
    env_ckpt if env_ckpt else "",
    os.path.join(PROJECT_ROOT, "models", "story_slm2_sft_best.pt"),
    os.path.join(PROJECT_ROOT, "model2", "story_slm2_sft_best.pt"),
    os.path.join(PROJECT_ROOT, "models", "model2_sft_v2.pt"),
    os.path.join(PROJECT_ROOT, "models", "model2.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints_model2", "latest.pt"),
]
CHECKPOINT_PATH = next((p for p in candidate_paths if p and os.path.exists(p)), candidate_paths[1])

# Host & Port configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Device configuration
requested_device = os.getenv("DEVICE", "auto").lower()
if requested_device == "cuda" and torch.cuda.is_available():
    device = "cuda"
elif requested_device == "cpu":
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    # 11th Gen Core i5 (4 physical cores) achieves optimal latency with 4 threads
    torch.set_num_threads(min(4, os.cpu_count() or 4))

state = {"model": None, "tokenizer": None, "generator": None, "checkpoint_path": CHECKPOINT_PATH, "step": None, "val_loss": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n========================================================")
    print(f"[SERVER STARTUP] Loading Karna-49M on {device.upper()}...")
    print(f"Target Checkpoint: '{CHECKPOINT_PATH}'")
    print(f"========================================================")

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        config = checkpoint.get("config", Karna49MConfig())

        model = Karna49M(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        tokenizer = Karna49MTokenizer()
        state["model"] = model
        state["tokenizer"] = tokenizer
        state["generator"] = Karna49MGenerator(model, tokenizer, device)

        total_params, _ = model.count_parameters()
        step = checkpoint.get("step", checkpoint.get("epoch", "?"))
        val_loss = checkpoint.get("val_loss", "?")
        state["step"] = step
        val_loss_display = f"{val_loss:.4f}" if isinstance(val_loss, (float, int)) else str(val_loss)
        print(f"[SUCCESS] Loaded Karna-49M ({total_params:,} params) | step/epoch={step} | val_loss={val_loss_display}")
    else:
        print(f"[WARNING] No checkpoint at '{CHECKPOINT_PATH}'. Server running uninitialized.")

    yield


app = FastAPI(title="Karna-49M API Server", version="2.0.0", lifespan=lifespan)

cors_env = os.getenv("CORS_ORIGINS", "*")
origins = [origin.strip() for origin in cors_env.split(",")] if cors_env != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
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
    return "<h1>Karna-49M API Server Running</h1><p>Frontend expected at http://localhost:3000</p>"


@app.get("/api/health")
async def health_check():
    model = state["model"]
    return {
        "status": "online",
        "model": "Karna-49M",
        "device": device,
        "checkpoint_loaded": model is not None,
        "checkpoint_path": state.get("checkpoint_path"),
        "step": state.get("step"),
        "val_loss": state.get("val_loss"),
        "parameters": f"{model.count_parameters()[0]:,}" if model is not None else None,
    }


@app.post("/api/generate/stream")
async def stream_generate(req: GenerateRequest):
    generator = state["generator"]
    if generator is None:
        raise HTTPException(status_code=500, detail="Karna-49M not loaded. Train a checkpoint first (train.py).")

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

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_publisher(), media_type="text/event-stream", headers=headers)


if __name__ == "__main__":
    import uvicorn
    print(f"\n[INFO] Starting Karna-49M FastAPI Server on http://{HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT, reload=False)

