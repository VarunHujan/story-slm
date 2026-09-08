import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from generator import RealtimeGenerator
from model import RealtimeSLM

app = FastAPI(title='Real-Time SLM Server')

model = RealtimeSLM(vocab_size=32000, dim=256, n_layers=6, n_heads=8, n_kv_heads=2)

class DummyTokenizer:
    def encode(self, text): return [101] + [ord(c) for c in text[:30]] + [102]
    def decode(self, ids): return ''.join([chr(i) if i < 256 else '' for i in ids])
    
generator = RealtimeGenerator(model, DummyTokenizer())

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7

@app.get('/', response_class=HTMLResponse)
async def get_ui():
    with open('web_ui.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.post('/api/generate/stream')
async def stream_generate(req: GenerateRequest):
    def event_publisher():
        for token in generator.stream_generate(req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature):
            t_json = json.dumps({'token': token})
            yield f'data: {t_json}\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(event_publisher(), media_type='text/event-stream')

if __name__ == '__main__':
    import uvicorn
    print('Starting Real-Time SLM Server on http://localhost:8000 ...')
    uvicorn.run(app, host='0.0.0.0', port=8000)
