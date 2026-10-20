"use client";

import React, { useState, useEffect } from "react";
import {
  Send,
  MessageSquare,
  FileCode,
  Sliders,
  CheckCircle2,
  AlertCircle,
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  Cpu,
  Zap,
  ArrowRight,
  Database,
  Layers,
  Terminal,
  Activity,
  Code2,
  HelpCircle,
  Radio,
  Sparkles,
  Copy,
  Check,
  Lightbulb,
  GraduationCap
} from "lucide-react";

interface PipelineNode {
  id: string;
  stageNumber: number;
  title: string;
  subtitle: string;
  category: "input" | "tokenization" | "cache" | "backbone" | "logits" | "sampling" | "streaming";
  tensorShape: string;
  analogy: string;
  description: string;
  pythonSnippet: string;
  sampleInput: string;
  sampleOutput: string;
}

const PIPELINE_NODES: PipelineNode[] = [
  {
    id: "user_input",
    stageNumber: 1,
    title: "User Prompt & ChatML Ingestion",
    subtitle: "Wraps raw message into the ChatML instruction format",
    category: "input",
    tensorShape: "string (UTF-8)",
    analogy: "Like placing a handwritten letter into a formal royal envelope with exact sender and recipient seals.",
    description: "Accepts the raw user prompt from the API request and wraps it into the standardized ChatML chat format (<|im_start|>user ... <|im_end|>\\n<|im_start|>assistant\\n). This explicitly signals to the model where user instructions conclude and where the narrative assistant story should begin.",
    pythonSnippet: `prompt_formatted = (
    f"<|im_start|>user\\n{user_prompt}<|im_end|>\\n"
    f"<|im_start|>assistant\\n"
)`,
    sampleInput: `"Write a story about Prince Karna."`,
    sampleOutput: `"<|im_start|>user\\nWrite a story about Prince Karna.<|im_end|>\\n<|im_start|>assistant\\n"`
  },
  {
    id: "tokenization",
    stageNumber: 2,
    title: "TikToken BPE Sub-word Tokenizer",
    subtitle: "Converts raw English string into vocabulary token IDs",
    category: "tokenization",
    tensorShape: "torch.LongTensor [1, S]",
    analogy: "Like translating continuous sentences into numbered Lego bricks using a fixed 50,304-piece catalog.",
    description: "The OpenAI TikToken engine executes Byte-Pair Encoding (BPE). It greedily merges common character sequences into sub-word token IDs from our custom 50,304 vocabulary table, resulting in approximately 1.6 tokens per English word.",
    pythonSnippet: `token_ids = tokenizer.encode(prompt_formatted)
input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)`,
    sampleInput: `"<|im_start|>user\\nWrite a story about Prince Karna.<|im_end|>..."`,
    sampleOutput: `[100258, 882, 198, 14217, 264, 47101, 921, 2873, 58491, 100259, 100258, 78191, 198]`
  },
  {
    id: "kv_cache_guard",
    stageNumber: 3,
    title: "Context Window Guard & KV-Cache Slicing",
    subtitle: "Separates Prefill phase (all tokens) from Decode phase (1 token)",
    category: "cache",
    tensorShape: "K, V: [1, 2048, 4, 56] per layer",
    analogy: "Like bookmarking previously read chapters in a book so you never need to re-read from chapter 1 on every new sentence.",
    description: "Validates that the sequence length S does not exceed max_seq_len (2,048). During prompt prefill, all S tokens are processed together. During subsequent autoregressive decode steps, only 1 new token is passed through the network while previous Key/Value activations are retrieved from RAM in O(1) time.",
    pythonSnippet: `if kv_cache is not None:
    # Retrieve past keys/values in place
    past_len = kv_cache.length
    # Slice only the single new token during autoregressive decoding
    if past_len > 0:
        x = x[:, -1:]`,
    sampleInput: `input_ids [1, 13] | kv_cache.length = 0 (Prefill)`,
    sampleOutput: `Allocated K/V Buffers: 12 layers × [1, 2048, 4, 56]`
  },
  {
    id: "embedding_rope",
    stageNumber: 4,
    title: "Vector Embedding & RoPE Rotation",
    subtitle: "Maps token IDs to vectors & rotates by position angles",
    category: "backbone",
    tensorShape: "x: [1, S, 448], cos/sin: [S, 28]",
    analogy: "Like assigning every word a musical pitch and turning the tuning peg to match its exact position in the song.",
    description: "Token IDs look up 448-dimensional continuous vector representations in the embedding matrix. Simultaneously, Rotary Positional Embedding (RoPE) computes frequency rotation angles [cos(mθ), sin(mθ)] for sequence positions m.",
    pythonSnippet: `x = self.tok_embeddings(input_ids)  # [1, S, 448]
cos, sin = self.rope(seq_len=x.shape[1], offset=past_len)`,
    sampleInput: `input_ids: [100258, 882, ...]`,
    sampleOutput: `Tensor x: shape [1, 13, 448] | Frequencies: cos/sin [13, 28]`
  },
  {
    id: "transformer_backbone",
    stageNumber: 5,
    title: "12-Layer Transformer Blocks (GQA + SwiGLU)",
    subtitle: "Deep contextual reasoning & feature refinement",
    category: "backbone",
    tensorShape: "x: [1, S, 448] through 12 blocks",
    analogy: "Like a council of 12 master story editors, each adding depth, character motives, and plot consistency.",
    description: "The tensor traverses 12 identical transformer blocks. In each block: (1) Pre-RMSNorm scales inputs, (2) Grouped-Query Attention (8 Q-heads / 4 KV-heads) mixes contextual tokens, (3) SwiGLU FFN applies non-linear reasoning, and (4) Residual skip-connections protect gradient flow.",
    pythonSnippet: `for block in self.layers:  # 12 blocks
    # Attention + Residual
    x = x + block.attn(block.attn_norm(x), cos, sin, kv_cache)
    # SwiGLU + Residual
    x = x + block.ffn(block.ffn_norm(x))`,
    sampleInput: `Input Hidden Vector x: [1, 13, 448]`,
    sampleOutput: `Context-Enriched Hidden Vector x: [1, 13, 448]`
  },
  {
    id: "lm_head_logits",
    stageNumber: 6,
    title: "Final RMSNorm & LM Head Logits",
    subtitle: "Projects hidden states back to vocabulary distribution",
    category: "logits",
    tensorShape: "logits: [1, 1, 50304]",
    analogy: "Like calculating the election poll percentages for all 50,304 candidate words in the language.",
    description: "Applies final Root Mean Square Normalization (RMSNorm) to stabilize magnitudes, then projects the last token's 448-dimensional vector through the LM Head weight matrix to produce 50,304 unnormalized log-probability scores (logits).",
    pythonSnippet: `x = self.norm(x)
# Slice the final token position
logits = self.lm_head(x[:, -1, :])  # [1, 50304]`,
    sampleInput: `Final Layer Output x: [1, 1, 448]`,
    sampleOutput: `Logits Tensor: shape [1, 50304] (min: -14.2, max: +11.8)`
  },
  {
    id: "sampling_sse",
    stageNumber: 7,
    title: "Temperature Sampling & SSE Token Stream",
    subtitle: "Picks the next word & streams to web browser",
    category: "sampling",
    tensorShape: "next_token: int → UTF-8 string chunk",
    analogy: "Like the storyteller speaking the next word out loud the millisecond it forms in their mind.",
    description: "Applies temperature scaling (logits / T), filters out tail probability noise with Top-K cutoff, applies softmax, and draws a random categorical sample. The chosen token ID is decoded into a text chunk and flushed as a Server-Sent Event (SSE) to the frontend.",
    pythonSnippet: `# Temperature scaling & Top-K filtering
filtered_logits = top_k_filter(logits / temperature, k=top_k)
probs = F.softmax(filtered_logits, dim=-1)
next_token_id = torch.multinomial(probs, num_samples=1)
token_str = tokenizer.decode([next_token_id.item()])
yield f"data: {json.dumps({'token': token_str})}\\n\\n"`,
    sampleInput: `Logits: [1, 50304] | T=0.75 | Top-K=40`,
    sampleOutput: `Selected: Token #14217 (" Prince") → SSE chunk sent to client`
  }
];

export default function InferencePipelineNodes() {
  const [activeNodeIndex, setActiveNodeIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [copiedCode, setCopiedCode] = useState<boolean>(false);

  const activeNode = PIPELINE_NODES[activeNodeIndex];

  // Auto-play timer for presentation mode
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isPlaying) {
      timer = setInterval(() => {
        setActiveNodeIndex((prev) => (prev + 1) % PIPELINE_NODES.length);
      }, 3500);
    }
    return () => clearInterval(timer);
  }, [isPlaying]);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(activeNode.pythonSnippet);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  return (
    <div className="flex flex-col gap-6 text-slate-800 font-sans pb-12">
      {/* ========================================================= */}
      {/* TOP HEADER: INFERENCE PIPELINE GRAPH                     */}
      {/* ========================================================= */}
      <div className="bg-white border border-stone-200 rounded-2xl p-6 md:p-8 shadow-sm relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-purple-50 text-purple-700 border border-purple-200 tracking-wider uppercase font-mono">
                End-to-End Execution Graph
              </span>
              <span className="text-xs text-slate-400 font-serif italic">
                From Prompt to Next-Token Prediction
              </span>
            </div>
            <h2 className="text-2xl md:text-3xl font-bold font-serif text-slate-900 tracking-tight flex items-center gap-2.5">
              <Activity className="w-7 h-7 text-purple-600" />
              <span>SLM Inference Pipeline & Stage Nodes</span>
            </h2>
            <p className="text-xs md:text-sm text-slate-600 max-w-3xl leading-relaxed">
              Step through the exact lifecycle of how text is ingested, formatted in ChatML, converted into tokens, processed across 12 transformer layers, and streamed to the user via Server-Sent Events.
            </p>
          </div>

          {/* Stepper Controls */}
          <div className="flex items-center gap-2 bg-stone-50 p-2 rounded-xl border border-stone-200 shrink-0">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition cursor-pointer ${
                isPlaying
                  ? "bg-amber-600 text-white shadow-xs"
                  : "bg-purple-600 text-white shadow-xs"
              }`}
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              <span>{isPlaying ? "Pause" : "Auto-Play"}</span>
            </button>

            <button
              onClick={() => {
                setIsPlaying(false);
                setActiveNodeIndex((prev) => (prev + 1) % PIPELINE_NODES.length);
              }}
              className="p-1.5 rounded-lg bg-white hover:bg-stone-100 text-slate-700 border border-stone-200 text-xs transition cursor-pointer"
              title="Next Node Step"
            >
              <SkipForward className="w-4 h-4" />
            </button>

            <button
              onClick={() => {
                setIsPlaying(false);
                setActiveNodeIndex(0);
              }}
              className="p-1.5 rounded-lg bg-white hover:bg-stone-100 text-slate-700 border border-stone-200 text-xs transition cursor-pointer"
              title="Reset to Stage 1"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Horizontal Mini-Stage Progress Bar */}
        <div className="mt-6 pt-5 border-t border-stone-100 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
          {PIPELINE_NODES.map((node, idx) => {
            const isActive = idx === activeNodeIndex;
            const isCompleted = idx < activeNodeIndex;

            return (
              <button
                key={node.id}
                onClick={() => {
                  setIsPlaying(false);
                  setActiveNodeIndex(idx);
                }}
                className={`p-2.5 rounded-xl border text-left transition cursor-pointer ${
                  isActive
                    ? "bg-purple-50 border-purple-300 ring-2 ring-purple-500/20 shadow-xs"
                    : isCompleted
                    ? "bg-emerald-50/50 border-emerald-200 text-emerald-900"
                    : "bg-white border-stone-200 text-slate-600 hover:border-stone-300"
                }`}
              >
                <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                  <span className={isActive ? "text-purple-700 font-bold" : "text-slate-400"}>
                    ST {node.stageNumber}
                  </span>
                  {isCompleted && <CheckCircle2 className="w-3 h-3 text-emerald-600" />}
                </div>
                <p className="text-[11px] font-semibold truncate text-slate-900 font-serif">
                  {node.title.split("&")[0]}
                </p>
                <span className="text-[9px] font-mono text-slate-400 block truncate mt-0.5">
                  {node.tensorShape}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ========================================================= */}
      {/* ACTIVE STAGE DEEP-DIVE INSPECTOR                          */}
      {/* ========================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Detailed Explanation & Math */}
        <div className="lg:col-span-2 space-y-6">
          {/* Stage Hero Card */}
          <div className="p-6 md:p-8 rounded-2xl bg-white border border-stone-200 shadow-sm space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-100 pb-4">
              <div>
                <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono uppercase font-bold bg-purple-100 text-purple-800">
                  Stage {activeNode.stageNumber} of {PIPELINE_NODES.length}
                </span>
                <h3 className="text-xl md:text-2xl font-bold font-serif text-slate-900 mt-1">
                  {activeNode.title}
                </h3>
                <p className="text-xs text-slate-500">{activeNode.subtitle}</p>
              </div>

              <div className="px-3 py-1.5 rounded-xl bg-stone-50 border border-stone-200 font-mono text-xs text-purple-700 font-semibold">
                Tensor: {activeNode.tensorShape}
              </div>
            </div>

            {/* Beginner Friendly Real-World Analogy */}
            <div className="p-4 rounded-xl bg-amber-50/80 border border-amber-200/80 space-y-1.5">
              <div className="flex items-center gap-2 text-amber-900 font-bold text-xs font-serif">
                <Lightbulb className="w-4 h-4 text-amber-600" />
                <span>Teacher's Real-World Analogy:</span>
              </div>
              <p className="text-xs text-slate-700 leading-relaxed italic">
                "{activeNode.analogy}"
              </p>
            </div>

            {/* Technical Explanation */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono">
                Technical Execution Details:
              </h4>
              <p className="text-xs md:text-sm text-slate-600 leading-relaxed">
                {activeNode.description}
              </p>
            </div>

            {/* Live Input & Output Tensors Sample */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 text-xs">
              <div className="p-3.5 rounded-xl bg-stone-50 border border-stone-200 space-y-1">
                <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold block">
                  Stage Input
                </span>
                <p className="font-mono text-slate-800 break-all text-[11px]">
                  {activeNode.sampleInput}
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-purple-50/50 border border-purple-200 space-y-1">
                <span className="text-[10px] font-mono uppercase text-purple-700 font-semibold block">
                  Stage Output
                </span>
                <p className="font-mono text-purple-900 break-all text-[11px] font-medium">
                  {activeNode.sampleOutput}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Col: Real Python Code Viewer */}
        <div className="space-y-6">
          <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-stone-100 pb-3">
              <div className="flex items-center gap-2">
                <Code2 className="w-4 h-4 text-purple-600" />
                <h4 className="font-bold text-xs font-mono uppercase tracking-wider text-slate-700">
                  Python Backend Snippet
                </h4>
              </div>

              <button
                onClick={handleCopyCode}
                className="p-1.5 rounded-md hover:bg-stone-100 text-slate-500 hover:text-slate-800 transition cursor-pointer text-xs flex items-center gap-1"
                title="Copy Python Code"
              >
                {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                <span className="text-[11px] font-mono">{copiedCode ? "Copied" : "Copy"}</span>
              </button>
            </div>

            {/* Code Box */}
            <pre className="p-4 rounded-xl bg-stone-900 text-stone-100 font-mono text-[11px] leading-relaxed overflow-x-auto shadow-inner">
              <code>{activeNode.pythonSnippet}</code>
            </pre>

            <div className="p-3 rounded-xl bg-stone-50 border border-stone-200 text-xs text-slate-500 space-y-1">
              <span className="font-bold text-slate-700 block">Production Location:</span>
              <p className="font-mono text-[10px] text-slate-600 break-all">
                slm_realtime_project/backend/model2/server.py
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
