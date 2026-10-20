"use client";

import React, { useState, useMemo } from "react";
import TransformerVisualFlow from "./TransformerVisualFlow";
import {
  Layers,
  Sparkles,
  Sliders,
  Eye,
  Info,
  Zap,
  Cpu,
  ArrowRight,
  HelpCircle,
  Activity,
  CheckCircle2,
  RefreshCw,
  GitBranch,
  Split,
  Binary,
  Maximize2,
  Compass,
  BookOpen,
  Scale,
  Hash,
  Lightbulb,
  GraduationCap
} from "lucide-react";

interface TokenInfo {
  id: number;
  text: string;
  pos: number;
}

// Preset prompts for exploration
const PRESET_PROMPTS = [
  "Prince Karna held his celestial bow Vijaya with focus",
  "The brave little wanderer entered the enchanted vault",
  "Once upon a time in a kingdom of golden sun",
  "A glowing silver star fell into the quiet lake"
];

// Tokenizer simulation for interactive exploration
function tokenizeText(prompt: string): TokenInfo[] {
  const words = prompt.trim().split(/\s+/);
  return words.map((w, idx) => {
    let hash = 0;
    for (let i = 0; i < w.length; i++) {
      hash = (hash << 5) - hash + w.charCodeAt(i);
      hash |= 0;
    }
    const id = Math.abs(hash % 50000) + 300;
    return {
      id,
      text: idx === 0 ? w : ` ${w}`,
      pos: idx
    };
  });
}

export default function TransformerArchitecture() {
  const [prompt, setPrompt] = useState<string>("Prince Karna held his celestial bow Vijaya with focus");
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [selectedHead, setSelectedHead] = useState<number>(0);
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);
  const [selectedViewTab, setSelectedViewTab] = useState<"flow" | "overview" | "attention" | "swiglu" | "rope" | "lm_head">("flow");

  // Interactive sampling states for Output Head
  const [temperature, setTemperature] = useState<number>(0.75);
  const [topK, setTopK] = useState<number>(8);

  const tokens = useMemo(() => tokenizeText(prompt), [prompt]);
  const nTokens = tokens.length;

  // Generate simulated causal attention weights for each head
  const attentionWeights = useMemo(() => {
    const matrix: number[][] = [];
    for (let i = 0; i < nTokens; i++) {
      const row: number[] = [];
      let sum = 0;
      for (let j = 0; j < nTokens; j++) {
        if (j > i) {
          // Causal mask: cannot attend to future tokens
          row.push(0);
        } else {
          let baseScore = 1.0;
          if (selectedHead === 0) {
            baseScore = j === i ? 0.35 : j === i - 1 ? 0.45 : 0.2 / (i - j + 1);
          } else if (selectedHead === 1) {
            baseScore = j === 0 ? 0.55 : 0.45 / (i + 1);
          } else if (selectedHead % 2 === 0) {
            baseScore = 1.0 / (Math.abs(i - j) + 1.2);
          } else {
            baseScore = j === i ? 0.6 : 0.4 / (i - j + 1);
          }
          row.push(baseScore);
          sum += baseScore;
        }
      }
      // Normalize row with softmax
      const normalizedRow = row.map((val) => (sum > 0 ? val / sum : 0));
      matrix.push(normalizedRow);
    }
    return matrix;
  }, [nTokens, selectedHead]);

  // Simulated vocabulary next-token candidate pool
  const candidateTokens = useMemo(() => {
    const baseCandidates = [
      { token: "and", logit: 8.4, meaning: "Coordinate conjunction continuation" },
      { token: "aimed", logit: 7.9, meaning: "Direct action verb for archery" },
      { token: "toward", logit: 6.8, meaning: "Spatial preposition" },
      { token: "with", logit: 6.1, meaning: "Adverbial modifier" },
      { token: "unmatched", logit: 5.7, meaning: "Descriptive adjective" },
      { token: "as", logit: 5.2, meaning: "Temporal connector" },
      { token: "releasing", logit: 4.8, meaning: "Dynamic participle action" },
      { token: "his", logit: 4.3, meaning: "Possessive pronoun" },
      { token: "the", logit: 3.9, meaning: "Definite article" },
      { token: "divine", logit: 3.4, meaning: "Mythic epithet" }
    ];

    // Apply temperature scaling: logits / T
    const scaled = baseCandidates.map((c) => ({
      ...c,
      scaledLogit: c.logit / Math.max(0.05, temperature)
    }));

    // Softmax over top-K
    const topKCands = scaled.slice(0, topK);
    const maxLogit = Math.max(...topKCands.map((c) => c.scaledLogit));
    const exps = topKCands.map((c) => Math.exp(c.scaledLogit - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);

    return topKCands.map((c, i) => ({
      ...c,
      probability: (exps[i] / sumExps) * 100
    }));
  }, [temperature, topK]);

  return (
    <div className="flex flex-col gap-6 text-slate-800 font-sans pb-12">
      {/* ========================================================= */}
      {/* MASTERCLASS HEADER & MODEL ARCHITECTURE AT A GLANCE       */}
      {/* ========================================================= */}
      <div className="bg-white border border-stone-200 rounded-2xl p-6 md:p-8 shadow-sm relative overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div className="space-y-2 max-w-2xl">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-blue-50 text-blue-700 border border-blue-200 tracking-wider uppercase font-mono">
                Interactive Neural Masterclass
              </span>
              <span className="text-xs text-slate-400 font-serif italic">
                Polo Club Explainer Architecture
              </span>
            </div>
            <h2 className="text-2xl md:text-3xl font-bold font-serif text-slate-900 tracking-tight flex items-center gap-2.5">
              <Layers className="w-7 h-7 text-blue-600" />
              <span>Karna-49M Transformer Architecture</span>
            </h2>
            <p className="text-xs md:text-sm text-slate-600 leading-relaxed">
              Explore how our <strong>49.38 Million parameter</strong> Small Language Model operates from token embeddings to causal self-attention, rotary position embedding (RoPE), SwiGLU gating, and next-token generation.
            </p>
          </div>

          {/* Model Specification Card Matrix */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 bg-stone-50 border border-stone-200 p-4 rounded-xl shrink-0 text-xs">
            <div className="p-2 bg-white rounded-lg border border-stone-200/80 shadow-2xs">
              <span className="text-slate-400 block text-[10px] uppercase font-mono font-semibold">Parameters</span>
              <span className="font-mono font-bold text-blue-700 text-sm">49.38M</span>
            </div>
            <div className="p-2 bg-white rounded-lg border border-stone-200/80 shadow-2xs">
              <span className="text-slate-400 block text-[10px] uppercase font-mono font-semibold">Hidden Dim (d)</span>
              <span className="font-mono font-bold text-purple-700 text-sm">448</span>
            </div>
            <div className="p-2 bg-white rounded-lg border border-stone-200/80 shadow-2xs">
              <span className="text-slate-400 block text-[10px] uppercase font-mono font-semibold">Layers</span>
              <span className="font-mono font-bold text-emerald-700 text-sm">12 Blocks</span>
            </div>
            <div className="p-2 bg-white rounded-lg border border-stone-200/80 shadow-2xs">
              <span className="text-slate-400 block text-[10px] uppercase font-mono font-semibold">GQA Ratio</span>
              <span className="font-mono font-bold text-amber-700 text-sm">8Q : 4KV</span>
            </div>
          </div>
        </div>

        {/* Live Tokenizer Input Strip (shown on deep-dive tabs) */}
        {selectedViewTab !== "flow" && (
          <div className="mt-6 pt-5 border-t border-stone-100 flex flex-col gap-3 animate-in fade-in">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex items-center gap-1.5 font-mono">
                <Sparkles className="w-3.5 h-3.5 text-blue-600" />
                <span>Interactive Sequence Input:</span>
              </label>
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
                <span className="text-[11px] text-slate-400 whitespace-nowrap">Try Prompts:</span>
                {PRESET_PROMPTS.map((p, idx) => (
                  <button
                    key={idx}
                    onClick={() => setPrompt(p)}
                    className={`text-[11px] px-2.5 py-1 rounded-lg border transition whitespace-nowrap cursor-pointer ${
                      prompt === p
                        ? "bg-blue-50 border-blue-300 text-blue-700 font-semibold"
                        : "bg-white border-stone-200 text-slate-600 hover:border-slate-300 hover:bg-stone-50"
                    }`}
                  >
                    {p.slice(0, 24)}...
                  </button>
                ))}
              </div>
            </div>

            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Type any sentence to decompose into sub-word tokens..."
              className="w-full bg-white border border-stone-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-600 shadow-xs transition"
            />

            {/* Token Chips Representation */}
            <div className="flex flex-wrap items-center gap-2 p-3.5 bg-stone-50 border border-stone-200/80 rounded-xl">
              <span className="text-xs font-mono font-semibold text-slate-500 mr-2 uppercase text-[10px]">
                Decomposed Tokens ({tokens.length}):
              </span>
              {tokens.map((tok, idx) => (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredTokenIdx(idx)}
                  onMouseLeave={() => setHoveredTokenIdx(null)}
                  className={`px-3 py-1.5 rounded-lg border text-xs font-mono transition-all cursor-pointer flex items-center gap-2 ${
                    hoveredTokenIdx === idx
                      ? "bg-blue-600 text-white border-blue-600 shadow-sm scale-105"
                      : "bg-white border-stone-200 text-slate-700 hover:border-blue-300"
                  }`}
                >
                  <span className={hoveredTokenIdx === idx ? "text-blue-200" : "text-slate-400"}>
                    p{tok.pos}
                  </span>
                  <span className="font-semibold">"{tok.text}"</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                      hoveredTokenIdx === idx
                        ? "bg-blue-700 text-blue-100"
                        : "bg-stone-100 text-blue-700"
                    }`}
                  >
                    #{tok.id}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* NAVIGATION TABS FOR ARCHITECTURAL DEEP-DIVES              */}
      {/* ========================================================= */}
      <div className="flex items-center gap-2 border-b border-stone-200 pb-2 overflow-x-auto text-xs font-medium">
        <button
          onClick={() => setSelectedViewTab("flow")}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition whitespace-nowrap cursor-pointer ${
            selectedViewTab === "flow"
              ? "bg-violet-700 text-white font-bold shadow-xs ring-1 ring-violet-500"
              : "bg-white border border-stone-200 text-slate-700 hover:bg-stone-50"
          }`}
        >
          <Activity className="w-3.5 h-3.5" />
          <span>🌟 Visual Pipeline Flow (Transformer Explainer)</span>
        </button>

        <button
          onClick={() => setSelectedViewTab("overview")}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition whitespace-nowrap cursor-pointer ${
            selectedViewTab === "overview"
              ? "bg-blue-600 text-white font-bold shadow-xs"
              : "bg-white border border-stone-200 text-slate-700 hover:bg-stone-50"
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>1. Full Transformer Blueprint</span>
        </button>

        <button
          onClick={() => setSelectedViewTab("attention")}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition whitespace-nowrap cursor-pointer ${
            selectedViewTab === "attention"
              ? "bg-blue-600 text-white font-bold shadow-xs"
              : "bg-white border border-stone-200 text-slate-700 hover:bg-stone-50"
          }`}
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>2. Grouped-Query Attention (GQA)</span>
        </button>

        <button
          onClick={() => setSelectedViewTab("rope")}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition whitespace-nowrap cursor-pointer ${
            selectedViewTab === "rope"
              ? "bg-blue-600 text-white font-bold shadow-xs"
              : "bg-white border border-stone-200 text-slate-700 hover:bg-stone-50"
          }`}
        >
          <Compass className="w-3.5 h-3.5" />
          <span>3. Rotary Positional Embeddings (RoPE)</span>
        </button>

        <button
          onClick={() => setSelectedViewTab("swiglu")}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition whitespace-nowrap cursor-pointer ${
            selectedViewTab === "swiglu"
              ? "bg-blue-600 text-white font-bold shadow-xs"
              : "bg-white border border-stone-200 text-slate-700 hover:bg-stone-50"
          }`}
        >
          <Split className="w-3.5 h-3.5" />
          <span>4. SwiGLU Gated Feed-Forward</span>
        </button>

        <button
          onClick={() => setSelectedViewTab("lm_head")}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition whitespace-nowrap cursor-pointer ${
            selectedViewTab === "lm_head"
              ? "bg-blue-600 text-white font-bold shadow-xs"
              : "bg-white border border-stone-200 text-slate-700 hover:bg-stone-50"
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>5. LM Head & Sampling Lab</span>
        </button>
      </div>

      {/* ========================================================= */}
      {/* TAB 0: INTERACTIVE VISUAL PIPELINE FLOW                   */}
      {/* ========================================================= */}
      {selectedViewTab === "flow" && (
        <div className="animate-in fade-in">
          <TransformerVisualFlow onNavigateTab={(tab) => setSelectedViewTab(tab)} />
        </div>
      )}

      {/* ========================================================= */}
      {/* TAB 1: FULL TRANSFORMER BLUEPRINT                         */}
      {/* ========================================================= */}
      {selectedViewTab === "overview" && (
        <div className="space-y-6 animate-in fade-in">
          {/* Teacher Persona Analogy Banner */}
          <div className="p-5 rounded-2xl bg-amber-50/80 border border-amber-200/80 space-y-2">
            <div className="flex items-center gap-2 text-amber-900 font-bold text-sm font-serif">
              <Lightbulb className="w-4 h-4 text-amber-600" />
              <span>Teacher's Intuition: How Does a Transformer Actually "Think"?</span>
            </div>
            <p className="text-xs md:text-sm text-slate-700 leading-relaxed">
              Before Transformers, language models were like people reading a book through a keyhole (RNNs): they looked at one word at a time and frequently forgot what happened three paragraphs ago because of vanishing gradients.
              The <strong>Transformer</strong> changed everything by looking at all words simultaneously. It allows every single word to ask every other word: <em>"How relevant are you to my meaning right now?"</em>
            </p>
          </div>

          {/* End-to-End Visual Stage Flow */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {/* Step 1 */}
            <div className="p-4 rounded-xl bg-white border border-stone-200 shadow-2xs space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-blue-700 font-semibold">
                <span>STAGE 1</span>
                <span>Dim: [1, S]</span>
              </div>
              <h4 className="font-bold text-slate-900 text-sm font-serif">Token Embedding</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Tokens are mapped from discrete integers (#47101) into dense 448-dimensional semantic space.
              </p>
              <div className="pt-2 text-[10px] font-mono text-slate-400">Embedding: 50,304 × 448</div>
            </div>

            {/* Step 2 */}
            <div className="p-4 rounded-xl bg-white border border-stone-200 shadow-2xs space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-purple-700 font-semibold">
                <span>STAGE 2</span>
                <span>Angle: mθ</span>
              </div>
              <h4 className="font-bold text-slate-900 text-sm font-serif">RoPE Rotation</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Rather than adding static position tags, vector pairs rotate in 2D frequency dials to naturally encode distance.
              </p>
              <div className="pt-2 text-[10px] font-mono text-slate-400">Head Dim: 56 per Head</div>
            </div>

            {/* Step 3 */}
            <div className="p-4 rounded-xl bg-white border border-stone-200 shadow-2xs space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-amber-700 font-semibold">
                <span>STAGE 3</span>
                <span>12 Layers</span>
              </div>
              <h4 className="font-bold text-slate-900 text-sm font-serif">GQA Attention</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                8 Query heads attend over 4 shared Key/Value heads, slashing KV cache memory by 50% without quality loss.
              </p>
              <div className="pt-2 text-[10px] font-mono text-slate-400">Causal Masking: Lower Triangular</div>
            </div>

            {/* Step 4 */}
            <div className="p-4 rounded-xl bg-white border border-stone-200 shadow-2xs space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-emerald-700 font-semibold">
                <span>STAGE 4</span>
                <span>d_ffn = 1216</span>
              </div>
              <h4 className="font-bold text-slate-900 text-sm font-serif">SwiGLU FFN</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Non-linear knowledge memory. Uses continuous gating multiplication (x · W_gate · SiLU) ⊗ (x · W_up).
              </p>
              <div className="pt-2 text-[10px] font-mono text-slate-400">Replaces legacy ReLU/GELU</div>
            </div>

            {/* Step 5 */}
            <div className="p-4 rounded-xl bg-white border border-stone-200 shadow-2xs space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-rose-700 font-semibold">
                <span>STAGE 5</span>
                <span>Vocab: 50,304</span>
              </div>
              <h4 className="font-bold text-slate-900 text-sm font-serif">LM Head Projection</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Final RMSNorm projects hidden vectors back to 50,304 vocabulary logits for temperature & top-K sampling.
              </p>
              <div className="pt-2 text-[10px] font-mono text-slate-400">Next Token Probability Distribution</div>
            </div>
          </div>

          {/* Historical Evolution & Why Karna-49M Architecture Was Chosen */}
          <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-4">
            <h3 className="font-bold text-base font-serif text-slate-900 flex items-center gap-2">
              <GraduationCap className="w-5 h-5 text-blue-600" />
              <span>Architectural Evolution: CNNs vs RNNs vs Transformers</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="p-4 rounded-xl bg-stone-50 border border-stone-200 space-y-2">
                <span className="font-bold text-slate-900 block font-serif">1. Convolutional Networks (CNNs)</span>
                <p className="text-slate-600 leading-relaxed">
                  <strong>Strength:</strong> Local spatial feature detection with translation invariance.<br />
                  <strong>Fatal Flaw for NLP:</strong> Extremely small receptive field. To connect word 1 to word 100, you need 50 convolutional layers stacked together.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-stone-50 border border-stone-200 space-y-2">
                <span className="font-bold text-slate-900 block font-serif">2. Recurrent Networks (LSTMs / GRUs)</span>
                <p className="text-slate-600 leading-relaxed">
                  <strong>Strength:</strong> Handles sequential time steps with internal hidden memory state $h_t$.<br />
                  <strong>Fatal Flaw for NLP:</strong> Strictly sequential ($O(N)$ runtime). Cannot be parallelized across GPUs; catastrophic forgetting across long texts.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-blue-50/70 border border-blue-200 space-y-2">
                <span className="font-bold text-blue-900 block font-serif">3. Modern Decoder-Only Transformers (Karna-49M)</span>
                <p className="text-slate-700 leading-relaxed">
                  <strong>Strength:</strong> Self-attention connects any two tokens in $O(1)$ operations regardless of distance. Highly parallel GPU tensor contraction.<br />
                  <strong>Karna Advantage:</strong> GQA + RoPE + SwiGLU modern stack delivers 70B-grade engineering at 49M efficiency.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* TAB 2: GROUPED-QUERY ATTENTION (GQA)                      */}
      {/* ========================================================= */}
      {selectedViewTab === "attention" && (
        <div className="space-y-6 animate-in fade-in">
          {/* GQA Concept Explainer */}
          <div className="p-5 rounded-2xl bg-amber-50/80 border border-amber-200/80 space-y-2">
            <div className="flex items-center gap-2 text-amber-900 font-bold text-sm font-serif">
              <Lightbulb className="w-4 h-4 text-amber-600" />
              <span>Beginner Analogy: What is Grouped-Query Attention (GQA)?</span>
            </div>
            <p className="text-xs md:text-sm text-slate-700 leading-relaxed">
              Imagine an elite research library. In traditional <strong>Multi-Head Attention (MHA)</strong>, 8 researchers each need their own dedicated assistant holding copies of the books. This eats huge amounts of memory (KV Cache).
              In <strong>Grouped-Query Attention (GQA)</strong>, 8 researchers share <strong>4 expert assistants</strong> (a 2:1 ratio). Each assistant serves 2 researchers. The memory consumption is cut in half, while the answers remain just as accurate!
            </p>
          </div>

          {/* Interactive Attention Matrix Heatmap */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-100 pb-3">
                <div>
                  <h4 className="font-bold text-sm text-slate-900 font-serif">
                    Live Causal Attention Matrix (Layer {selectedLayer + 1}, Head {selectedHead + 1})
                  </h4>
                  <p className="text-xs text-slate-500">
                    Row token attends to column tokens. The upper triangle is zeroed out by the causal mask.
                  </p>
                </div>

                {/* Head Selector */}
                <div className="flex items-center gap-1 bg-stone-100 p-1 rounded-lg">
                  {[0, 1, 2, 3, 4, 5, 6, 7].map((h) => (
                    <button
                      key={h}
                      onClick={() => setSelectedHead(h)}
                      className={`px-2 py-0.5 rounded text-xs font-mono cursor-pointer transition ${
                        selectedHead === h
                          ? "bg-blue-600 text-white font-bold"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      H{h + 1}
                    </button>
                  ))}
                </div>
              </div>

              {/* Heatmap Grid */}
              <div className="overflow-x-auto pb-2">
                <table className="min-w-full border-collapse text-center text-xs font-mono">
                  <thead>
                    <tr>
                      <th className="p-2 text-slate-400 font-normal">Token</th>
                      {tokens.map((tok, j) => (
                        <th key={j} className="p-2 text-slate-600 font-semibold truncate max-w-[70px]">
                          {tok.text}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tokens.map((rowTok, i) => (
                      <tr key={i}>
                        <td className="p-2 text-slate-600 font-semibold text-right pr-3 whitespace-nowrap">
                          {rowTok.text}
                        </td>
                        {tokens.map((colTok, j) => {
                          const weight = attentionWeights[i]?.[j] || 0;
                          const isMasked = j > i;
                          const bgIntensity = isMasked
                            ? "bg-stone-100 text-stone-300"
                            : weight > 0.4
                            ? "bg-blue-600 text-white font-bold"
                            : weight > 0.2
                            ? "bg-blue-200 text-blue-900 font-medium"
                            : weight > 0.05
                            ? "bg-blue-50 text-blue-800"
                            : "bg-white text-slate-400";

                          return (
                            <td
                              key={j}
                              className={`p-2 border border-stone-200/60 rounded transition-colors ${bgIntensity}`}
                              title={
                                isMasked
                                  ? "Causally Masked: cannot look into future tokens"
                                  : `Attention: ${(weight * 100).toFixed(1)}%`
                              }
                            >
                              {isMasked ? "—" : (weight * 100).toFixed(0) + "%"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Architectural Explanation Sidebar */}
            <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-4">
              <h4 className="font-bold text-sm text-slate-900 font-serif">Attention Mathematics</h4>
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200 text-xs font-mono text-slate-800 space-y-1">
                <span className="text-[10px] uppercase text-slate-400 block font-semibold">Scaled Dot-Product Formula</span>
                <p className="font-bold text-blue-800">Attention(Q, K, V) = softmax((Q K^T) / √d_k + M) V</p>
              </div>

              <div className="text-xs text-slate-600 space-y-2.5 leading-relaxed">
                <p>
                  <strong>1. Q (Query):</strong> What the current token is seeking context for.
                </p>
                <p>
                  <strong>2. K (Key):</strong> What other tokens offer as identifying tags.
                </p>
                <p>
                  <strong>3. V (Value):</strong> The actual factual information payload carried forward.
                </p>
                <p>
                  <strong>4. Scaling Factor (√d_k = √56 ≈ 7.48):</strong> Prevents dot-products from exploding into regions where softmax derivatives vanish.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-900 space-y-1">
                <span className="font-bold block">KV-Cache Footprint in Karna-49M:</span>
                <p className="text-[11px] leading-relaxed">
                  Standard MHA requires <strong>1.5 MB per sequence</strong>. Karna's GQA uses only <strong>768 KB per sequence</strong>, doubling batch throughput on low-resource CPU servers.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* TAB 3: ROTARY POSITION EMBEDDING (RoPE)                   */}
      {/* ========================================================= */}
      {selectedViewTab === "rope" && (
        <div className="space-y-6 animate-in fade-in">
          <div className="p-5 rounded-2xl bg-amber-50/80 border border-amber-200/80 space-y-2">
            <div className="flex items-center gap-2 text-amber-900 font-bold text-sm font-serif">
              <Lightbulb className="w-4 h-4 text-amber-600" />
              <span>Beginner Analogy: How Does RoPE Know Word Order?</span>
            </div>
            <p className="text-xs md:text-sm text-slate-700 leading-relaxed">
              Older models added static numbers to words (e.g. Word 1 gets +1, Word 2 gets +2). This broke whenever texts exceeded the training length.
              <strong>Rotary Position Embedding (RoPE)</strong> treats token pairs like hands on a clock. Every time you move to the next position, the clock hand rotates by an angle $\theta$. The angle difference between two words naturally tells the model their exact distance, whether they are 5 words apart or 5,000 words apart!
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Visual RoPE Rotation Dial */}
            <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-4">
              <h4 className="font-bold text-sm text-slate-900 font-serif">
                2D Rotation Plane Representation
              </h4>
              <p className="text-xs text-slate-500">
                Token vector slices (x₂ᵢ, x₂ᵢ₊₁) rotate by position angle m · θᵢ.
              </p>

              <div className="h-64 flex items-center justify-center p-4 bg-stone-50 rounded-xl border border-stone-200 relative">
                {/* Coordinate Grid */}
                <div className="absolute inset-4 border border-dashed border-stone-200 rounded-full" />
                <div className="absolute inset-12 border border-dashed border-stone-200 rounded-full" />
                <div className="w-full h-0.5 bg-stone-200 absolute" />
                <div className="h-full w-0.5 bg-stone-200 absolute" />

                {/* Rotating Vectors for first 5 tokens */}
                {tokens.slice(0, 5).map((tok, idx) => {
                  const angleDeg = (idx * 45) % 360;
                  const rad = (angleDeg * Math.PI) / 180;
                  const radius = 80;
                  const x = Math.cos(rad) * radius;
                  const y = -Math.sin(rad) * radius;

                  return (
                    <div
                      key={idx}
                      className="absolute transition-all duration-300 flex items-center gap-1 group cursor-pointer"
                      style={{ transform: `translate(${x}px, ${y}px)` }}
                    >
                      <div className="w-3 h-3 rounded-full bg-blue-600 border-2 border-white shadow-xs group-hover:scale-125 transition" />
                      <span className="text-[10px] font-mono font-bold bg-white px-1.5 py-0.5 rounded border border-stone-200 shadow-2xs text-slate-800 whitespace-nowrap">
                        p{tok.pos}: {tok.text.trim()}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                <span>Base Frequency: $\theta = 10,000$</span>
                <span>Context Window: 2,048 (Extensible to 16k)</span>
              </div>
            </div>

            {/* Technical RoPE Formula Card */}
            <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-4">
              <h4 className="font-bold text-sm text-slate-900 font-serif">Mathematical Definition</h4>
              <div className="p-4 bg-stone-50 rounded-xl border border-stone-200 text-xs font-mono text-slate-800 space-y-2">
                <span className="text-[10px] uppercase text-slate-400 block font-semibold">RoPE Rotation Matrix</span>
                <p className="font-bold text-purple-900">
                  R_θ,m = diag(R_θ1,m, R_θ2,m, ..., R_θd/2,m)
                </p>
                <p className="text-[11px] text-slate-600">
                  Where each 2x2 sub-matrix is: [cos(mθ) -sin(mθ); sin(mθ) cos(mθ)]
                </p>
              </div>

              <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
                <p>
                  <strong>Why does this beat Absolute Positional Encoding?</strong><br />
                  Absolute embeddings treat position 5 and position 10 as completely independent learned numbers. RoPE ensures that the inner product between position $m$ and position $n$ depends purely on the relative distance $(m - n)$:
                </p>
                <div className="p-2.5 rounded bg-blue-50 text-blue-900 font-mono text-[11px]">
                  {"⟨R_m q, R_n k⟩ = q^T R_(n - m) k"}
                </div>
                <p>
                  This property is what allows Karna-49M to preserve chapter-to-chapter storyline continuity across thousands of tokens!
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* TAB 4: SWIGLU GATED FEED-FORWARD NETWORK                   */}
      {/* ========================================================= */}
      {selectedViewTab === "swiglu" && (
        <div className="space-y-6 animate-in fade-in">
          <div className="p-5 rounded-2xl bg-amber-50/80 border border-amber-200/80 space-y-2">
            <div className="flex items-center gap-2 text-amber-900 font-bold text-sm font-serif">
              <Lightbulb className="w-4 h-4 text-amber-600" />
              <span>Beginner Analogy: What is SwiGLU Gating?</span>
            </div>
            <p className="text-xs md:text-sm text-slate-700 leading-relaxed">
              Standard neural network layers use simple ON/OFF switches like ReLU (if input is negative, turn it off).
              <strong>SwiGLU (Swish Gated Linear Unit)</strong> works like an intelligent dimmer valve. One branch calculates the raw information, while a second parallel branch calculates a smooth gating coefficient from 0.0 to 1.0. When multiplied together, subtle narrative nuances pass through while irrelevant noise is smoothly extinguished.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Visual Gating Block */}
            <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-4">
              <h4 className="font-bold text-sm text-slate-900 font-serif">
                SwiGLU Two-Branch Gating Flow
              </h4>

              <div className="p-4 bg-stone-50 rounded-xl border border-stone-200 space-y-3 text-xs">
                {/* Branch 1 */}
                <div className="p-3 bg-white rounded-lg border border-purple-200 flex items-center justify-between">
                  <div>
                    <span className="font-bold text-purple-900 block font-serif">Gate Branch</span>
                    <span className="text-[11px] text-slate-500 font-mono">x @ W_gate [448 → 1,216] → SiLU(·)</span>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-purple-100 text-purple-800 text-[10px] font-mono">Continuous Valve</span>
                </div>

                <div className="text-center font-bold text-slate-400 font-mono text-sm">
                  ⊗ Element-Wise Hadamard Multiplication
                </div>

                {/* Branch 2 */}
                <div className="p-3 bg-white rounded-lg border border-blue-200 flex items-center justify-between">
                  <div>
                    <span className="font-bold text-blue-900 block font-serif">Up-Projection Branch</span>
                    <span className="text-[11px] text-slate-500 font-mono">x @ W_up [448 → 1,216]</span>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-blue-100 text-blue-800 text-[10px] font-mono">Raw Signal</span>
                </div>

                <div className="text-center font-bold text-slate-400 font-mono text-sm">
                  ↓ Down Projection
                </div>

                {/* Output */}
                <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200 flex items-center justify-between">
                  <div>
                    <span className="font-bold text-emerald-900 block font-serif">Output Projection</span>
                    <span className="text-[11px] text-slate-500 font-mono">(Gated Result) @ W_down [1,216 → 448]</span>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 text-[10px] font-mono">Hidden State x</span>
                </div>
              </div>
            </div>

            {/* Why SwiGLU Beats Legacy FFNs */}
            <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-4">
              <h4 className="font-bold text-sm text-slate-900 font-serif">Why SwiGLU is Superior</h4>

              <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
                <p>
                  Research from Noam Shazeer (2020) and modern models (LLaMA-3, Mistral, Gemma) confirmed that GLU variants consistently outperform standard ReLU or GELU MLPs in language modeling perplexity.
                </p>
                <div className="p-3 bg-stone-50 rounded-xl border border-stone-200 font-mono text-[11px] text-slate-800 space-y-1">
                  <span className="text-[10px] text-slate-400 uppercase font-semibold">SwiGLU Dimension Allocation:</span>
                  <p className="font-bold text-amber-800">d_ffn = round_up((8/3) * d) = 1,216</p>
                </div>
                <p>
                  By setting the intermediate dimension to (8/3)d instead of the traditional 4d, SwiGLU maintains the exact same parameter budget while achieving higher expressive capacity and smoother gradient flow during backpropagation.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* TAB 5: LM HEAD & SAMPLING LAB                             */}
      {/* ========================================================= */}
      {selectedViewTab === "lm_head" && (
        <div className="space-y-6 animate-in fade-in">
          <div className="p-5 rounded-2xl bg-amber-50/80 border border-amber-200/80 space-y-2">
            <div className="flex items-center gap-2 text-amber-900 font-bold text-sm font-serif">
              <Lightbulb className="w-4 h-4 text-amber-600" />
              <span>Beginner Analogy: What is Temperature & Top-K Sampling?</span>
            </div>
            <p className="text-xs md:text-sm text-slate-700 leading-relaxed">
              When the model finishes reading a prompt, it doesn't output a single word. It outputs 50,304 probability numbers for every known word in the English language!
              <strong>Temperature</strong> controls the risk tolerance: at T=0.1, it always picks the safest, most obvious word. At T=1.2, it takes bold, creative risks.
              <strong>Top-K</strong> acts as a bouncer, throwing out the 50,000 least likely words so only the top K contenders are considered.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Interactive Sampling Lab Controls */}
            <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-5">
              <h4 className="font-bold text-sm text-slate-900 font-serif">
                Interactive Sampler Controls
              </h4>

              {/* Temperature Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-700">Temperature (T)</span>
                  <span className="font-mono font-bold text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                    {temperature}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-stone-200 rounded accent-amber-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>Deterministic (0.1)</span>
                  <span>Creative (1.5)</span>
                </div>
              </div>

              {/* Top-K Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-700">Top-K Candidate Cutoff</span>
                  <span className="font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                    {topK}
                  </span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="10"
                  step="1"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-stone-200 rounded accent-blue-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>Top 2</span>
                  <span>Top 10</span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-stone-50 border border-stone-200 text-xs text-slate-600 space-y-1">
                <span className="font-bold text-slate-800 block">Active Formula:</span>
                <p className="font-mono text-[11px] text-blue-800">
                  P(w_i) = exp(z_i / T) / Σ exp(z_j / T)
                </p>
              </div>
            </div>

            {/* Next-Token Probability Bar Chart */}
            <div className="lg:col-span-2 p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-4">
              <div className="flex items-center justify-between border-b border-stone-100 pb-3">
                <h4 className="font-bold text-sm text-slate-900 font-serif">
                  Next-Token Probability Distribution (Given Top-K = {topK})
                </h4>
                <span className="text-xs font-mono text-slate-400">Target: After last prompt token</span>
              </div>

              <div className="space-y-3 pt-2">
                {candidateTokens.map((cand, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold px-1.5 py-0.5 rounded bg-stone-100 text-slate-700 text-[11px]">
                          #{idx + 1}
                        </span>
                        <span className="font-bold text-slate-900 font-serif">"{cand.token}"</span>
                        <span className="text-slate-400 text-[11px] hidden sm:inline">({cand.meaning})</span>
                      </div>
                      <span className="font-mono font-bold text-blue-700">{cand.probability.toFixed(1)}%</span>
                    </div>

                    {/* Visual Probability Bar */}
                    <div className="w-full h-2.5 bg-stone-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full transition-all duration-300"
                        style={{ width: `${cand.probability}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
