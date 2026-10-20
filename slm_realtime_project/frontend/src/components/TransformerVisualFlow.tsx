"use client";

import React, { useState, useMemo } from "react";
import {
  Sparkles,
  Sliders,
  Eye,
  Info,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  X,
  Layers,
  Search,
  BookOpen,
  ArrowRight,
  Lightbulb,
  CheckCircle2,
  RotateCcw
} from "lucide-react";

// Preset prompts modeled after Transformer Explainer
const PRESET_PROMPTS = [
  {
    label: "Data visualization (Default)",
    text: "Data visualization empowers users to",
    predicted: "visualize",
    candidates: [
      { token: "visualize", prob: 54.67 },
      { token: "create", prob: 20.87 },
      { token: "see", prob: 12.09 },
      { token: "make", prob: 6.26 },
      { token: "easily", prob: 6.11 },
      { token: "understand", prob: 2.15 },
      { token: "explore", prob: 1.40 },
    ]
  },
  {
    label: "Prince Karna (Story Lore)",
    text: "Prince Karna raised his celestial bow Vijaya to",
    predicted: "strike",
    candidates: [
      { token: "strike", prob: 48.30 },
      { token: "defend", prob: 24.15 },
      { token: "protect", prob: 14.20 },
      { token: "aim", prob: 8.10 },
      { token: "challenge", prob: 5.25 },
      { token: "unleash", prob: 2.40 },
    ]
  },
  {
    label: "Ancient Temple",
    text: "The ancient temple hidden inside the mountains held",
    predicted: "secrets",
    candidates: [
      { token: "secrets", prob: 51.40 },
      { token: "mysteries", prob: 22.10 },
      { token: "relics", prob: 13.80 },
      { token: "treasures", prob: 7.50 },
      { token: "sacred", prob: 3.20 },
      { token: "golden", prob: 2.00 },
    ]
  },
  {
    label: "Machine Learning",
    text: "Language models transform sequences of tokens into",
    predicted: "vectors",
    candidates: [
      { token: "vectors", prob: 58.20 },
      { token: "embeddings", prob: 21.60 },
      { token: "probabilities", prob: 11.30 },
      { token: "meaning", prob: 5.40 },
      { token: "words", prob: 3.50 },
    ]
  }
];

interface TokenItem {
  id: number;
  text: string;
  index: number;
}

// Interactive Walkthrough Guide Steps
const GUIDE_STEPS = [
  {
    title: "What is a Transformer?",
    summary:
      "A Transformer is the neural backbone behind modern AI like ChatGPT and Karna-49M. It operates on sequences of tokens simultaneously using Self-Attention, allowing every word to connect with every other word in parallel.",
    highlight: "all"
  },
  {
    title: "1. Input Tokenization & Embedding",
    summary:
      "Raw text is split into tokens (words or sub-words). Each token integer is looked up in an embedding table to become a 448-dimensional continuous vector representing its semantic meaning.",
    highlight: "embedding"
  },
  {
    title: "2. The Residual Highway",
    summary:
      "The wide translucent stream arching over the top is the Residual Connection. It passes unchanged input vectors directly to the layer output, preventing the 'vanishing gradient' problem across deep networks.",
    highlight: "residual"
  },
  {
    title: "3. Queries, Keys, and Values (Q, K, V)",
    summary:
      "Inside each Attention Head, token vectors project into three roles: Queries ('What am I looking for?'), Keys ('What do I contain?'), and Values ('The actual information payload').",
    highlight: "qkv"
  },
  {
    title: "4. The Causal Attention Matrix",
    summary:
      "Queries and Keys multiply to form an N×N grid of scores. The upper-right triangle is masked out (causal mask) so words cannot look into the future. Softmax converts scores into attention percentage weights.",
    highlight: "matrix"
  },
  {
    title: "5. Grouped-Query Attention (GQA)",
    summary:
      "Karna-49M uses GQA with 8 Query heads sharing 4 Key/Value heads. This cuts KV-Cache RAM in half while preserving the multi-perspective power of 8 independent attention heads.",
    highlight: "gqa"
  },
  {
    title: "6. SwiGLU Gated Feed-Forward",
    summary:
      "After tokens gather context, they pass through SwiGLU FFN. A dual-branch gate (SiLU activation × linear projection) acts as associative memory, retrieving factual knowledge stored in the weights.",
    highlight: "swiglu"
  },
  {
    title: "7. 12 Stacked Transformer Blocks",
    summary:
      "This process repeats across 12 stacked blocks. Lower layers capture surface syntax and grammar; higher layers build deep storyline context, character relationships, and overarching narrative coherence.",
    highlight: "blocks"
  },
  {
    title: "8. LM Head & Sampling Probabilities",
    summary:
      "The final token vector is projected into a 50,304-dimensional vocabulary distribution. Temperature scales logit entropy, and Top-K selects the most probable next word to generate.",
    highlight: "probabilities"
  }
];

// Helper to construct smooth SVG ribbon paths
function ribbonPath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  h1: number,
  h2: number,
  curvature: number = 0.5
): string {
  const dx = x2 - x1;
  const p1y = y1 - h1 / 2;
  const p2y = y2 - h2 / 2;
  const p3y = y2 + h2 / 2;
  const p4y = y1 + h1 / 2;
  return `M ${x1} ${p1y} C ${x1 + dx * curvature} ${p1y}, ${x2 - dx * curvature} ${p2y}, ${x2} ${p2y} L ${x2} ${p3y} C ${x2 - dx * curvature} ${p3y}, ${x1 + dx * curvature} ${p4y}, ${x1} ${p4y} Z`;
}

// Helper to construct single smooth bezier curves
function curvePath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  curvature: number = 0.5
): string {
  const dx = x2 - x1;
  return `M ${x1} ${y1} C ${x1 + dx * curvature} ${y1}, ${x2 - dx * curvature} ${y2}, ${x2} ${y2}`;
}

interface TransformerVisualFlowProps {
  onNavigateTab?: (tab: "overview" | "attention" | "swiglu" | "rope" | "lm_head") => void;
}

export default function TransformerVisualFlow({ onNavigateTab }: TransformerVisualFlowProps) {
  // Input prompt state
  const [selectedPresetIndex, setSelectedPresetIndex] = useState<number>(0);
  const [promptText, setPromptText] = useState<string>(PRESET_PROMPTS[0].text);
  const [activeLayer, setActiveLayer] = useState<number>(0); // 0 to 11
  const [activeHead, setActiveHead] = useState<number>(0); // 0 to 7 (8 heads in GQA)
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; weight: number } | null>(null);

  // Hyperparameters
  const [temperature, setTemperature] = useState<number>(0.8);
  const [topK, setTopK] = useState<number>(5);
  const [samplingMode, setSamplingMode] = useState<"top_k" | "top_p">("top_k");

  // Floating Guided Walkthrough
  const [guideStep, setGuideStep] = useState<number>(0);
  const [showGuide, setShowGuide] = useState<boolean>(true);

  // Tokenize text into items (up to 7 tokens for visual clarity)
  const tokens: TokenItem[] = useMemo(() => {
    const rawWords = promptText.trim().split(/\s+/).filter(Boolean);
    const words = rawWords.slice(0, 7);
    return words.map((w, idx) => {
      let hash = 0;
      for (let i = 0; i < w.length; i++) {
        hash = (hash << 5) - hash + w.charCodeAt(i);
        hash |= 0;
      }
      return {
        id: Math.abs(hash % 49000) + 1000,
        text: w,
        index: idx
      };
    });
  }, [promptText]);

  const nTokens = tokens.length;

  // Active preset prediction metadata
  const activePreset = useMemo(() => {
    return PRESET_PROMPTS[selectedPresetIndex] || PRESET_PROMPTS[0];
  }, [selectedPresetIndex]);

  // Dynamically temperature-scale candidate probabilities
  const dynamicCandidates = useMemo(() => {
    const base = activePreset.candidates.slice(0, topK);
    // Apply softmax temperature scaling: prob^(1/T)
    const scaled = base.map((c) => Math.pow(c.prob, 1 / Math.max(0.2, temperature)));
    const sum = scaled.reduce((acc, v) => acc + v, 0);
    return base.map((c, i) => ({
      token: c.token,
      prob: (scaled[i] / sum) * 100
    }));
  }, [activePreset, temperature, topK]);

  // Compute simulated causal attention matrix (N x N)
  const attentionMatrix = useMemo(() => {
    const matrix: number[][] = [];
    for (let i = 0; i < nTokens; i++) {
      const row: number[] = [];
      let rowSum = 0;
      for (let j = 0; j < nTokens; j++) {
        if (j > i) {
          row.push(0); // Masked
        } else {
          // Dynamic weights based on selected head pattern
          let weight = 1.0;
          if (activeHead === 0) {
            // Local token focus
            weight = j === i ? 0.45 : j === i - 1 ? 0.35 : 0.2 / (i - j + 1);
          } else if (activeHead === 1) {
            // Global first-token focus (anchor)
            weight = j === 0 ? 0.55 : 0.45 / (i + 1);
          } else if (activeHead === 2) {
            // Subject focus (e.g. token 0 or 1)
            weight = j === 1 ? 0.6 : 0.4 / (i - j + 1);
          } else {
            // Balanced distributed focus
            weight = 1.0 / (Math.abs(i - j) + 1);
          }
          row.push(weight);
          rowSum += weight;
        }
      }
      // Softmax normalization
      const normRow = row.map((val) => (rowSum > 0 && val > 0 ? val / rowSum : 0));
      matrix.push(normRow);
    }
    return matrix;
  }, [nTokens, activeHead]);

  // Handle Preset Select
  const handleSelectPreset = (idx: number) => {
    setSelectedPresetIndex(idx);
    setPromptText(PRESET_PROMPTS[idx].text);
  };

  // Canvas geometry constants
  const canvasW = 1320;
  const canvasH = 600;

  // Token vertical spacing
  const tokenStartY = 160;
  const tokenGapY = Math.min(52, 280 / Math.max(1, nTokens));

  // Highlighting filter based on active guide step
  const currentHighlight = GUIDE_STEPS[guideStep].highlight;

  return (
    <div className="space-y-6">
      {/* ========================================================= */}
      {/* TOP HEADER CONTROLS (MODELED AFTER TRANSFORMER EXPLAINER) */}
      {/* ========================================================= */}
      <div className="p-6 rounded-2xl bg-white border border-stone-200 shadow-xs space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-xl font-black font-serif tracking-tight bg-gradient-to-r from-violet-700 via-indigo-700 to-blue-700 bg-clip-text text-transparent">
                TRANSFORMER EXPLAINER
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-violet-50 text-violet-700 border border-violet-200">
                Karna-49M Flow
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Interactive end-to-end data flow through the 12-Layer Transformer block, Grouped-Query Attention, and SwiGLU.
            </p>
          </div>

          {/* Hyperparameter Controls */}
          <div className="flex flex-wrap items-center gap-4 text-xs bg-stone-50 p-3 rounded-xl border border-stone-200">
            {/* Temperature Slider */}
            <div className="flex items-center gap-2">
              <span className="text-slate-500 font-medium">Temperature:</span>
              <input
                type="range"
                min="0.2"
                max="1.5"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-20 h-1.5 bg-stone-200 rounded accent-violet-600 cursor-pointer"
              />
              <span className="font-mono font-bold text-violet-700 w-8">{temperature.toFixed(2)}</span>
            </div>

            <div className="h-4 w-px bg-stone-300" />

            {/* Sampling Mode & Top-K */}
            <div className="flex items-center gap-2">
              <span className="text-slate-500 font-medium">Sampling:</span>
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="radio"
                  name="sampling"
                  checked={samplingMode === "top_k"}
                  onChange={() => setSamplingMode("top_k")}
                  className="accent-violet-600 cursor-pointer"
                />
                <span className="font-semibold text-slate-700">Top-k</span>
              </label>

              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="radio"
                  name="sampling"
                  checked={samplingMode === "top_p"}
                  onChange={() => setSamplingMode("top_p")}
                  className="accent-violet-600 cursor-pointer"
                />
                <span className="text-slate-600">Top-p</span>
              </label>

              <span className="text-slate-400 font-mono ml-1">k=</span>
              <input
                type="range"
                min="2"
                max="7"
                step="1"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-16 h-1.5 bg-stone-200 rounded accent-violet-600 cursor-pointer"
              />
              <span className="font-mono font-bold text-violet-700">{topK}</span>
            </div>
          </div>
        </div>

        {/* Prompt Input Row with Examples Dropdown */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
          {/* Preset Selector */}
          <div className="relative min-w-[200px]">
            <select
              value={selectedPresetIndex}
              onChange={(e) => handleSelectPreset(parseInt(e.target.value))}
              className="w-full text-xs font-medium px-3 py-2.5 rounded-xl bg-stone-50 border border-stone-200 text-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-500 cursor-pointer"
            >
              {PRESET_PROMPTS.map((p, idx) => (
                <option key={idx} value={idx}>
                  Examples: {p.label}
                </option>
              ))}
            </select>
          </div>

          {/* Editable Prompt Input Bar */}
          <div className="flex-1 flex items-center gap-2 px-4 py-2 rounded-xl bg-stone-50 border border-stone-200 focus-within:ring-2 focus-within:ring-violet-500 focus-within:bg-white transition">
            <span className="text-xs text-slate-400 font-mono">Prompt:</span>
            <input
              type="text"
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              className="flex-1 bg-transparent text-xs text-slate-900 font-serif font-medium focus:outline-none"
              placeholder="Type any sentence to see real-time Transformer token flow..."
            />
            {/* Predicted next token badge */}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-violet-100 border border-violet-200">
              <span className="text-[10px] text-violet-600 font-mono uppercase font-bold">Next:</span>
              <span className="text-xs font-bold font-serif text-violet-900">{dynamicCandidates[0]?.token}</span>
            </div>
          </div>

          <button
            onClick={() => setPromptText(PRESET_PROMPTS[selectedPresetIndex].text)}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white border border-stone-200 text-slate-700 text-xs hover:bg-stone-50 cursor-pointer"
            title="Reset prompt"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* ========================================================= */}
      {/* MAIN SANKEY / RIVER PIPELINE SVG CANVAS                   */}
      {/* ========================================================= */}
      <div className="relative rounded-2xl bg-white border border-stone-200 shadow-xs overflow-hidden">
        {/* Layer Controls Bar above Canvas */}
        <div className="flex flex-wrap items-center justify-between px-6 py-3 border-b border-stone-100 bg-stone-50/70 text-xs">
          {/* Transformer Block Navigator */}
          <div className="flex items-center gap-3">
            <span className="text-slate-400 font-mono uppercase text-[11px] font-semibold">Active Layer:</span>
            <div className="flex items-center gap-1.5 bg-white border border-stone-200 rounded-lg px-2 py-1 shadow-2xs">
              <button
                onClick={() => setActiveLayer((prev) => Math.max(0, prev - 1))}
                disabled={activeLayer === 0}
                className="p-1 rounded hover:bg-stone-100 disabled:opacity-30 cursor-pointer"
              >
                <ChevronLeft className="w-3.5 h-3.5 text-slate-700" />
              </button>
              <span className="font-serif font-bold text-slate-900 px-1">
                Transformer Block {activeLayer + 1} of 12
              </span>
              <button
                onClick={() => setActiveLayer((prev) => Math.min(11, prev + 1))}
                disabled={activeLayer === 11}
                className="p-1 rounded hover:bg-stone-100 disabled:opacity-30 cursor-pointer"
              >
                <ChevronRight className="w-3.5 h-3.5 text-slate-700" />
              </button>
            </div>
            <span className="text-[11px] text-slate-400 hidden md:inline">
              (11 identical blocks stacked behind)
            </span>
          </div>

          {/* Attention Head Navigator (8 Heads) */}
          <div className="flex items-center gap-2">
            <span className="text-slate-400 font-mono uppercase text-[11px] font-semibold">GQA Attention Head:</span>
            <div className="flex items-center gap-1">
              {[0, 1, 2, 3, 4, 5, 6, 7].map((h) => (
                <button
                  key={h}
                  onClick={() => setActiveHead(h)}
                  className={`w-6 h-6 rounded-md text-[11px] font-mono font-bold transition cursor-pointer ${
                    activeHead === h
                      ? "bg-violet-600 text-white shadow-2xs"
                      : "bg-white border border-stone-200 text-slate-600 hover:bg-stone-100"
                  }`}
                >
                  {h + 1}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* SVG Drawing Canvas Container */}
        <div className="w-full overflow-x-auto p-4 flex justify-center bg-[#fafafc]">
          <svg
            viewBox={`0 0 ${canvasW} ${canvasH}`}
            className="w-full max-w-[1320px] h-auto select-none font-sans"
            style={{ minWidth: "1150px" }}
          >
            <defs>
              {/* Gradients for ribbons */}
              <linearGradient id="residualGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#818cf8" stopOpacity="0.25" />
                <stop offset="50%" stopColor="#c084fc" stopOpacity="0.30" />
                <stop offset="100%" stopColor="#818cf8" stopOpacity="0.25" />
              </linearGradient>

              <linearGradient id="qGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.35" />
                <stop offset="100%" stopColor="#2563eb" stopOpacity="0.45" />
              </linearGradient>

              <linearGradient id="kGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.35" />
                <stop offset="100%" stopColor="#e11d48" stopOpacity="0.45" />
              </linearGradient>

              <linearGradient id="vGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.35" />
                <stop offset="100%" stopColor="#059669" stopOpacity="0.45" />
              </linearGradient>

              <linearGradient id="outGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.4" />
                <stop offset="100%" stopColor="#6d28d9" stopOpacity="0.5" />
              </linearGradient>

              <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
                <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.08" />
              </filter>
            </defs>

            {/* ========================================================= */}
            {/* STAGE LABELS (TOP OF SVG)                                 */}
            {/* ========================================================= */}
            {/* Embedding Label */}
            <g
              className="cursor-pointer group"
              onClick={() => onNavigateTab && onNavigateTab("overview")}
            >
              <text x="60" y="70" className="text-[13px] font-serif font-bold fill-slate-800">
                Embedding
              </text>
              <circle cx="132" cy="66" r="7" className="fill-stone-100 stroke-stone-300 stroke-1" />
              <text x="129" y="70" className="text-[10px] font-bold fill-slate-500">
                🔍
              </text>
            </g>

            {/* Transformer Block Label */}
            <text x="210" y="70" className="text-[14px] font-serif font-bold fill-slate-900">
              Transformer Block {activeLayer + 1}
            </text>
            <text x="210" y="86" className="text-[11px] font-sans fill-slate-500">
              Multi-head Self Attention (GQA 8:4)
            </text>

            {/* 11 more blocks stack indicator */}
            <g transform="translate(850, 48)">
              <rect x="0" y="0" width="75" height="18" rx="4" className="fill-stone-100 stroke-stone-200 stroke-1" />
              <text x="6" y="13" className="text-[9px] font-mono fill-slate-500 font-bold">
                +11 BLOCKS
              </text>
            </g>

            {/* SwiGLU / MLP Label */}
            <g
              className="cursor-pointer group"
              onClick={() => onNavigateTab && onNavigateTab("swiglu")}
            >
              <text x="735" y="70" className="text-[13px] font-serif font-bold fill-slate-800">
                SwiGLU FFN
              </text>
              <circle cx="810" cy="66" r="7" className="fill-stone-100 stroke-stone-300 stroke-1" />
              <text x="807" y="70" className="text-[10px] font-bold fill-slate-500">
                🔍
              </text>
            </g>

            {/* Probabilities Label */}
            <g
              className="cursor-pointer group"
              onClick={() => onNavigateTab && onNavigateTab("lm_head")}
            >
              <text x="1060" y="70" className="text-[13px] font-serif font-bold fill-slate-800">
                Probabilities
              </text>
              <circle cx="1145" cy="66" r="7" className="fill-stone-100 stroke-stone-300 stroke-1" />
              <text x="1142" y="70" className="text-[10px] font-bold fill-slate-500">
                🔍
              </text>
            </g>

            {/* ========================================================= */}
            {/* UPPER RESIDUAL STREAM (THE ARCHING BYPASS HIGHWAY)       */}
            {/* ========================================================= */}
            <g opacity={currentHighlight === "residual" || currentHighlight === "all" ? 1 : 0.25}>
              {/* Residual bypass 1 (over Attention) */}
              <path
                d={`M 140 ${tokenStartY - 10} C 180 50, 600 50, 680 ${tokenStartY + 20}`}
                fill="none"
                stroke="url(#residualGrad)"
                strokeWidth="24"
                strokeLinecap="round"
              />
              <path
                d={`M 140 ${tokenStartY - 10} C 180 50, 600 50, 680 ${tokenStartY + 20}`}
                fill="none"
                stroke="#6366f1"
                strokeWidth="1.5"
                strokeDasharray="4 4"
                opacity="0.6"
              />
              <text x="380" y="44" className="text-[10px] font-mono font-bold fill-indigo-600 text-center">
                Residual Bypass #1
              </text>

              {/* Residual bypass 2 (over SwiGLU) */}
              <path
                d={`M 700 ${tokenStartY + 20} C 730 65, 870 65, 910 ${tokenStartY + 30}`}
                fill="none"
                stroke="url(#residualGrad)"
                strokeWidth="20"
                strokeLinecap="round"
              />
              <text x="785" y="60" className="text-[9px] font-mono font-bold fill-indigo-600 text-center">
                Residual #2
              </text>
            </g>

            {/* ========================================================= */}
            {/* 11 STACKED TRANSFORMER BLOCKS (BACKGROUND CARDS)         */}
            {/* ========================================================= */}
            <g opacity={currentHighlight === "blocks" || currentHighlight === "all" ? 0.9 : 0.25}>
              {[5, 4, 3, 2, 1].map((offset) => (
                <rect
                  key={offset}
                  x={210 + offset * 5}
                  y={105 + offset * 4}
                  width="470"
                  height="360"
                  rx="16"
                  className="fill-white stroke-stone-200 stroke-1"
                  filter="url(#softShadow)"
                  opacity="0.4"
                />
              ))}
            </g>

            {/* Active Transformer Block Frame */}
            <rect
              x="210"
              y="105"
              width="470"
              height="360"
              rx="16"
              className="fill-white stroke-stone-300 stroke-1"
              filter="url(#softShadow)"
            />

            {/* Sub-header inside active block */}
            <text x="225" y="128" className="text-[10px] font-mono font-bold fill-slate-400">
              LAYER {activeLayer + 1} • HEAD {activeHead + 1} OF 8 (4 KV GROUPS)
            </text>

            {/* ========================================================= */}
            {/* COLUMN 1: INPUT TOKENS & EMBEDDINGS (X = 30 to 140)      */}
            {/* ========================================================= */}
            <g opacity={currentHighlight === "embedding" || currentHighlight === "all" ? 1 : 0.3}>
              {tokens.map((tok, idx) => {
                const y = tokenStartY + idx * tokenGapY;
                const isHovered = hoveredToken === idx;

                return (
                  <g
                    key={tok.id}
                    className="cursor-pointer transition-all"
                    onMouseEnter={() => setHoveredToken(idx)}
                    onMouseLeave={() => setHoveredToken(null)}
                  >
                    {/* Token background badge */}
                    <rect
                      x="30"
                      y={y - 14}
                      width="100"
                      height="28"
                      rx="6"
                      className={`transition-colors ${
                        isHovered
                          ? "fill-violet-600 stroke-violet-700 stroke-1"
                          : "fill-white stroke-stone-200 stroke-1"
                      }`}
                      filter="url(#softShadow)"
                    />

                    {/* Token text */}
                    <text
                      x="40"
                      y={y + 4}
                      className={`text-[11px] font-serif font-bold transition-colors ${
                        isHovered ? "fill-white" : "fill-slate-800"
                      }`}
                    >
                      {tok.text}
                    </text>

                    {/* Token ID pill */}
                    <text
                      x="100"
                      y={y + 3}
                      className={`text-[9px] font-mono transition-colors ${
                        isHovered ? "fill-violet-200" : "fill-slate-400"
                      }`}
                    >
                      #{tok.id % 999}
                    </text>

                    {/* 4 Embedding vector dots */}
                    {[0, 1, 2, 3].map((dot) => (
                      <circle
                        key={dot}
                        cx={145 + dot * 6}
                        cy={y}
                        r="2"
                        className={isHovered ? "fill-violet-600" : "fill-stone-300"}
                      />
                    ))}
                  </g>
                );
              })}
            </g>

            {/* ========================================================= */}
            {/* Q, K, V PROJECTION BADGES & BEZIER SPREAD (X = 170 - 240) */}
            {/* ========================================================= */}
            <g opacity={currentHighlight === "qkv" || currentHighlight === "all" ? 1 : 0.3}>
              {/* Q, K, V Badges on left edge of block */}
              {[
                { label: "Q", color: "fill-blue-500", text: "fill-white", y: 160, name: "Query" },
                { label: "K", color: "fill-rose-500", text: "fill-white", y: 220, name: "Key" },
                { label: "V", color: "fill-emerald-500", text: "fill-white", y: 280, name: "Value" }
              ].map((proj) => (
                <g key={proj.label} transform={`translate(225, ${proj.y})`}>
                  <rect x="0" y="0" width="22" height="22" rx="5" className={proj.color} />
                  <text x="6" y="15" className={`text-[11px] font-mono font-bold ${proj.text}`}>
                    {proj.label}
                  </text>
                  <text x="28" y="15" className="text-[10px] font-mono font-bold fill-slate-500">
                    {proj.name}
                  </text>
                </g>
              ))}

              {/* Connecting ribbons from Tokens to Q, K, V */}
              {tokens.map((tok, idx) => {
                const startY = tokenStartY + idx * tokenGapY;
                const isHovered = hoveredToken === idx;

                return (
                  <g key={`tok-ribbon-${idx}`} opacity={hoveredToken === null || isHovered ? 0.7 : 0.15}>
                    {/* Flow into Q */}
                    <path
                      d={curvePath(170, startY, 225, 171, 0.4)}
                      fill="none"
                      stroke="#3b82f6"
                      strokeWidth={isHovered ? 2.5 : 1.2}
                      opacity="0.6"
                    />
                    {/* Flow into K */}
                    <path
                      d={curvePath(170, startY, 225, 231, 0.4)}
                      fill="none"
                      stroke="#f43f5e"
                      strokeWidth={isHovered ? 2.5 : 1.2}
                      opacity="0.6"
                    />
                    {/* Flow into V */}
                    <path
                      d={curvePath(170, startY, 225, 291, 0.4)}
                      fill="none"
                      stroke="#10b981"
                      strokeWidth={isHovered ? 2.5 : 1.2}
                      opacity="0.6"
                    />
                  </g>
                );
              })}
            </g>

            {/* ========================================================= */}
            {/* ATTENTION MATRIX (N x N CAUSAL HEATMAP GRID)              */}
            {/* ========================================================= */}
            <g
              transform="translate(360, 160)"
              opacity={currentHighlight === "matrix" || currentHighlight === "all" ? 1 : 0.25}
            >
              {/* Matrix Background Card */}
              <rect
                x="-15"
                y="-25"
                width={nTokens * 26 + 30}
                height={nTokens * 26 + 45}
                rx="10"
                className="fill-stone-50 stroke-stone-200 stroke-1"
              />

              <text x="-5" y="-10" className="text-[11px] font-serif font-bold fill-slate-700">
                Attention Matrix (N×N)
              </text>
              <text x={nTokens * 26 - 15} y="-10" className="text-[9px] font-mono fill-violet-600 font-bold">
                Softmax(QKᵀ/√d)
              </text>

              {/* Column labels (Keys) */}
              {tokens.map((tok, j) => (
                <text
                  key={`col-tok-${j}`}
                  x={j * 26 + 10}
                  y="10"
                  textAnchor="middle"
                  className="text-[9px] font-serif fill-rose-600 font-semibold"
                >
                  {tok.text.slice(0, 3)}
                </text>
              ))}

              {/* Rows (Queries) and Cells */}
              {attentionMatrix.map((row, i) => (
                <g key={`row-${i}`}>
                  {/* Row label (Query) */}
                  <text
                    x="-8"
                    y={i * 26 + 32}
                    textAnchor="end"
                    className="text-[9px] font-serif fill-blue-600 font-semibold"
                  >
                    {tokens[i]?.text.slice(0, 3)}
                  </text>

                  {/* Cells in row i */}
                  {row.map((weight, j) => {
                    const isMasked = j > i;
                    const isCellHovered = hoveredCell?.row === i && hoveredCell?.col === j;
                    const cellX = j * 26;
                    const cellY = i * 26 + 20;

                    return (
                      <g
                        key={`cell-${i}-${j}`}
                        className="cursor-pointer"
                        onMouseEnter={() =>
                          !isMasked && setHoveredCell({ row: i, col: j, weight })
                        }
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        {isMasked ? (
                          // Masked cell (future token)
                          <circle
                            cx={cellX + 10}
                            cy={cellY + 10}
                            r="6"
                            className="fill-stone-200 stroke-stone-300 stroke-1 stroke-dasharray-[2,2]"
                            opacity="0.5"
                          />
                        ) : (
                          // Active attention cell
                          <circle
                            cx={cellX + 10}
                            cy={cellY + 10}
                            r={isCellHovered ? 9 : 7}
                            style={{
                              fill: `rgba(124, 58, 237, ${Math.max(0.12, weight * 1.8)})`
                            }}
                            className={`transition-all ${
                              isCellHovered ? "stroke-violet-800 stroke-2" : "stroke-violet-400 stroke-1"
                            }`}
                          />
                        )}
                      </g>
                    );
                  })}
                </g>
              ))}

              {/* Lower caption in matrix */}
              <text x="0" y={nTokens * 26 + 38} className="text-[8px] font-mono fill-slate-400">
                Upper triangle causal mask (locked)
              </text>
            </g>

            {/* Q, K Curves connecting into Matrix */}
            <g opacity={currentHighlight === "matrix" || currentHighlight === "all" ? 0.7 : 0.2}>
              {/* Q lines to matrix rows */}
              {tokens.map((_, i) => (
                <path
                  key={`q-to-mat-${i}`}
                  d={curvePath(290, 171, 350, 180 + i * 26, 0.4)}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="1.2"
                  opacity="0.5"
                />
              ))}

              {/* K lines curving around top to matrix columns */}
              {tokens.map((_, j) => (
                <path
                  key={`k-to-mat-${j}`}
                  d={curvePath(290, 231, 360 + j * 26, 150, 0.5)}
                  fill="none"
                  stroke="#f43f5e"
                  strokeWidth="1.2"
                  opacity="0.5"
                />
              ))}
            </g>

            {/* ========================================================= */}
            {/* ATTENTION OUTPUT & MERGE (X = 580 to 680)                 */}
            {/* ========================================================= */}
            <g opacity={currentHighlight === "matrix" || currentHighlight === "all" ? 1 : 0.3}>
              <text x="590" y="195" className="text-[11px] font-mono font-bold fill-violet-700">
                Out
              </text>

              {/* Purple output lines converging from matrix */}
              {tokens.map((_, i) => (
                <path
                  key={`out-line-${i}`}
                  d={curvePath(360 + nTokens * 26 + 10, 180 + i * 26, 620, 210 + i * 14, 0.4)}
                  fill="none"
                  stroke="url(#outGrad)"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              ))}

              {/* Output projection vector bar */}
              <rect x="625" y="195" width="8" height={tokens.length * 14 + 10} rx="3" className="fill-violet-600" />
            </g>

            {/* ========================================================= */}
            {/* RESIDUAL ADDITION NODE & RMSNORM 1 (X = 660 to 710)       */}
            {/* ========================================================= */}
            <g
              transform="translate(660, 220)"
              opacity={currentHighlight === "residual" || currentHighlight === "all" ? 1 : 0.3}
            >
              {/* Residual Addition circle */}
              <circle cx="20" cy="10" r="14" className="fill-white stroke-indigo-500 stroke-2" filter="url(#softShadow)" />
              <text x="14" y="15" className="text-[14px] font-bold fill-indigo-700">
                +
              </text>
              <text x="-5" y="36" className="text-[9px] font-mono font-bold fill-slate-500">
                Add & Norm
              </text>

              {/* RMSNorm Bar */}
              <rect x="44" y="-5" width="8" height="30" rx="3" className="fill-stone-400" />
              <text x="36" y="36" className="text-[8px] font-mono fill-slate-400">
                RMS
              </text>
            </g>

            {/* ========================================================= */}
            {/* SWIGLU FEED-FORWARD NETWORK BLOCK (X = 730 to 880)        */}
            {/* ========================================================= */}
            <g opacity={currentHighlight === "swiglu" || currentHighlight === "all" ? 1 : 0.25}>
              {/* SwiGLU Outer Card */}
              <rect
                x="730"
                y="115"
                width="145"
                height="340"
                rx="14"
                className="fill-white stroke-stone-300 stroke-1"
                filter="url(#softShadow)"
              />

              <text x="742" y="138" className="text-[11px] font-serif font-bold fill-slate-900">
                SwiGLU Gated FFN
              </text>
              <text x="742" y="152" className="text-[9px] font-mono fill-emerald-600 font-bold">
                d_ffn = 1,216
              </text>

              {/* Upper Gate Branch */}
              <g transform="translate(745, 175)">
                <rect x="0" y="0" width="115" height="42" rx="6" className="fill-amber-50 stroke-amber-200 stroke-1" />
                <text x="8" y="16" className="text-[9px] font-mono font-bold fill-amber-800">
                  Gate Branch:
                </text>
                <text x="8" y="30" className="text-[9px] font-mono fill-slate-600">
                  xW_gate · SiLU(·)
                </text>
              </g>

              {/* Multiplier Node (⊗) */}
              <circle cx="802" cy="245" r="10" className="fill-white stroke-stone-400 stroke-1" />
              <text x="797" y="249" className="text-[12px] font-bold fill-slate-700">
                ⊗
              </text>

              {/* Lower Up Branch */}
              <g transform="translate(745, 270)">
                <rect x="0" y="0" width="115" height="42" rx="6" className="fill-blue-50 stroke-blue-200 stroke-1" />
                <text x="8" y="16" className="text-[9px] font-mono font-bold fill-blue-800">
                  Up Branch:
                </text>
                <text x="8" y="30" className="text-[9px] font-mono fill-slate-600">
                  xW_up (Linear)
                </text>
              </g>

              {/* Down Projection W_down */}
              <g transform="translate(745, 335)">
                <rect x="0" y="0" width="115" height="34" rx="6" className="fill-emerald-50 stroke-emerald-200 stroke-1" />
                <text x="8" y="15" className="text-[9px] font-mono font-bold fill-emerald-800">
                  Down Projection:
                </text>
                <text x="8" y="26" className="text-[8px] font-mono fill-slate-500">
                  W_down (1216 → 448)
                </text>
              </g>

              {/* Connecting paths into SwiGLU */}
              <path
                d={curvePath(715, 230, 745, 196, 0.3)}
                fill="none"
                stroke="#d97706"
                strokeWidth="1.5"
              />
              <path
                d={curvePath(715, 230, 745, 291, 0.3)}
                fill="none"
                stroke="#2563eb"
                strokeWidth="1.5"
              />
              <path
                d="M 802 217 L 802 235"
                fill="none"
                stroke="#d97706"
                strokeWidth="1.5"
              />
              <path
                d="M 802 270 L 802 255"
                fill="none"
                stroke="#2563eb"
                strokeWidth="1.5"
              />
              <path
                d="M 802 255 L 802 335"
                fill="none"
                stroke="#059669"
                strokeWidth="1.5"
              />
            </g>

            {/* Second Residual Addition & Norm */}
            <g
              transform="translate(895, 340)"
              opacity={currentHighlight === "residual" || currentHighlight === "all" ? 1 : 0.3}
            >
              <circle cx="15" cy="10" r="12" className="fill-white stroke-indigo-500 stroke-2" filter="url(#softShadow)" />
              <text x="10" y="14" className="text-[13px] font-bold fill-indigo-700">
                +
              </text>
              <rect x="34" y="-2" width="6" height="24" rx="2" className="fill-stone-400" />
            </g>

            {/* ========================================================= */}
            {/* TUNNEL BRIDGE: 11 MORE IDENTICAL BLOCKS (X = 940 to 1030) */}
            {/* ========================================================= */}
            <g opacity={currentHighlight === "blocks" || currentHighlight === "all" ? 1 : 0.25}>
              <path
                d="M 940 340 C 970 340, 990 320, 1030 320"
                fill="none"
                stroke="url(#residualGrad)"
                strokeWidth="18"
                strokeLinecap="round"
              />

              {/* Repeating blocks indicator */}
              {[0, 1, 2].map((i) => (
                <rect
                  key={i}
                  x={955 + i * 18}
                  y={290}
                  width="12"
                  height="70"
                  rx="3"
                  className="fill-white stroke-stone-300 stroke-1"
                  filter="url(#softShadow)"
                  opacity={0.6 + i * 0.15}
                />
              ))}

              <text x="950" y="380" className="text-[9px] font-mono font-bold fill-slate-400 text-center">
                Layers 2 → 12
              </text>
            </g>

            {/* Final token vector projection (last token e.g. "to") */}
            <g opacity={currentHighlight === "probabilities" || currentHighlight === "all" ? 1 : 0.3}>
              <path
                d="M 1030 320 C 1050 320, 1060 280, 1080 200"
                fill="none"
                stroke="#8b5cf6"
                strokeWidth="3"
              />
              <circle cx="1030" cy="320" r="4" className="fill-violet-600" />
              <text x="1010" y="305" className="text-[9px] font-mono font-bold fill-violet-700">
                Last Token: "{tokens[nTokens - 1]?.text}"
              </text>
            </g>

            {/* ========================================================= */}
            {/* PROBABILITY DISTRIBUTION COLUMN (X = 1060 to 1290)        */}
            {/* ========================================================= */}
            <g
              transform="translate(1060, 105)"
              opacity={currentHighlight === "probabilities" || currentHighlight === "all" ? 1 : 0.25}
            >
              {/* Card Container */}
              <rect
                x="0"
                y="0"
                width="230"
                height="360"
                rx="14"
                className="fill-white stroke-stone-300 stroke-1"
                filter="url(#softShadow)"
              />

              <text x="16" y="26" className="text-[12px] font-serif font-bold fill-slate-900">
                Next-Token Probabilities
              </text>
              <text x="16" y="40" className="text-[9px] font-mono fill-slate-400">
                Softmax Logits (Vocab: 50,304)
              </text>

              {/* Candidate tokens list */}
              {dynamicCandidates.map((cand, idx) => {
                const y = 60 + idx * 38;
                const isTop1 = idx === 0;

                return (
                  <g key={cand.token} transform={`translate(16, ${y})`}>
                    {/* Candidate word label */}
                    <text
                      x="0"
                      y="12"
                      className={`text-[12px] font-serif font-bold ${
                        isTop1 ? "fill-violet-700" : "fill-slate-700"
                      }`}
                    >
                      {cand.token}
                    </text>

                    {/* Percentage text */}
                    <text
                      x="195"
                      y="12"
                      textAnchor="end"
                      className={`text-[11px] font-mono font-bold ${
                        isTop1 ? "fill-violet-700" : "fill-slate-500"
                      }`}
                    >
                      {cand.prob.toFixed(1)}%
                    </text>

                    {/* Horizontal Probability Bar */}
                    <rect x="0" y="18" width="195" height="7" rx="3.5" className="fill-stone-100" />
                    <rect
                      x="0"
                      y="18"
                      width={Math.max(4, (cand.prob / 100) * 195)}
                      height="7"
                      rx="3.5"
                      className={`transition-all duration-300 ${
                        isTop1 ? "fill-violet-600" : "fill-stone-400"
                      }`}
                    />
                  </g>
                );
              })}

              <text x="16" y="340" className="text-[9px] font-mono fill-violet-600 font-bold">
                T = {temperature.toFixed(2)} • Top-K = {topK}
              </text>
            </g>

            {/* Hover Tooltip for Attention Matrix Cell */}
            {hoveredCell && (
              <g transform={`translate(${380 + hoveredCell.col * 26}, ${130 + hoveredCell.row * 26})`}>
                <rect
                  x="-75"
                  y="-40"
                  width="150"
                  height="34"
                  rx="6"
                  className="fill-slate-900 stroke-slate-800 stroke-1"
                  filter="url(#softShadow)"
                />
                <text x="0" y="-24" textAnchor="middle" className="text-[10px] font-serif fill-white font-bold">
                  "{tokens[hoveredCell.row]?.text}" → "{tokens[hoveredCell.col]?.text}"
                </text>
                <text x="0" y="-12" textAnchor="middle" className="text-[9px] font-mono fill-violet-300">
                  Weight: {(hoveredCell.weight * 100).toFixed(1)}%
                </text>
              </g>
            )}
          </svg>
        </div>

        {/* ========================================================= */}
        {/* FLOATING INTERACTIVE GUIDE / WALKTHROUGH DRAWER           */}
        {/* ========================================================= */}
        {showGuide && (
          <div className="absolute bottom-5 right-6 w-96 max-w-[calc(100vw-3rem)] p-4 rounded-2xl bg-white/95 backdrop-blur-md border border-stone-300 shadow-xl space-y-3 animate-in fade-in slide-in-from-bottom-2">
            <div className="flex items-center justify-between border-b border-stone-200 pb-2">
              <div className="flex items-center gap-2">
                <span className="p-1 rounded bg-violet-100 text-violet-700">
                  <Lightbulb className="w-3.5 h-3.5" />
                </span>
                <h4 className="font-serif font-bold text-slate-900 text-xs">
                  {GUIDE_STEPS[guideStep].title}
                </h4>
              </div>
              <button
                onClick={() => setShowGuide(false)}
                className="p-1 text-slate-400 hover:text-slate-600 rounded cursor-pointer"
                title="Minimize guide"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            <p className="text-[11px] text-slate-600 leading-relaxed">
              {GUIDE_STEPS[guideStep].summary}
            </p>

            {/* Stepper Footer */}
            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center gap-1">
                {GUIDE_STEPS.map((_, idx) => (
                  <button
                    key={idx}
                    onClick={() => setGuideStep(idx)}
                    className={`h-1.5 rounded-full transition-all cursor-pointer ${
                      guideStep === idx ? "w-4 bg-violet-600" : "w-1.5 bg-stone-300"
                    }`}
                  />
                ))}
              </div>

              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => setGuideStep((prev) => Math.max(0, prev - 1))}
                  disabled={guideStep === 0}
                  className="p-1 rounded bg-stone-100 text-slate-700 disabled:opacity-30 cursor-pointer"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <span className="text-[10px] font-mono text-slate-500 font-bold px-1">
                  {guideStep + 1} / {GUIDE_STEPS.length}
                </span>
                <button
                  onClick={() => setGuideStep((prev) => Math.min(GUIDE_STEPS.length - 1, prev + 1))}
                  disabled={guideStep === GUIDE_STEPS.length - 1}
                  className="p-1 rounded bg-stone-100 text-slate-700 disabled:opacity-30 cursor-pointer"
                >
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Toggle guide open button if minimized */}
        {!showGuide && (
          <button
            onClick={() => setShowGuide(true)}
            className="absolute bottom-5 right-6 flex items-center gap-2 px-3 py-1.5 rounded-full bg-violet-600 text-white text-xs font-bold shadow-lg hover:bg-violet-700 cursor-pointer"
          >
            <Lightbulb className="w-3.5 h-3.5" />
            <span>Open Architecture Guide ({guideStep + 1}/{GUIDE_STEPS.length})</span>
          </button>
        )}
      </div>

      {/* Quick Deep-Dive Navigation Shortcuts */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
        <button
          onClick={() => onNavigateTab && onNavigateTab("attention")}
          className="p-3 rounded-xl bg-white border border-stone-200 text-left hover:border-violet-300 hover:shadow-2xs transition group cursor-pointer"
        >
          <span className="text-[10px] font-mono uppercase text-violet-600 font-bold block">Deep Dive</span>
          <span className="text-xs font-serif font-bold text-slate-800 group-hover:text-violet-700">
            Grouped-Query Attention →
          </span>
        </button>

        <button
          onClick={() => onNavigateTab && onNavigateTab("rope")}
          className="p-3 rounded-xl bg-white border border-stone-200 text-left hover:border-violet-300 hover:shadow-2xs transition group cursor-pointer"
        >
          <span className="text-[10px] font-mono uppercase text-violet-600 font-bold block">Deep Dive</span>
          <span className="text-xs font-serif font-bold text-slate-800 group-hover:text-violet-700">
            Rotary Embeddings (RoPE) →
          </span>
        </button>

        <button
          onClick={() => onNavigateTab && onNavigateTab("swiglu")}
          className="p-3 rounded-xl bg-white border border-stone-200 text-left hover:border-violet-300 hover:shadow-2xs transition group cursor-pointer"
        >
          <span className="text-[10px] font-mono uppercase text-violet-600 font-bold block">Deep Dive</span>
          <span className="text-xs font-serif font-bold text-slate-800 group-hover:text-violet-700">
            SwiGLU Gated FFN →
          </span>
        </button>

        <button
          onClick={() => onNavigateTab && onNavigateTab("lm_head")}
          className="p-3 rounded-xl bg-white border border-stone-200 text-left hover:border-violet-300 hover:shadow-2xs transition group cursor-pointer"
        >
          <span className="text-[10px] font-mono uppercase text-violet-600 font-bold block">Deep Dive</span>
          <span className="text-xs font-serif font-bold text-slate-800 group-hover:text-violet-700">
            LM Head & Sampling Lab →
          </span>
        </button>
      </div>
    </div>
  );
}
