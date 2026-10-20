"use client";

import React, { useState, useRef, useEffect } from "react";
import { 
  Sparkles, 
  Send, 
  Cpu, 
  Zap, 
  Sliders, 
  Layers, 
  Activity,
  BookOpen,
  User,
  Bot,
  RefreshCw,
  BarChart3,
  Terminal,
  HelpCircle,
  Eye,
  CheckCircle2,
  ExternalLink,
  BookMarked
} from "lucide-react";

import StoryStudio from "@/components/StoryStudio";
import TransformerArchitecture from "@/components/TransformerArchitecture";
import InferencePipelineNodes from "@/components/InferencePipelineNodes";
import BenchmarkSuite from "@/components/BenchmarkSuite";

interface Message {
  id: string;
  sender: "user" | "slm";
  text: string;
  timestamp: string;
  tokensPerSec?: number;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

type ActiveTab = "studio" | "chat" | "transformer" | "pipeline" | "benchmarks";

export default function Home() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("studio");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "slm",
      text: "Welcome to Karna-49M Studio! I am a 49.38 Million parameter Small Language Model trained with Grouped-Query Attention (GQA), RoPE, and SwiGLU architecture. You can craft multi-chapter tales in the Story Studio, test prompts in the chat terminal, or explore the Transformer Architecture and Benchmarks.",
      timestamp: "Just now"
    }
  ]);
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.75);
  const [maxTokens, setMaxTokens] = useState(256);
  const [topK, setTopK] = useState(40);
  const [tokensPerSec, setTokensPerSec] = useState<number>(0);
  const [totalTokens, setTotalTokens] = useState<number>(0);
  const [backendHealth, setBackendHealth] = useState<{
    status: string;
    model: string;
    device: string;
    checkpoint_loaded: boolean;
    parameters?: string;
  }>({
    status: "connecting",
    model: "Karna-49M",
    device: "CPU",
    checkpoint_loaded: false,
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeTab === "chat") {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, activeTab]);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/health`);
        if (res.ok) {
          const data = await res.json();
          setBackendHealth(data);
        } else {
          setBackendHealth((prev) => ({ ...prev, checkpoint_loaded: false, status: "error" }));
        }
      } catch {
        setBackendHealth((prev) => ({ ...prev, checkpoint_loaded: false, status: "offline" }));
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 4000);
    return () => clearInterval(interval);
  }, []);

  const streamStory = async () => {
    if (!prompt.trim() || isGenerating) return;

    const userMsgText = prompt.trim();
    const userMsgId = "user_" + Date.now();
    const botMsgId = "bot_" + Date.now();
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const newMessages: Message[] = [
      ...messages,
      { id: userMsgId, sender: "user", text: userMsgText, timestamp: timeStr },
      { id: botMsgId, sender: "slm", text: "", timestamp: timeStr }
    ];

    setMessages(newMessages);
    setPrompt("");
    setIsGenerating(true);

    let firstTokenTime: number | null = null;
    let tokenCount = 0;
    let buffer = "";

    try {
      const response = await fetch(`${BACKEND_URL}/api/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: userMsgText,
          max_tokens: maxTokens,
          temperature: temperature,
          top_k: topK
        })
      });

      if (!response.body) throw new Error("ReadableStream not supported");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let streamedText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const event of events) {
          const trimmed = event.trim();
          if (trimmed.startsWith("data: ") && !trimmed.includes("[DONE]")) {
            try {
              const data = JSON.parse(trimmed.replace(/^data:\s*/, ""));
              streamedText += data.token;
              tokenCount++;

              if (!firstTokenTime) {
                firstTokenTime = performance.now();
              }
              const elapsedSec = (performance.now() - firstTokenTime) / 1000;
              const currentTps = (tokenCount > 1 && elapsedSec > 0.05)
                ? Math.round(tokenCount / elapsedSec)
                : 38;

              setTokensPerSec(currentTps);
              setTotalTokens((prev) => prev + 1);

              setMessages((prevMsgs) =>
                prevMsgs.map((msg) =>
                  msg.id === botMsgId
                    ? { ...msg, text: streamedText, tokensPerSec: currentTps }
                    : msg
                )
              );
            } catch (err) {
              console.error("JSON parse error", err);
            }
          }
        }
      }
    } catch (err: any) {
      setMessages((prevMsgs) =>
        prevMsgs.map((msg) =>
          msg.id === botMsgId
            ? { ...msg, text: `[Backend Error: Ensure server.py is running on ${BACKEND_URL} | ${err.message}]` }
            : msg
        )
      );
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#faf7f2] text-slate-900 flex flex-col font-sans selection:bg-amber-500 selection:text-white">
      {/* Universal Top Header */}
      <header className="border-b border-amber-900/10 bg-white/90 backdrop-blur-xl px-6 py-2.5 flex flex-col md:flex-row items-center justify-between gap-3 sticky top-0 z-50 shadow-xs">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-amber-600 to-yellow-600 p-0.5 flex items-center justify-center shadow-md shadow-amber-500/20">
            <div className="w-full h-full bg-amber-600 rounded-[10px] flex items-center justify-center">
              <Zap className="w-4 h-4 text-white fill-white" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-base text-slate-900 tracking-tight font-serif">Karna-49M</h1>
              <span className="text-[10px] font-semibold tracking-wider px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-800/20 uppercase font-mono">
                Folio Studio
              </span>
            </div>
            <p className="text-[11px] text-slate-600">GQA + RoPE + SwiGLU &bull; {backendHealth.parameters || "49.38M"} Parameters</p>
          </div>
        </div>

        {/* Center Main Tab Navigation */}
        <div className="flex items-center bg-amber-100/70 p-1 rounded-xl border border-amber-900/15 text-xs font-medium overflow-x-auto max-w-full shadow-xs">
          <button
            onClick={() => setActiveTab("studio")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition whitespace-nowrap cursor-pointer ${
              activeTab === "studio"
                ? "bg-amber-600 text-white font-bold shadow-md shadow-amber-600/20"
                : "text-slate-700 hover:text-slate-950 hover:bg-white/60"
            }`}
          >
            <BookOpen className="w-4 h-4" />
            <span>✨ Story Studio</span>
          </button>

          <button
            onClick={() => setActiveTab("chat")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition whitespace-nowrap cursor-pointer ${
              activeTab === "chat"
                ? "bg-white text-slate-900 font-bold shadow-sm border border-amber-900/10"
                : "text-slate-600 hover:text-slate-950 hover:bg-white/50"
            }`}
          >
            <Bot className="w-4 h-4" />
            <span>Chat Playground</span>
          </button>

          <button
            onClick={() => setActiveTab("transformer")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition whitespace-nowrap ${
              activeTab === "transformer"
                ? "bg-white text-slate-900 font-bold shadow-sm border border-amber-900/10"
                : "text-slate-600 hover:text-slate-950 hover:bg-white/50"
            }`}
          >
            <Layers className="w-4 h-4 text-purple-600" />
            <span>Transformer Explainer</span>
          </button>

          <button
            onClick={() => setActiveTab("pipeline")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition whitespace-nowrap ${
              activeTab === "pipeline"
                ? "bg-white text-slate-900 font-bold shadow-sm border border-amber-900/10"
                : "text-slate-600 hover:text-slate-950 hover:bg-white/50"
            }`}
          >
            <Activity className="w-4 h-4 text-emerald-600" />
            <span>Inference Nodes</span>
          </button>

          <button
            onClick={() => setActiveTab("benchmarks")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition whitespace-nowrap ${
              activeTab === "benchmarks"
                ? "bg-white text-slate-900 font-bold shadow-sm border border-amber-900/10"
                : "text-slate-600 hover:text-slate-950 hover:bg-white/50"
            }`}
          >
            <BarChart3 className="w-4 h-4 text-rose-600" />
            <span>Benchmarks</span>
          </button>
        </div>

        {/* Telemetry Chips */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-white border border-amber-900/15 text-xs text-slate-700 shadow-xs">
            <Activity className="w-3.5 h-3.5 text-emerald-600 animate-pulse" />
            <span className="text-slate-500">Speed:</span>
            <span className="font-mono font-bold text-emerald-700">{tokensPerSec} tok/s</span>
          </div>

          <div className={`flex items-center gap-2 px-2.5 py-1 rounded-full border text-xs font-medium ${
            backendHealth.checkpoint_loaded
              ? "bg-emerald-50 border-emerald-300 text-emerald-800"
              : "bg-amber-50 border-amber-300 text-amber-800"
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              backendHealth.checkpoint_loaded ? "bg-emerald-500 shadow-[0_0_8px_#10b981]" : "bg-amber-500 shadow-[0_0_8px_#f59e0b]"
            }`} />
            {backendHealth.checkpoint_loaded ? "Online" : "Connecting..."}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto">
        {/* TAB 1: MODERN STORY STUDIO */}
        {activeTab === "studio" && (
          <StoryStudio
            onOpenArchitecture={() => setActiveTab("transformer")}
            onOpenPipeline={() => setActiveTab("pipeline")}
          />
        )}

        {/* TAB 2: CLASSIC CHAT TERMINAL */}
        {activeTab === "chat" && (
          <div className="h-[calc(100vh-65px)] grid grid-cols-1 lg:grid-cols-4 overflow-hidden">
            {/* Left Sidebar Controls */}
            <aside className="lg:col-span-1 border-r border-amber-900/10 bg-white/70 p-5 flex flex-col gap-6 overflow-y-auto">
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-700">
                  <Sliders className="w-4 h-4 text-amber-600" />
                  <span>Hyperparameters</span>
                </div>

                <div className="bg-white border border-amber-900/15 rounded-xl p-4 space-y-4 shadow-sm text-slate-800">
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-slate-600">Temperature</span>
                      <span className="font-mono text-amber-700 font-bold">{temperature}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.5"
                      step="0.05"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-amber-100 rounded-lg appearance-none cursor-pointer accent-amber-600"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-slate-600">Max Tokens</span>
                      <span className="font-mono text-amber-700 font-bold">{maxTokens}</span>
                    </div>
                    <input
                      type="range"
                      min="32"
                      max="2048"
                      step="32"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      className="w-full h-1.5 bg-amber-100 rounded-lg appearance-none cursor-pointer accent-amber-600"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-slate-600">Top-K Sampling</span>
                      <span className="font-mono text-amber-700 font-bold">{topK}</span>
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      step="5"
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value))}
                      className="w-full h-1.5 bg-amber-100 rounded-lg appearance-none cursor-pointer accent-amber-600"
                    />
                  </div>
                </div>
              </div>
            </aside>

            {/* Right Main Chat Workspace */}
            <main className="lg:col-span-3 flex flex-col bg-[#fdfbf7] justify-between overflow-hidden">
              <div className="flex-1 p-6 overflow-y-auto space-y-6">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex gap-4 max-w-3xl ${
                      msg.sender === "user" ? "ml-auto flex-row-reverse" : ""
                    }`}
                  >
                    <div
                      className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 shadow-md ${
                        msg.sender === "user"
                          ? "bg-amber-600 text-white"
                          : "bg-gradient-to-tr from-amber-700 to-yellow-600 text-white"
                      }`}
                    >
                      {msg.sender === "user" ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                    </div>

                    <div
                      className={`rounded-2xl p-5 border text-sm leading-relaxed ${
                        msg.sender === "user"
                          ? "bg-amber-600 text-white border-amber-700 shadow-sm rounded-tr-none"
                          : "bg-white border-amber-900/15 text-slate-900 rounded-tl-none shadow-sm"
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{msg.text}</p>
                      <div className={`mt-2.5 flex items-center justify-between text-[11px] border-t pt-2 ${
                        msg.sender === "user" ? "border-white/20 text-white/80" : "border-slate-100 text-slate-500"
                      }`}>
                        <span suppressHydrationWarning>{msg.timestamp}</span>
                        {msg.tokensPerSec && (
                          <span className={`font-mono ${msg.sender === "user" ? "text-amber-100" : "text-emerald-700 font-bold"}`}>
                            {msg.tokensPerSec} tokens/sec
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Dock */}
              <div className="p-4 bg-white/90 border-t border-amber-900/10 backdrop-blur-md">
                <div className="max-w-3xl mx-auto flex gap-3">
                  <input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && streamStory()}
                    placeholder="Ask Karna-49M to generate a story..."
                    disabled={isGenerating}
                    className="flex-1 bg-white border border-amber-900/20 rounded-xl px-4 py-3 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:border-amber-600 transition duration-200 disabled:opacity-50 shadow-xs"
                  />
                  <button
                    onClick={streamStory}
                    disabled={isGenerating || !prompt.trim()}
                    className="px-5 py-3 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-medium text-sm flex items-center gap-2 shadow-md shadow-amber-600/20 disabled:opacity-50 disabled:cursor-not-allowed transition duration-200 cursor-pointer"
                  >
                    {isGenerating ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Streaming...</span>
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4" />
                        <span>Generate</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </main>
          </div>
        )}

        {/* TAB 3: POLO CLUB TRANSFORMER ARCHITECTURE EXPLAINER */}
        {activeTab === "transformer" && (
          <div className="p-6 max-w-7xl mx-auto">
            <TransformerArchitecture />
          </div>
        )}

        {/* TAB 4: INFERENCE PIPELINE & STAGE NODES */}
        {activeTab === "pipeline" && (
          <div className="p-6 max-w-7xl mx-auto">
            <InferencePipelineNodes />
          </div>
        )}

        {/* TAB 5: BENCHMARKS & EVALUATION SUITE */}
        {activeTab === "benchmarks" && (
          <div className="p-6 max-w-7xl mx-auto">
            <BenchmarkSuite />
          </div>
        )}
      </div>
    </div>
  );
}
