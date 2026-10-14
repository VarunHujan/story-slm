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
  RefreshCw
} from "lucide-react";

interface Message {
  id: string;
  sender: "user" | "slm";
  text: string;
  timestamp: string;
  tokensPerSec?: number;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "slm",
      text: "Welcome to Story-SLM2 Studio! I am a 49.38 Million parameter Small Language Model trained on GQA Attention, RoPE, & SwiGLU architecture. Enter a prompt below to stream responses in real-time.",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.75);
  const [maxTokens, setMaxTokens] = useState(256);
  const [topK, setTopK] = useState(40);
  const [tokensPerSec, setTokensPerSec] = useState<number>(0);
  const [totalTokens, setTotalTokens] = useState<number>(0);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handlePresetPrompt = (presetText: string) => {
    setPrompt(presetText);
  };

  const streamStory = async () => {
    if (!prompt.trim() || isGenerating) return;

    const userMsgText = prompt.trim();
    const userMsgId = "user_" + Date.now();
    const botMsgId = "bot_" + Date.now();
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Append User Message
    const newMessages: Message[] = [
      ...messages,
      { id: userMsgId, sender: "user", text: userMsgText, timestamp: timeStr },
      { id: botMsgId, sender: "slm", text: "", timestamp: timeStr }
    ];

    setMessages(newMessages);
    setPrompt("");
    setIsGenerating(true);

    const startTime = performance.now();
    let tokenCount = 0;

    try {
      let response;
      try {
        response = await fetch("http://localhost:8000/api/generate/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: userMsgText,
            max_tokens: maxTokens,
            temperature: temperature,
            top_k: topK
          })
        });
      } catch (err) {
        response = await fetch("http://127.0.0.1:8000/api/generate/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: userMsgText,
            max_tokens: maxTokens,
            temperature: temperature,
            top_k: topK
          })
        });
      }

      if (!response.body) throw new Error("ReadableStream not supported");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let streamedText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ") && !line.includes("[DONE]")) {
            try {
              const data = JSON.parse(line.replace("data: ", ""));
              streamedText += data.token;
              tokenCount++;

              const elapsedSec = (performance.now() - startTime) / 1000;
              const currentTps = elapsedSec > 0 ? Math.round(tokenCount / elapsedSec) : 0;
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
            ? { ...msg, text: `[Backend Error: Ensure server.py is running on http://localhost:8000 | ${err.message}]` }
            : msg
        )
      );
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-blue-500 selection:text-white">
      {/* Header */}
      <header className="h-16 border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-xl px-6 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-purple-600 p-0.5 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Zap className="w-5 h-5 text-blue-400 fill-blue-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-lg text-slate-100 tracking-tight">Story-SLM2</h1>
              <span className="text-[10px] font-semibold tracking-wider px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 uppercase">
                Next.js Studio
              </span>
            </div>
            <p className="text-xs text-slate-400">GQA + RoPE + SwiGLU &bull; 49.38M Parameters &bull; FastAPI Backend</p>
          </div>
        </div>

        {/* Telemetry Chips */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
            <span className="text-slate-400">Speed:</span>
            <span className="font-mono font-bold text-emerald-400">{tokensPerSec} tok/s</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <Cpu className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-slate-400">Device:</span>
            <span className="font-mono font-bold text-purple-400">PyTorch GPU</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#10b981]"></span>
            Model Online
          </div>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 overflow-hidden">
        {/* Left Sidebar Controls */}
        <aside className="lg:col-span-1 border-r border-slate-800/80 bg-slate-900/40 p-5 flex flex-col gap-6 overflow-y-auto">
          {/* Hyperparameters Card */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
              <Sliders className="w-4 h-4 text-blue-400" />
              <span>Hyperparameters</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-4 shadow-sm">
              {/* Temperature */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-300">Temperature</span>
                  <span className="font-mono text-blue-400 font-bold">{temperature}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <p className="text-[11px] text-slate-500">Higher = creative, Lower = deterministic</p>
              </div>

              {/* Max Tokens */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-300">Max Tokens</span>
                  <span className="font-mono text-blue-400 font-bold">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="32"
                  max="2048"
                  step="32"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              {/* Top-K */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-300">Top-K Sampling</span>
                  <span className="font-mono text-blue-400 font-bold">{topK}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>
            </div>
          </div>

          {/* Preset Prompts Card */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
              <BookOpen className="w-4 h-4 text-purple-400" />
              <span>Sample Prompts</span>
            </div>

            <div className="space-y-2">
              <button
                onClick={() => handlePresetPrompt("Tell a story about a brave little girl named Lily who found a magical golden key.")}
                className="w-full text-left p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-blue-500/50 hover:bg-slate-800/50 transition duration-200 group"
              >
                <p className="text-xs font-semibold text-slate-200 group-hover:text-blue-400">✨ The Magical Golden Key</p>
                <p className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">Lily discovers a glowing key in an enchanted forest...</p>
              </button>

              <button
                onClick={() => handlePresetPrompt("Once upon a time, a friendly puppy named Max lost his favorite ball in the park.")}
                className="w-full text-left p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-purple-500/50 hover:bg-slate-800/50 transition duration-200 group"
              >
                <p className="text-xs font-semibold text-slate-200 group-hover:text-purple-400">🐶 Max's Park Adventure</p>
                <p className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">A friendly puppy goes on a journey to find his ball...</p>
              </button>

              <button
                onClick={() => handlePresetPrompt("Who created you and what is your architecture?")}
                className="w-full text-left p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-emerald-500/50 hover:bg-slate-800/50 transition duration-200 group"
              >
                <p className="text-xs font-semibold text-slate-200 group-hover:text-emerald-400">⚡ Model Identity Prompt</p>
                <p className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">Inquire about the model architecture and creator...</p>
              </button>
            </div>
          </div>
        </aside>

        {/* Right Main Chat Workspace */}
        <main className="lg:col-span-3 flex flex-col bg-slate-950/60 justify-between overflow-hidden">
          {/* Chat Stream Window */}
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
                      ? "bg-blue-600 text-white"
                      : "bg-gradient-to-tr from-purple-600 to-indigo-600 text-white"
                  }`}
                >
                  {msg.sender === "user" ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                </div>

                <div
                  className={`rounded-2xl p-5 border text-sm leading-relaxed ${
                    msg.sender === "user"
                      ? "bg-blue-600/10 border-blue-500/30 text-slate-100 rounded-tr-none"
                      : "bg-slate-900/80 border-slate-800/80 text-slate-200 rounded-tl-none shadow-sm"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>

                  <div className="mt-2.5 flex items-center justify-between text-[11px] text-slate-500 border-t border-slate-800/60 pt-2">
                    <span>{msg.timestamp}</span>
                    {msg.tokensPerSec && (
                      <span className="font-mono text-emerald-400">{msg.tokensPerSec} tokens/sec</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Dock */}
          <div className="p-4 bg-slate-900/80 border-t border-slate-800/80 backdrop-blur-md">
            <div className="max-w-3xl mx-auto flex gap-3">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && streamStory()}
                placeholder="Ask Story-SLM 50M to generate a story..."
                disabled={isGenerating}
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 transition duration-200 disabled:opacity-50"
              />
              <button
                onClick={streamStory}
                disabled={isGenerating || !prompt.trim()}
                className="px-5 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium text-sm flex items-center gap-2 shadow-lg shadow-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition duration-200"
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
    </div>
  );
}
