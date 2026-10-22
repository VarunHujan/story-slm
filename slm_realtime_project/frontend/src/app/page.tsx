"use client";

import React, { useState, useRef, useEffect, useMemo } from "react";
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
  BookMarked,
  X,
  Check,
  Copy,
  Plus,
  Trash2,
  ArrowRight,
  LogOut
} from "lucide-react";
import confetti from "canvas-confetti";

import LoginPage, { UserSession } from "@/components/LoginPage";
import ThreeLibraryView, { STORAGE_BOOKS_KEY } from "@/components/ThreeLibraryView";
import { ThreeBookData, StoryChapter } from "@/components/ThreeLibraryCorridor";
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

type ActiveTab = "chat" | "library" | "transformer" | "pipeline" | "benchmarks";

export default function Home() {
  // Authentication & Chronicler Identity State
  const [currentUser, setCurrentUser] = useState<UserSession | null>(null);
  const [isAuthLoading, setIsAuthLoading] = useState(true);

  // 1. Primary Default View: Clean Chatbot UI
  const [activeTab, setActiveTab] = useState<ActiveTab>("chat");
  const [savedBooksCount, setSavedBooksCount] = useState<number>(0);

  // Messages state
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "slm",
      text: "Welcome to Karna-49M! I am your real-time Small Language Model trained on mythology, epic narratives, and creative fantasy. Type any story premise or dialogue to begin. When you're ready, click 'Bind Chat to 3D Library' to place this tale onto your 3D oak shelves!",
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

  // Bind to Library Modal State
  const [isBindModalOpen, setIsBindModalOpen] = useState<boolean>(false);
  const [bindTitle, setBindTitle] = useState<string>("");
  const [bindSubtitle, setBindSubtitle] = useState<string>("");
  const [bindGenre, setBindGenre] = useState<string>("Epic Mythology");
  const [bindEmblem, setBindEmblem] = useState<string>("☀️");
  const [bindLeatherColor, setBindLeatherColor] = useState<string>("#6a1b24");
  const [justBoundBook, setJustBoundBook] = useState<{ id: string; title: string; shelfLocation: string } | null>(null);

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

  // User-specific book storage key
  const userBookStorageKey = useMemo(() => {
    if (!currentUser || currentUser.profileId === "profile_varun") {
      return STORAGE_BOOKS_KEY;
    }
    return `karna_saved_books_${currentUser.profileId}`;
  }, [currentUser]);

  // Refresh saved books count for active profile
  const refreshSavedBooksCount = () => {
    try {
      const raw = localStorage.getItem(userBookStorageKey);
      if (raw) {
        const parsed: ThreeBookData[] = JSON.parse(raw);
        setSavedBooksCount(parsed.length);
      } else {
        setSavedBooksCount(0);
      }
    } catch {
      setSavedBooksCount(0);
    }
  };

  // Load persistent user session
  useEffect(() => {
    try {
      const raw = localStorage.getItem("karna_user_session_v1");
      if (raw) {
        setCurrentUser(JSON.parse(raw));
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsAuthLoading(false);
    }
  }, []);

  const handleLogin = (session: UserSession) => {
    setCurrentUser(session);
    try {
      localStorage.setItem("karna_user_session_v1", JSON.stringify(session));
    } catch (e) {
      console.error(e);
    }
  };

  const handleLogout = () => {
    if (confirm("Do you wish to depart and inscribe another chronicler seal?")) {
      setCurrentUser(null);
      try {
        localStorage.removeItem("karna_user_session_v1");
      } catch (e) {
        console.error(e);
      }
    }
  };

  useEffect(() => {
    refreshSavedBooksCount();
  }, [activeTab]);

  useEffect(() => {
    if (activeTab === "chat") {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, activeTab]);

  // Health poll
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

  // Stream Story Generation from Backend
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
              const data = JSON.parse(trimmed.slice(6));
              if (data.token) {
                if (!firstTokenTime) firstTokenTime = performance.now();
                tokenCount++;
                streamedText += data.token;

                // Live tokens per second
                const elapsedSec = (performance.now() - firstTokenTime) / 1000;
                const currentTps = elapsedSec > 0 ? Math.round(tokenCount / elapsedSec) : 0;
                setTokensPerSec(currentTps);

                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === botMsgId
                      ? { ...m, text: streamedText, tokensPerSec: currentTps }
                      : m
                  )
                );
              }
            } catch {
              // ignore partial event chunks
            }
          }
        }
      }

      setTotalTokens((prev) => prev + tokenCount);
    } catch (err) {
      console.error(err);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botMsgId
            ? { ...m, text: "Error: Could not connect to Karna-49M backend on port 8000." }
            : m
        )
      );
    } finally {
      setIsGenerating(false);
    }
  };

  // Open the Bind Modal
  const openBindModal = () => {
    const firstUserMsg = messages.find((m) => m.sender === "user");
    if (firstUserMsg && !bindTitle) {
      setBindTitle(firstUserMsg.text.slice(0, 32));
    }
    setJustBoundBook(null);
    setIsBindModalOpen(true);
  };

  // Bind Chat into a 3D Library Book
  const handleBindChatToLibrary = (e: React.FormEvent) => {
    e.preventDefault();

    // Group user prompts with the following assistant responses into chapters
    const chapters: StoryChapter[] = [];
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].sender === "user") {
        const userPrompt = messages[i].text;
        const botResponse = messages[i + 1]?.sender === "slm" ? messages[i + 1].text : "";
        if (botResponse && botResponse.trim().length > 0) {
          chapters.push({
            prompt: userPrompt,
            story: botResponse,
            timestamp: messages[i].timestamp,
            tokensPerSec: messages[i + 1].tokensPerSec
          });
        }
      }
    }

    if (chapters.length === 0) {
      alert("Please generate at least one story turn before binding into a book!");
      return;
    }

    try {
      const raw = localStorage.getItem(userBookStorageKey);
      const existing: ThreeBookData[] = raw ? JSON.parse(raw) : [];
      const count = existing.length;

      // Assign position on 3D shelves (Left 12, Right 12)
      const side = count < 12 ? "left" : "right";
      const tier = count < 12 ? Math.floor(count / 3) + 1 : Math.floor((count - 12) / 3) + 1;
      const slot = (count % 3) + 1;

      const newBook: ThreeBookData = {
        id: `book_${Date.now()}`,
        side,
        tier,
        slot,
        title: bindTitle.trim() || "Chronicle of Karna",
        subtitle: bindSubtitle.trim() || "An inscribed folio from the chat realm",
        genre: bindGenre,
        leatherColor: bindLeatherColor,
        emblem: bindEmblem,
        isInscribed: true,
        chapters,
        createdAt: new Date().toLocaleDateString()
      };

      existing.push(newBook);
      localStorage.setItem(userBookStorageKey, JSON.stringify(existing));
      setSavedBooksCount(existing.length);

      // Confetti celebration
      try {
        confetti({
          particleCount: 80,
          spread: 70,
          origin: { y: 0.6 }
        });
      } catch {}

      setJustBoundBook({
        id: newBook.id,
        title: newBook.title,
        shelfLocation: `Tier ${tier}, Slot ${slot} (${side === "left" ? "Left Wall" : "Right Wall"})`
      });
    } catch (err) {
      console.error(err);
      alert("Failed to bind book to localStorage.");
    }
  };

  // Generated stories count in current chat
  const eligibleStoriesCount = useMemo(() => {
    return messages.filter((m) => m.sender === "slm" && m.id !== "welcome" && m.text.trim().length > 0).length;
  }, [messages]);

  // Authentication Gate: Render creative Login Page if not logged in
  if (isAuthLoading) {
    return (
      <div className="min-h-screen bg-[#faf8f5] flex items-center justify-center">
        <div className="flex items-center gap-2 text-amber-800 font-serif text-sm">
          <RefreshCw className="w-4 h-4 animate-spin text-amber-600" />
          <span>Opening Sanctuary Gateway...</span>
        </div>
      </div>
    );
  }

  if (!currentUser) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <div className="flex flex-col h-screen bg-[#faf8f5] text-slate-800 font-sans selection:bg-amber-200">
      {/* Top Header Navigation */}
      <header className="h-16 border-b border-amber-900/15 bg-white/90 backdrop-blur-md px-6 flex items-center justify-between z-30 shrink-0 shadow-xs">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-600 to-amber-700 flex items-center justify-center text-white shadow-sm border border-amber-500/30">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-base text-slate-900 tracking-tight font-serif">Karna-49M</h1>
              <span className="text-[10px] font-semibold tracking-wider px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-800/20 uppercase font-mono">
                SLM Masterclass
              </span>
            </div>
            <p className="text-[11px] text-slate-600">GQA + RoPE + SwiGLU &bull; {backendHealth.parameters || "49.38M"} Parameters</p>
          </div>
        </div>

        {/* Center Main Tab Navigation */}
        <div className="flex items-center bg-amber-100/70 p-1 rounded-xl border border-amber-900/15 text-xs font-medium overflow-x-auto max-w-full shadow-xs">
          {/* TAB 1: CLEAN CHATBOT UI (DEFAULT) */}
          <button
            onClick={() => setActiveTab("chat")}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition whitespace-nowrap cursor-pointer ${
              activeTab === "chat"
                ? "bg-amber-600 text-white font-bold shadow-md shadow-amber-600/20"
                : "text-slate-700 hover:text-slate-950 hover:bg-white/60"
            }`}
          >
            <Bot className="w-4 h-4" />
            <span>💬 Chat with SLM</span>
          </button>

          {/* TAB 2: 3D LIBRARY */}
          <button
            onClick={() => setActiveTab("library")}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition whitespace-nowrap cursor-pointer ${
              activeTab === "library"
                ? "bg-amber-600 text-white font-bold shadow-md shadow-amber-600/20"
                : "text-slate-700 hover:text-slate-950 hover:bg-white/60"
            }`}
          >
            <BookOpen className="w-4 h-4" />
            <span>🏛️ 3D Library ({savedBooksCount})</span>
          </button>

          {/* TAB 3: TRANSFORMER ARCHITECTURE */}
          <button
            onClick={() => setActiveTab("transformer")}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition whitespace-nowrap ${
              activeTab === "transformer"
                ? "bg-white text-slate-900 font-bold shadow-sm border border-amber-900/10"
                : "text-slate-600 hover:text-slate-950 hover:bg-white/50"
            }`}
          >
            <Layers className="w-4 h-4 text-purple-600" />
            <span>Transformer Explainer</span>
          </button>

          {/* TAB 4: INFERENCE NODES */}
          <button
            onClick={() => setActiveTab("pipeline")}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition whitespace-nowrap ${
              activeTab === "pipeline"
                ? "bg-white text-slate-900 font-bold shadow-sm border border-amber-900/10"
                : "text-slate-600 hover:text-slate-950 hover:bg-white/50"
            }`}
          >
            <Activity className="w-4 h-4 text-emerald-600" />
            <span>Inference Nodes</span>
          </button>

          {/* TAB 5: BENCHMARKS */}
          <button
            onClick={() => setActiveTab("benchmarks")}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg transition whitespace-nowrap ${
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

          {/* Chronicler Profile & Logout Action */}
          {currentUser && (
            <div className="flex items-center gap-2 pl-3 border-l border-amber-900/15">
              <div
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-amber-50 border border-amber-200/80 text-xs shadow-2xs"
                title={`House: ${currentUser.house}\nRole: ${currentUser.title}`}
              >
                <span className="text-sm filter drop-shadow-xs">{currentUser.emblem}</span>
                <span className="font-serif font-bold text-slate-900 truncate max-w-[110px]">
                  {currentUser.username}
                </span>
              </div>
              <button
                onClick={handleLogout}
                className="p-1.5 rounded-lg text-slate-400 hover:text-rose-700 hover:bg-rose-50 transition cursor-pointer"
                title="Depart / Switch Chronicler Seal"
              >
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Workspace Body */}
      <div className="flex-1 overflow-y-auto">
        {/* ========================================================= */}
        {/* TAB 1: CLEAN CHATBOT UI WITH "BIND TO 3D LIBRARY" TASK    */}
        {/* ========================================================= */}
        {activeTab === "chat" && (
          <div className="h-[calc(100vh-64px)] grid grid-cols-1 lg:grid-cols-4 overflow-hidden">
            {/* Left Controls & Parameters Sidebar */}
            <aside className="lg:col-span-1 border-r border-amber-900/10 bg-white/70 p-5 flex flex-col gap-6 overflow-y-auto">
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-700 font-mono">
                  <Sliders className="w-4 h-4 text-amber-600" />
                  <span>Sampling Parameters</span>
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

              {/* Quick Prompt Ideas Card */}
              <div className="p-4 bg-amber-50/70 border border-amber-900/15 rounded-xl space-y-2">
                <span className="text-xs font-serif font-bold text-amber-950 block flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-amber-700" />
                  <span>Story Seed Ideas</span>
                </span>
                <p className="text-[11px] text-slate-600 leading-relaxed font-serif">
                  Try asking Karna to recount his vow to Duryodhana, describe the celestial Vijaya bow, or weave an ancient tale of hidden kingdoms.
                </p>
              </div>

              {/* Session Controls */}
              <div className="mt-auto pt-4 border-t border-amber-900/10 space-y-2">
                <button
                  onClick={() => {
                    if (confirm("Clear current conversation and start fresh?")) {
                      setMessages([]);
                      setPrompt("");
                    }
                  }}
                  className="w-full py-2 px-3 rounded-lg border border-stone-200 text-xs text-slate-600 hover:text-slate-900 hover:bg-stone-50 transition cursor-pointer"
                >
                  Clear Chat Session
                </button>
              </div>
            </aside>

            {/* Right Main Chat Workspace */}
            <main className="lg:col-span-3 flex flex-col bg-[#fdfbf7] justify-between overflow-hidden">
              {/* Chat Sub-Header with "Bind Chat to 3D Library" Action */}
              <div className="px-6 py-3 border-b border-amber-900/10 bg-white/70 backdrop-blur-md flex items-center justify-between shrink-0">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="font-serif font-bold text-sm text-slate-900">Conversational Chronicle</span>
                  <span className="text-[11px] font-mono text-slate-500">
                    &bull; {eligibleStoriesCount} {eligibleStoriesCount === 1 ? "story turn" : "story turns"} generated
                  </span>
                </div>

                {/* THE BIND TO LIBRARY BUTTON */}
                <button
                  onClick={openBindModal}
                  disabled={eligibleStoriesCount === 0}
                  className="px-4 py-1.5 rounded-xl bg-gradient-to-r from-amber-600 to-amber-700 hover:from-amber-500 hover:to-amber-600 text-white font-serif font-bold text-xs flex items-center gap-2 shadow-sm transition disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                  title={eligibleStoriesCount === 0 ? "Generate a story verse first to bind it into a physical 3D book!" : "Forge this chat into a 3D book on the library shelves"}
                >
                  <BookOpen className="w-3.5 h-3.5" />
                  <span>📖 Bind Chat to 3D Library</span>
                </button>
              </div>

              {/* Chat Messages Feed */}
              <div className="flex-1 p-6 overflow-y-auto space-y-6">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center max-w-md mx-auto space-y-4 my-auto py-12">
                    <div className="w-14 h-14 rounded-2xl bg-amber-100 border border-amber-200 text-amber-800 flex items-center justify-center shadow-sm">
                      <Bot className="w-7 h-7" />
                    </div>
                    <div className="space-y-1.5">
                      <h2 className="font-serif font-bold text-xl text-slate-900">Start Your Story With Karna</h2>
                      <p className="text-xs text-slate-600 leading-relaxed font-serif">
                        Enter any instruction or opening scene below. Once you are satisfied with the response, click <strong>"Bind Chat to 3D Library"</strong> above to store this dialogue as an inscribed book on the 3D shelves!
                      </p>
                    </div>
                  </div>
                ) : (
                  messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex gap-4 max-w-3xl ${
                        msg.sender === "user" ? "ml-auto flex-row-reverse" : ""
                      }`}
                    >
                      <div
                        className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 shadow-sm ${
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
                            ? "bg-amber-600 text-white border-amber-700 shadow-sm rounded-tr-none font-sans"
                            : "bg-white border-amber-900/15 text-slate-900 rounded-tl-none shadow-sm font-serif"
                        }`}
                      >
                        <p className="whitespace-pre-wrap">{msg.text}</p>
                        <div className={`mt-2.5 flex items-center justify-between text-[11px] border-t pt-2 ${
                          msg.sender === "user" ? "border-white/20 text-white/80 font-mono" : "border-slate-100 text-slate-500 font-mono"
                        }`}>
                          <span suppressHydrationWarning>{msg.timestamp}</span>
                          {msg.tokensPerSec && (
                            <span className={msg.sender === "user" ? "text-amber-100" : "text-emerald-700 font-bold"}>
                              {msg.tokensPerSec} tokens/sec
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Chat Input Dock */}
              <div className="p-4 bg-white/90 border-t border-amber-900/10 backdrop-blur-md">
                <div className="max-w-3xl mx-auto flex gap-3">
                  <input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && streamStory()}
                    placeholder="Ask Karna-49M to generate a story, scene, or chronicle..."
                    disabled={isGenerating}
                    className="flex-1 bg-white border border-amber-900/20 rounded-xl px-4 py-3 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:border-amber-600 transition duration-200 disabled:opacity-50 shadow-xs font-serif"
                  />
                  <button
                    onClick={streamStory}
                    disabled={isGenerating || !prompt.trim()}
                    className="px-5 py-3 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-medium text-sm flex items-center gap-2 shadow-md shadow-amber-600/20 disabled:opacity-50 disabled:cursor-not-allowed transition duration-200 cursor-pointer"
                  >
                    {isGenerating ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Generating...</span>
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

        {/* ========================================================= */}
        {/* TAB 2: 3D LIBRARY VIEW (EMPTY UNTIL CHATS ARE BOUND)     */}
        {/* ========================================================= */}
        {activeTab === "library" && (
          <ThreeLibraryView
            profileId={currentUser?.profileId}
            onNavigateToChat={() => setActiveTab("chat")}
            onOpenArchitecture={() => setActiveTab("transformer")}
          />
        )}

        {/* ========================================================= */}
        {/* TAB 3: POLO CLUB TRANSFORMER ARCHITECTURE EXPLAINER       */}
        {/* ========================================================= */}
        {activeTab === "transformer" && (
          <div className="p-6 max-w-7xl mx-auto">
            <TransformerArchitecture />
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 4: INFERENCE PIPELINE & STAGE NODES                   */}
        {/* ========================================================= */}
        {activeTab === "pipeline" && (
          <div className="p-6 max-w-7xl mx-auto">
            <InferencePipelineNodes />
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 5: BENCHMARKS & EVALUATION SUITE                     */}
        {/* ========================================================= */}
        {activeTab === "benchmarks" && (
          <div className="p-6 max-w-7xl mx-auto">
            <BenchmarkSuite />
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* BIND CHAT TO 3D LIBRARY MODAL                             */}
      {/* ========================================================= */}
      {isBindModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 animate-in fade-in">
          <div className="bg-[#faf7f2] border border-amber-900/30 rounded-2xl p-6 sm:p-7 max-w-lg w-full shadow-2xl space-y-5 text-slate-900 animate-in zoom-in-95">
            {justBoundBook ? (
              // Success Screen
              <div className="text-center py-4 space-y-4">
                <div className="w-16 h-16 mx-auto rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center shadow-inner">
                  <CheckCircle2 className="w-8 h-8" />
                </div>
                <div className="space-y-1">
                  <h3 className="font-serif font-bold text-xl text-slate-900">
                    Chronicle Successfully Bound!
                  </h3>
                  <p className="text-xs text-slate-600 font-serif">
                    <strong>"{justBoundBook.title}"</strong> has been placed on <strong>{justBoundBook.shelfLocation}</strong> of your 3D Grimoire Sanctuary!
                  </p>
                </div>
                <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
                  <button
                    onClick={() => {
                      setIsBindModalOpen(false);
                      setActiveTab("library");
                    }}
                    className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-serif font-bold text-xs flex items-center justify-center gap-2 shadow-md cursor-pointer transition"
                  >
                    <span>🏛️ Visit 3D Library Shelf</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>

                  <button
                    onClick={() => setIsBindModalOpen(false)}
                    className="w-full sm:w-auto px-4 py-2.5 rounded-xl border border-stone-300 text-slate-700 hover:bg-stone-100 text-xs font-medium cursor-pointer transition"
                  >
                    Continue Chatting
                  </button>
                </div>
              </div>
            ) : (
              // Configuration Form
              <>
                <div className="flex items-center justify-between pb-3 border-b border-amber-900/15">
                  <div className="flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-amber-700" />
                    <h3 className="font-serif text-lg font-bold text-slate-900">
                      Bind Chat into 3D Library Book
                    </h3>
                  </div>
                  <button
                    onClick={() => setIsBindModalOpen(false)}
                    className="p-1 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-stone-200/50 transition cursor-pointer"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <form onSubmit={handleBindChatToLibrary} className="space-y-4 text-xs">
                  {/* Title */}
                  <div className="space-y-1">
                    <label className="text-slate-800 font-semibold block font-serif">Book Title</label>
                    <input
                      type="text"
                      required
                      placeholder="e.g. The Legend of Karna"
                      value={bindTitle}
                      onChange={(e) => setBindTitle(e.target.value)}
                      className="w-full bg-white border border-amber-900/20 rounded-xl p-2.5 text-sm font-serif text-slate-900 focus:outline-none focus:border-amber-600 shadow-2xs"
                    />
                  </div>

                  {/* Subtitle */}
                  <div className="space-y-1">
                    <label className="text-slate-800 font-semibold block font-serif">Tagline / Subtitle</label>
                    <input
                      type="text"
                      placeholder="e.g. Celestial armor and the golden vow"
                      value={bindSubtitle}
                      onChange={(e) => setBindSubtitle(e.target.value)}
                      className="w-full bg-white border border-amber-900/20 rounded-xl p-2.5 text-xs font-serif text-slate-900 focus:outline-none focus:border-amber-600 shadow-2xs"
                    />
                  </div>

                  {/* Genre & Emblem */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <label className="text-slate-800 font-semibold block font-serif">Genre</label>
                      <select
                        value={bindGenre}
                        onChange={(e) => setBindGenre(e.target.value)}
                        className="w-full bg-white border border-amber-900/20 rounded-xl p-2.5 text-xs text-slate-900 focus:outline-none focus:border-amber-600 shadow-2xs cursor-pointer"
                      >
                        <option value="Epic Mythology">Epic Mythology</option>
                        <option value="High Fantasy">High Fantasy</option>
                        <option value="Cyberpunk Noir">Cyberpunk Noir</option>
                        <option value="Gothic Mystery">Gothic Mystery</option>
                        <option value="Ancient History">Ancient History</option>
                        <option value="Cosmic Odyssey">Cosmic Odyssey</option>
                        <option value="Animal Fables">Animal Fables</option>
                      </select>
                    </div>

                    <div className="space-y-1">
                      <label className="text-slate-800 font-semibold block font-serif">Heraldic Emblem</label>
                      <div className="flex gap-1.5 overflow-x-auto pb-1">
                        {["☀️", "🌿", "🗝️", "🏛️", "⚡", "🐾", "🌌", "👁️", "🏹", "👑"].map((icon) => (
                          <button
                            key={icon}
                            type="button"
                            onClick={() => setBindEmblem(icon)}
                            className={`p-1.5 rounded-lg text-sm transition cursor-pointer ${
                              bindEmblem === icon
                                ? "bg-amber-100 border border-amber-600 scale-110 shadow-xs"
                                : "bg-white border border-amber-900/15"
                            }`}
                          >
                            {icon}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Leather Binding Tone Color */}
                  <div className="space-y-1.5">
                    <label className="text-slate-800 font-semibold block font-serif">Leather Binding Tone</label>
                    <div className="flex gap-2.5">
                      {[
                        { hex: "#6a1b24", label: "Crimson Wine" },
                        { hex: "#1e4d2b", label: "Deep Forest" },
                        { hex: "#24334f", label: "Midnight Blue" },
                        { hex: "#8a5a36", label: "Vintage Saddle" },
                        { hex: "#45225c", label: "Imperial Purple" },
                        { hex: "#2f3136", label: "Obsidian" }
                      ].map((col) => (
                        <button
                          key={col.hex}
                          type="button"
                          onClick={() => setBindLeatherColor(col.hex)}
                          style={{ backgroundColor: col.hex }}
                          className={`w-7 h-7 rounded-full border-2 transition cursor-pointer ${
                            bindLeatherColor === col.hex ? "border-amber-600 scale-110 shadow-md ring-2 ring-amber-300" : "border-slate-300"
                          }`}
                          title={col.label}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Summary Info */}
                  <div className="p-3 bg-amber-50/70 border border-amber-900/15 rounded-xl text-[11px] text-slate-700 font-serif">
                    This will bind <strong>{eligibleStoriesCount} story chapter{eligibleStoriesCount === 1 ? "" : "s"}</strong> from your active chat into a physical book placed directly on the 3D shelves of your library.
                  </div>

                  {/* Actions */}
                  <div className="pt-2 flex justify-end gap-3">
                    <button
                      type="button"
                      onClick={() => setIsBindModalOpen(false)}
                      className="px-4 py-2 rounded-xl bg-stone-100 hover:bg-stone-200 text-slate-700 text-xs font-medium transition cursor-pointer"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-5 py-2 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-serif font-bold text-xs flex items-center gap-1.5 shadow-md shadow-amber-600/20 transition cursor-pointer"
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>✨ Bind & Place on 3D Shelf</span>
                    </button>
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
