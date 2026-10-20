"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  BookOpen,
  Plus,
  Trash2,
  Copy,
  Check,
  Send,
  ChevronRight,
  Sliders,
  Search,
  Activity,
  Feather
} from "lucide-react";
import confetti from "canvas-confetti";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";
const STORAGE_STORIES_KEY = "karna_studio_stories_v2";

export interface Chapter {
  id: string;
  chapterNumber: number;
  prompt: string;
  story: string;
  timestamp: string;
  tokensPerSec?: number;
}

export interface StorySession {
  id: string;
  title: string;
  genre: string;
  createdAt: string;
  updatedAt: string;
  chapters: Chapter[];
}

interface StoryStudioProps {
  onOpenArchitecture?: () => void;
  onOpenPipeline?: () => void;
}

export default function StoryStudio({ onOpenArchitecture, onOpenPipeline }: StoryStudioProps) {
  // Pure User Stories State (starts empty, 0 hardcoded templates)
  const [stories, setStories] = useState<StorySession[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      const saved = localStorage.getItem(STORAGE_STORIES_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [activeStoryId, setActiveStoryId] = useState<string>(() => {
    return stories[0]?.id || "";
  });

  const [activeChapterIndex, setActiveChapterIndex] = useState<number>(0);
  const [viewMode, setViewMode] = useState<"chapter" | "continuous">("chapter");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Input & Generation State (100% from scratch)
  const [prompt, setPrompt] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [activeStreamingText, setActiveStreamingText] = useState<string>("");
  const [copied, setCopied] = useState<boolean>(false);

  // Initial scratch form state (when stories list is empty)
  const [scratchTitle, setScratchTitle] = useState<string>("");
  const [scratchGenre, setScratchGenre] = useState<string>("Fantasy");
  const [scratchPrompt, setScratchPrompt] = useState<string>("");

  // Hyperparameters
  const [temperature, setTemperature] = useState<number>(0.75);
  const [maxTokens, setMaxTokens] = useState<number>(380);
  const [topK, setTopK] = useState<number>(40);
  const [showSliders, setShowSliders] = useState<boolean>(false);

  // Telemetry
  const [tokensPerSec, setTokensPerSec] = useState<number>(0);
  const [totalTokensGenerated, setTotalTokensGenerated] = useState<number>(0);

  // Modal for New Story
  const [isCreatingNewStory, setIsCreatingNewStory] = useState<boolean>(false);
  const [newTitle, setNewTitle] = useState<string>("");
  const [newGenre, setNewGenre] = useState<string>("Fantasy");
  const [newInitialPrompt, setNewInitialPrompt] = useState<string>("");

  // Persist to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_STORIES_KEY, JSON.stringify(stories));
    } catch (e) {
      console.error(e);
    }
  }, [stories]);

  const activeStory = useMemo(() => {
    return stories.find((s) => s.id === activeStoryId) || stories[0] || null;
  }, [stories, activeStoryId]);

  const activeChapter = useMemo(() => {
    if (!activeStory || activeStory.chapters.length === 0) return null;
    return activeStory.chapters[activeChapterIndex] || activeStory.chapters[activeStory.chapters.length - 1];
  }, [activeStory, activeChapterIndex]);

  // Handle Starting First Story from Scratch
  const handleStartFromScratch = async (e: React.FormEvent) => {
    e.preventDefault();
    const title = scratchTitle.trim() || "Untitled Chronicle";
    const promptText = scratchPrompt.trim();
    if (!promptText || isGenerating) return;

    const newId = "story_" + Date.now();
    const newStory: StorySession = {
      id: newId,
      title: title,
      genre: scratchGenre,
      createdAt: "Just now",
      updatedAt: "Just now",
      chapters: []
    };

    // Add to stories
    const updated = [newStory, ...stories];
    setStories(updated);
    setActiveStoryId(newId);
    setActiveChapterIndex(0);

    // Reset scratch form
    setScratchTitle("");
    setScratchPrompt("");

    // Stream Chapter 1 immediately
    await executeStream(newStory, 1, promptText, updated);
  };

  // Handle Creating a New Story via Modal
  const handleCreateNewStoryModal = async (e: React.FormEvent) => {
    e.preventDefault();
    const title = newTitle.trim() || "Untitled Story";
    const promptText = newInitialPrompt.trim();

    const newId = "story_" + Date.now();
    const createdStory: StorySession = {
      id: newId,
      title: title,
      genre: newGenre,
      createdAt: "Just now",
      updatedAt: "Just now",
      chapters: []
    };

    const updated = [createdStory, ...stories];
    setStories(updated);
    setActiveStoryId(newId);
    setActiveChapterIndex(0);
    setIsCreatingNewStory(false);
    setNewTitle("");
    setNewInitialPrompt("");

    if (promptText) {
      await executeStream(createdStory, 1, promptText, updated);
    }
  };

  // Delete a Story
  const handleDeleteStory = (e: React.MouseEvent, storyId: string) => {
    e.stopPropagation();
    const updated = stories.filter((s) => s.id !== storyId);
    setStories(updated);
    if (activeStoryId === storyId) {
      setActiveStoryId(updated[0]?.id || "");
      setActiveChapterIndex(0);
    }
  };

  // Common Stream Execution Engine
  const executeStream = async (
    targetStory: StorySession,
    chapterNumber: number,
    textToSend: string,
    currentStoriesList: StorySession[]
  ) => {
    const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const newChapterId = "ch_" + Date.now();

    // Context from previous chapters (narrative continuity)
    const previousStorySnippets = targetStory.chapters
      .slice(-3)
      .map((ch) => `Chapter ${ch.chapterNumber}: ${ch.story}`)
      .join("\n\n");

    const promptPayload = previousStorySnippets
      ? `${previousStorySnippets}\n\nChapter ${chapterNumber} Inscription Prompt: ${textToSend}\nChapter ${chapterNumber}:`
      : textToSend;

    setIsGenerating(true);
    setActiveStreamingText("");

    let firstTokenTime: number | null = null;
    let tokenCount = 0;
    let buffer = "";
    let accumulated = "";

    try {
      const response = await fetch(`${BACKEND_URL}/api/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: promptPayload,
          max_tokens: maxTokens,
          temperature: temperature,
          top_k: topK
        })
      });

      if (!response.body) throw new Error("ReadableStream not supported");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

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
              accumulated += data.token;
              setActiveStreamingText(accumulated);
              tokenCount++;

              if (!firstTokenTime) {
                firstTokenTime = performance.now();
              }
              const elapsedSec = (performance.now() - firstTokenTime) / 1000;
              const currentTps = (tokenCount > 1 && elapsedSec > 0.05)
                ? Math.round(tokenCount / elapsedSec)
                : 39;

              setTokensPerSec(currentTps);
              setTotalTokensGenerated((prev) => prev + 1);
            } catch (err) {
              console.error("JSON parse error", err);
            }
          }
        }
      }

      if (accumulated.trim()) {
        const completedChapter: Chapter = {
          id: newChapterId,
          chapterNumber: chapterNumber,
          prompt: textToSend,
          story: accumulated.trim(),
          timestamp: timeStr,
          tokensPerSec: tokensPerSec || 39
        };

        const updatedChapters = [...targetStory.chapters, completedChapter];
        const nextStories = currentStoriesList.map((s) =>
          s.id === targetStory.id
            ? { ...s, chapters: updatedChapters, updatedAt: "Just now" }
            : s
        );

        setStories(nextStories);
        setActiveChapterIndex(updatedChapters.length - 1);

        try {
          confetti({
            particleCount: 30,
            spread: 55,
            origin: { y: 0.7 },
            colors: ["#d97706", "#f59e0b", "#fbbf24"]
          });
        } catch {}
      }
    } catch (err: any) {
      const errorChapter: Chapter = {
        id: newChapterId,
        chapterNumber: chapterNumber,
        prompt: textToSend,
        story: `[Inference Error: Ensure Karna-49M FastAPI server is running on ${BACKEND_URL}. Details: ${err.message}]`,
        timestamp: timeStr
      };
      const updatedChapters = [...targetStory.chapters, errorChapter];
      setStories(currentStoriesList.map((s) => s.id === targetStory.id ? { ...s, chapters: updatedChapters } : s));
      setActiveChapterIndex(updatedChapters.length - 1);
    } finally {
      setIsGenerating(false);
      setActiveStreamingText("");
    }
  };

  // Stream Next Chapter in Active Story
  const handleStreamNextChapter = async () => {
    if (!activeStory || isGenerating) return;
    const textToSend = prompt.trim();
    if (!textToSend) return;

    setPrompt("");
    const nextChapterNum = activeStory.chapters.length + 1;
    await executeStream(activeStory, nextChapterNum, textToSend, stories);
  };

  // Copy Entire Story Transcript
  const handleCopyStory = () => {
    if (!activeStory || activeStory.chapters.length === 0) return;
    const transcript = [
      `══════════════════════════════════════════════════════════════════`,
      `TITLE: ${activeStory.title}`,
      `GENRE: ${activeStory.genre}`,
      `CHAPTERS: ${activeStory.chapters.length}`,
      `══════════════════════════════════════════════════════════════════\n`,
      ...activeStory.chapters.map((ch) =>
        `[CHAPTER ${ch.chapterNumber}]\nPROMPT: ${ch.prompt}\n\n${ch.story}\n\n`
      )
    ].join("\n");

    navigator.clipboard.writeText(transcript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Filtered Stories
  const filteredStories = useMemo(() => {
    if (!searchQuery.trim()) return stories;
    const q = searchQuery.toLowerCase();
    return stories.filter(
      (s) =>
        s.title.toLowerCase().includes(q) ||
        s.genre.toLowerCase().includes(q)
    );
  }, [stories, searchQuery]);

  return (
    <div className="flex h-[calc(100vh-65px)] bg-[#faf8f5] text-slate-900 font-sans overflow-hidden">
      {/* ========================================================= */}
      {/* LEFT SIDEBAR: USER'S STORY SESSIONS                       */}
      {/* ========================================================= */}
      <aside className="w-80 border-r border-stone-200 bg-white/80 backdrop-blur-md flex flex-col justify-between shrink-0">
        <div className="p-4 space-y-3 flex-1 overflow-y-auto">
          {/* New Story Session Button */}
          <button
            onClick={() => setIsCreatingNewStory(true)}
            className="w-full py-2.5 px-3.5 rounded-xl bg-amber-600 hover:bg-amber-700 text-white font-medium text-xs flex items-center justify-center gap-2 shadow-sm transition cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            <span>New Story Session</span>
          </button>

          {/* Search Bar (if stories exist) */}
          {stories.length > 0 && (
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search your stories..."
                className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-stone-100/70 border border-stone-200 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:bg-white focus:border-amber-500 transition"
              />
            </div>
          )}

          {/* Story Sessions List */}
          <div className="pt-2 space-y-1.5">
            <div className="flex items-center justify-between text-[10px] font-mono uppercase text-slate-400 font-semibold px-1">
              <span>Your Stories ({filteredStories.length})</span>
              <span>Karna-49M</span>
            </div>

            {filteredStories.length === 0 ? (
              <div className="p-6 text-center text-slate-400 space-y-2 border border-dashed border-stone-200 rounded-xl my-2">
                <Feather className="w-8 h-8 text-stone-300 mx-auto" />
                <p className="text-xs font-serif">No stories created yet.</p>
                <p className="text-[11px] text-slate-400">
                  Write your first prompt from scratch to begin.
                </p>
              </div>
            ) : (
              filteredStories.map((s) => {
                const isActive = s.id === activeStoryId;
                return (
                  <div
                    key={s.id}
                    onClick={() => {
                      setActiveStoryId(s.id);
                      setActiveChapterIndex(Math.max(0, s.chapters.length - 1));
                    }}
                    className={`group relative p-3 rounded-xl border text-left transition cursor-pointer ${
                      isActive
                        ? "bg-amber-50/80 border-amber-300 shadow-xs"
                        : "bg-white border-stone-200/80 hover:border-amber-200 hover:bg-stone-50"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <BookOpen
                            className={`w-3.5 h-3.5 shrink-0 ${
                              isActive ? "text-amber-600" : "text-slate-400 group-hover:text-amber-600"
                            }`}
                          />
                          <h4 className="text-xs font-semibold text-slate-900 truncate font-serif">
                            {s.title}
                          </h4>
                        </div>
                        <p className="text-[11px] text-slate-500 line-clamp-1 mt-0.5">
                          {s.chapters.length > 0
                            ? s.chapters[s.chapters.length - 1].story
                            : "Awaiting Chapter 1..."}
                        </p>
                      </div>

                      <button
                        onClick={(e) => handleDeleteStory(e, s.id)}
                        className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-50 hover:text-red-600 text-slate-400 transition cursor-pointer"
                        title="Delete Story"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>

                    <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                      <span className="px-1.5 py-0.5 rounded bg-stone-100 text-slate-600 border border-stone-200">
                        {s.genre}
                      </span>
                      <span>
                        {s.chapters.length} {s.chapters.length === 1 ? "Chapter" : "Chapters"}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Sidebar Footer Info */}
        <div className="p-3 border-t border-stone-200 bg-stone-50/70 text-[11px] text-slate-600 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-mono font-medium text-slate-700">Karna-49M SLM</span>
          </div>
          <span className="text-[10px] text-slate-400 font-mono">GQA &bull; 8:4</span>
        </div>
      </aside>

      {/* ========================================================= */}
      {/* MAIN WORKSPACE                                            */}
      {/* ========================================================= */}
      <main className="flex-1 flex flex-col justify-between overflow-hidden bg-[#faf8f5]">
        {/* CASE 1: NO STORIES YET - PURE SCRATCH ONBOARDING FORM */}
        {stories.length === 0 ? (
          <div className="flex-1 overflow-y-auto p-6 md:p-12 flex items-center justify-center">
            <div className="max-w-xl w-full bg-white rounded-2xl border border-stone-200 shadow-sm p-8 space-y-6 animate-in fade-in">
              <div className="text-center space-y-2">
                <div className="w-12 h-12 rounded-2xl bg-amber-100 border border-amber-200 text-amber-800 flex items-center justify-center mx-auto shadow-xs">
                  <Feather className="w-6 h-6" />
                </div>
                <h2 className="text-xl font-bold font-serif text-slate-900">
                  Write Your Story From Scratch
                </h2>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  No prebuilt templates or hardcoded data. Name your chronicle and type your prompt to begin Chapter 1.
                </p>
              </div>

              <form onSubmit={handleStartFromScratch} className="space-y-4 text-xs text-slate-800">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="font-semibold block mb-1 text-slate-700">Story Title</label>
                    <input
                      type="text"
                      required
                      placeholder="e.g. Echoes of the Void"
                      value={scratchTitle}
                      onChange={(e) => setScratchTitle(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-stone-300 focus:outline-none focus:border-amber-600 bg-white"
                    />
                  </div>

                  <div>
                    <label className="font-semibold block mb-1 text-slate-700">Genre / Style</label>
                    <select
                      value={scratchGenre}
                      onChange={(e) => setScratchGenre(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-stone-300 focus:outline-none focus:border-amber-600 bg-white"
                    >
                      <option value="Fantasy">Fantasy</option>
                      <option value="Sci-Fi">Sci-Fi</option>
                      <option value="Mystery">Mystery</option>
                      <option value="Mythology">Mythology</option>
                      <option value="Gothic Horror">Gothic Horror</option>
                      <option value="Adventure">Adventure</option>
                      <option value="Custom">Custom</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="font-semibold block mb-1 text-slate-700">
                    Chapter 1 Opening Prompt (From Scratch)
                  </label>
                  <textarea
                    required
                    rows={4}
                    placeholder="Type your opening prompt from scratch... Describe your world, protagonist, or opening conflict."
                    value={scratchPrompt}
                    onChange={(e) => setScratchPrompt(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-xl border border-stone-300 focus:outline-none focus:border-amber-600 resize-none font-serif text-xs leading-relaxed"
                  />
                </div>

                <button
                  type="submit"
                  disabled={isGenerating || !scratchPrompt.trim()}
                  className="w-full py-3 rounded-xl bg-amber-600 hover:bg-amber-700 text-white font-medium text-xs flex items-center justify-center gap-2 shadow-sm transition disabled:opacity-50 cursor-pointer"
                >
                  {isGenerating ? (
                    <>
                      <Activity className="w-4 h-4 animate-spin" />
                      <span>Inking Chapter 1...</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      <span>Begin Story & Inscribe Chapter 1 ↵</span>
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        ) : (
          /* CASE 2: ACTIVE STORY WORKSPACE */
          <>
            {/* Top Header Bar of Active Story */}
            <header className="px-6 py-3 border-b border-stone-200 bg-white/80 backdrop-blur-md flex flex-wrap items-center justify-between gap-3 shrink-0">
              <div className="flex items-center gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 text-[10px] font-mono uppercase font-semibold">
                      {activeStory?.genre}
                    </span>
                    <h2 className="text-base font-bold text-slate-900 font-serif tracking-tight">
                      {activeStory?.title}
                    </h2>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {/* View Mode Toggle */}
                {activeStory?.chapters.length > 0 && (
                  <div className="flex items-center bg-stone-100 p-0.5 rounded-lg border border-stone-200 text-xs">
                    <button
                      onClick={() => setViewMode("chapter")}
                      className={`px-2.5 py-1 rounded-md transition cursor-pointer ${
                        viewMode === "chapter"
                          ? "bg-white text-slate-900 font-semibold shadow-xs"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      Chapter View
                    </button>
                    <button
                      onClick={() => setViewMode("continuous")}
                      className={`px-2.5 py-1 rounded-md transition cursor-pointer ${
                        viewMode === "continuous"
                          ? "bg-white text-slate-900 font-semibold shadow-xs"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      Full Story
                    </button>
                  </div>
                )}

                {/* Hyperparameters Button */}
                <button
                  onClick={() => setShowSliders(!showSliders)}
                  className={`p-2 rounded-lg border text-xs flex items-center gap-1.5 transition cursor-pointer ${
                    showSliders
                      ? "bg-amber-100 border-amber-300 text-amber-900"
                      : "bg-white border-stone-200 text-slate-700 hover:bg-stone-50"
                  }`}
                  title="Quill Hyperparameters"
                >
                  <Sliders className="w-3.5 h-3.5" />
                  <span className="hidden md:inline">Sliders</span>
                </button>

                {/* Copy Story Button */}
                {activeStory?.chapters.length > 0 && (
                  <button
                    onClick={handleCopyStory}
                    className="p-2 rounded-lg bg-white border border-stone-200 text-slate-700 hover:bg-stone-50 transition cursor-pointer flex items-center gap-1.5 text-xs shadow-xs"
                    title="Copy Full Story Transcript"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                    <span className="hidden sm:inline">{copied ? "Copied" : "Copy Story"}</span>
                  </button>
                )}
              </div>
            </header>

            {/* Sliders Drawer */}
            {showSliders && (
              <div className="px-6 py-3 bg-stone-50 border-b border-stone-200 grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs animate-in slide-in-from-top-1">
                <div className="space-y-1">
                  <div className="flex justify-between font-mono">
                    <span className="text-slate-600">Temperature (Creativity):</span>
                    <span className="font-bold text-amber-700">{temperature}</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="1.4"
                    step="0.05"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-stone-200 rounded accent-amber-600 cursor-pointer"
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between font-mono">
                    <span className="text-slate-600">Chapter Length (Max Tokens):</span>
                    <span className="font-bold text-amber-700">{maxTokens}</span>
                  </div>
                  <input
                    type="range"
                    min="128"
                    max="800"
                    step="32"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-stone-200 rounded accent-amber-600 cursor-pointer"
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between font-mono">
                    <span className="text-slate-600">Top-K Sampling:</span>
                    <span className="font-bold text-amber-700">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="80"
                    step="5"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-stone-200 rounded accent-amber-600 cursor-pointer"
                  />
                </div>
              </div>
            )}

            {/* Chapter Navigation Tabs Bar */}
            {viewMode === "chapter" && activeStory && activeStory.chapters.length > 0 && (
              <div className="px-6 py-2 border-b border-stone-200/80 bg-white/40 flex items-center justify-between text-xs overflow-x-auto gap-2">
                <div className="flex items-center gap-1.5">
                  {activeStory.chapters.map((ch, idx) => (
                    <button
                      key={ch.id}
                      onClick={() => setActiveChapterIndex(idx)}
                      className={`px-3 py-1 rounded-lg transition cursor-pointer flex items-center gap-1 font-serif text-xs ${
                        idx === activeChapterIndex
                          ? "bg-amber-600 text-white font-bold shadow-xs"
                          : "bg-white border border-stone-200 text-slate-700 hover:bg-stone-50"
                      }`}
                    >
                      <span>Chapter {ch.chapterNumber}</span>
                      <span className="text-[10px] opacity-70">✓</span>
                    </button>
                  ))}

                  {/* Next Chapter Add Pill */}
                  <button
                    onClick={() => {
                      setActiveChapterIndex(activeStory.chapters.length);
                      setPrompt("");
                    }}
                    className={`px-3 py-1 rounded-lg border border-dashed text-xs transition cursor-pointer flex items-center gap-1 ${
                      activeChapterIndex === activeStory.chapters.length
                        ? "bg-amber-100 border-amber-400 text-amber-900 font-bold"
                        : "border-stone-300 text-slate-500 hover:bg-stone-100"
                    }`}
                  >
                    <Plus className="w-3 h-3" />
                    <span>Chapter {activeStory.chapters.length + 1} (Blank)</span>
                  </button>
                </div>

                <span className="text-[11px] font-mono text-slate-400 shrink-0">
                  {tokensPerSec > 0 && `${tokensPerSec} tok/s • `}
                  {activeStory.chapters.length} Chapters Inscribed
                </span>
              </div>
            )}

            {/* Reading Workspace */}
            <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 max-w-4xl mx-auto w-full">
              {viewMode === "chapter" && (
                <>
                  {/* Viewing an existing completed chapter */}
                  {activeChapter && activeChapterIndex < activeStory.chapters.length ? (
                    <div className="space-y-6 animate-in fade-in">
                      {/* User's Prompt Card */}
                      <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-200/80 text-xs font-serif text-slate-800 space-y-1.5 shadow-xs">
                        <div className="flex items-center justify-between text-[10px] font-mono uppercase text-amber-900/70 font-semibold">
                          <span className="flex items-center gap-1.5">
                            <Feather className="w-3.5 h-3.5 text-amber-700" />
                            <span>Chapter {activeChapter.chapterNumber} Prompt</span>
                          </span>
                          <span>{activeChapter.timestamp}</span>
                        </div>
                        <p className="italic text-slate-800 leading-relaxed pl-1 whitespace-pre-wrap">
                          "{activeChapter.prompt}"
                        </p>
                      </div>

                      {/* Story Text Manuscript */}
                      <article className="p-6 md:p-8 rounded-2xl bg-white border border-stone-200/90 shadow-sm space-y-4">
                        <div className="flex items-center justify-between border-b border-stone-100 pb-3 text-xs text-slate-500 font-mono">
                          <span className="font-serif font-bold text-slate-900 text-sm">
                            Chapter {activeChapter.chapterNumber}
                          </span>
                          {activeChapter.tokensPerSec && (
                            <span>Speed: {activeChapter.tokensPerSec} tok/s</span>
                          )}
                        </div>

                        <div className="font-serif text-base md:text-lg leading-relaxed text-slate-800 whitespace-pre-wrap select-text drop-cap">
                          {activeChapter.story}
                        </div>

                        <div className="flex items-center justify-center pt-4 text-stone-300 text-xs font-mono">
                          ✦ &bull; ✦ &bull; ✦
                        </div>
                      </article>

                      {/* NEXT PAGE / NEXT CHAPTER BUTTON */}
                      <div className="p-4 rounded-xl bg-white border border-stone-200 flex flex-col sm:flex-row items-center justify-between gap-3 shadow-xs">
                        <div className="text-xs text-slate-600 font-serif italic">
                          Chapter {activeChapter.chapterNumber} concluded. Ready for the next development?
                        </div>

                        {activeChapterIndex < activeStory.chapters.length - 1 ? (
                          <button
                            onClick={() => setActiveChapterIndex(activeChapterIndex + 1)}
                            className="w-full sm:w-auto px-4 py-2 rounded-lg bg-stone-900 hover:bg-stone-800 text-white font-medium text-xs flex items-center justify-center gap-2 shadow-xs transition cursor-pointer"
                          >
                            <span>Next Chapter ({activeChapterIndex + 2})</span>
                            <ChevronRight className="w-4 h-4" />
                          </button>
                        ) : (
                          <button
                            onClick={() => {
                              setActiveChapterIndex(activeStory.chapters.length);
                              setPrompt("");
                            }}
                            className="w-full sm:w-auto px-5 py-2.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-medium text-xs flex items-center justify-center gap-2 shadow-sm transition cursor-pointer group"
                          >
                            <Feather className="w-3.5 h-3.5 text-amber-200 group-hover:rotate-12 transition transform" />
                            <span>Continue Tale & Inscribe Chapter {activeStory.chapters.length + 1} ➔</span>
                          </button>
                        )}
                      </div>
                    </div>
                  ) : (
                    /* BLANK CHAPTER READY FOR WRITING */
                    <div className="space-y-6 animate-in fade-in">
                      <div className="text-center py-6 space-y-2">
                        <div className="w-12 h-12 rounded-2xl bg-amber-100 border border-amber-200 text-amber-800 flex items-center justify-center mx-auto shadow-xs">
                          <Feather className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-serif font-bold text-slate-900">
                          Inscribe Chapter {activeStory.chapters.length + 1}
                        </h3>
                        <p className="text-xs text-slate-500 max-w-md mx-auto">
                          {activeStory.chapters.length > 0
                            ? `Continue the saga from Chapter ${activeStory.chapters.length}. Write what happens next below.`
                            : `Write the opening chapter prompt below to begin "${activeStory.title}".`}
                        </p>
                      </div>

                      {/* Previous Chapter Context Banner */}
                      {activeStory.chapters.length > 0 && (
                        <div className="p-3.5 rounded-xl bg-stone-100/80 border border-stone-200 text-xs font-serif text-slate-700 space-y-1">
                          <span className="text-[10px] font-mono uppercase text-slate-500 font-semibold block">
                            Continuing from Chapter {activeStory.chapters.length}:
                          </span>
                          <p className="italic line-clamp-2">
                            "{activeStory.chapters[activeStory.chapters.length - 1].story.slice(-140)}..."
                          </p>
                        </div>
                      )}

                      {/* Active Live Streaming Display (if generating this chapter) */}
                      {isGenerating && (
                        <article className="p-6 md:p-8 rounded-2xl bg-white border border-amber-300 shadow-md space-y-3">
                          <div className="flex items-center justify-between text-xs font-mono text-amber-800">
                            <span className="flex items-center gap-2 font-bold">
                              <span className="w-2 h-2 rounded-full bg-amber-600 animate-ping" />
                              <span>Generating Chapter {activeStory.chapters.length + 1} ({tokensPerSec} tok/s)...</span>
                            </span>
                          </div>
                          <div className="font-serif text-base md:text-lg leading-relaxed text-slate-900 whitespace-pre-wrap select-text drop-cap">
                            {activeStreamingText}
                            <span className="inline-block w-2 h-4 bg-amber-600 animate-pulse ml-1 translate-y-0.5" />
                          </div>
                        </article>
                      )}
                    </div>
                  )}
                </>
              )}

              {/* VIEW MODE 2: CONTINUOUS FULL STORY VIEW */}
              {viewMode === "continuous" && (
                <div className="p-8 md:p-12 rounded-2xl bg-white border border-stone-200 shadow-sm space-y-8 animate-in fade-in">
                  <div className="text-center pb-6 border-b border-stone-100 space-y-2">
                    <span className="text-xs font-mono uppercase text-amber-700 font-semibold tracking-wider">
                      {activeStory.genre} Chronicle
                    </span>
                    <h1 className="text-2xl md:text-3xl font-serif font-bold text-slate-900">
                      {activeStory.title}
                    </h1>
                  </div>

                  {activeStory.chapters.map((ch) => (
                    <section key={ch.id} className="space-y-3 border-b border-stone-100 pb-8 last:border-b-0">
                      <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                        <span className="font-serif font-bold text-slate-900 text-sm">
                          Chapter {ch.chapterNumber}
                        </span>
                        <span>{ch.timestamp}</span>
                      </div>
                      <div className="p-2.5 rounded bg-amber-50/50 border border-amber-100 text-xs italic font-serif text-slate-600">
                        Prompt: "{ch.prompt}"
                      </div>
                      <div className="font-serif text-base md:text-lg leading-relaxed text-slate-800 whitespace-pre-wrap drop-cap">
                        {ch.story}
                      </div>
                    </section>
                  ))}

                  {isGenerating && (
                    <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-200 text-xs font-mono text-amber-800 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-amber-600 animate-ping" />
                      <span>Writing next chapter ({tokensPerSec} tok/s)...</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Prompt & Continuation Input Dock (Clean, No Hardcoded Buttons) */}
            <div className="p-4 md:px-6 bg-white/95 border-t border-stone-200 backdrop-blur-md shrink-0">
              <div className="max-w-4xl mx-auto space-y-2">
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>
                    {activeStory.chapters.length === 0
                      ? "Inscribing Chapter 1"
                      : `Inscribing Chapter ${activeStory.chapters.length + 1}`}
                  </span>
                  <span>Enter ↵ to write chapter</span>
                </div>

                <div className="flex gap-2.5 items-end">
                  <div className="flex-1 bg-white rounded-xl border border-stone-300 p-2.5 focus-within:border-amber-600 focus-within:ring-2 focus-within:ring-amber-600/15 transition shadow-xs">
                    <textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleStreamNextChapter();
                        }
                      }}
                      placeholder={
                        activeStory.chapters.length === 0
                          ? `Type your Chapter 1 prompt from scratch for "${activeStory.title}"... (Press Enter to inscribe)`
                          : `Type what happens next in Chapter ${activeStory.chapters.length + 1}... (Press Enter to inscribe)`
                      }
                      disabled={isGenerating}
                      rows={2}
                      className="w-full text-xs font-serif text-slate-900 placeholder-slate-400 focus:outline-none resize-none bg-transparent leading-relaxed"
                    />
                  </div>

                  <button
                    onClick={handleStreamNextChapter}
                    disabled={isGenerating || !prompt.trim()}
                    className="px-5 py-3 rounded-xl bg-amber-600 hover:bg-amber-700 text-white font-medium text-xs flex items-center gap-2 shadow-sm transition disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer shrink-0"
                  >
                    {isGenerating ? (
                      <>
                        <Activity className="w-4 h-4 animate-spin text-white" />
                        <span>Inking...</span>
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4 text-white" />
                        <span>Inscribe Chapter</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </main>

      {/* ========================================================= */}
      {/* MODAL: NEW STORY SESSION (Clean from scratch)             */}
      {/* ========================================================= */}
      {isCreatingNewStory && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4 animate-in fade-in">
          <div className="bg-white rounded-2xl max-w-md w-full border border-stone-200 shadow-xl p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-stone-100 pb-3">
              <div className="flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-amber-600" />
                <h3 className="font-bold text-base text-slate-900 font-serif">
                  New Story Session
                </h3>
              </div>
              <button
                onClick={() => setIsCreatingNewStory(false)}
                className="text-slate-400 hover:text-slate-600 text-sm font-bold cursor-pointer"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateNewStoryModal} className="space-y-3.5 text-xs text-slate-800">
              <div>
                <label className="font-semibold block mb-1 text-slate-700">Story Title</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Chronicles of the Silver Spire"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-stone-300 focus:outline-none focus:border-amber-600 bg-white"
                />
              </div>

              <div>
                <label className="font-semibold block mb-1 text-slate-700">Genre / Theme</label>
                <select
                  value={newGenre}
                  onChange={(e) => setNewGenre(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-stone-300 focus:outline-none focus:border-amber-600 bg-white"
                >
                  <option value="Fantasy">Fantasy</option>
                  <option value="Sci-Fi">Sci-Fi</option>
                  <option value="Mystery">Mystery</option>
                  <option value="Mythology">Mythology</option>
                  <option value="Gothic Horror">Gothic Horror</option>
                  <option value="Adventure">Adventure</option>
                  <option value="Custom">Custom</option>
                </select>
              </div>

              <div>
                <label className="font-semibold block mb-1 text-slate-700">
                  Chapter 1 Opening Prompt (Optional)
                </label>
                <textarea
                  rows={3}
                  placeholder="Type your opening prompt from scratch (or inscribe it later inside the studio)..."
                  value={newInitialPrompt}
                  onChange={(e) => setNewInitialPrompt(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-stone-300 focus:outline-none focus:border-amber-600 resize-none font-serif text-xs leading-relaxed"
                />
              </div>

              <div className="pt-2 flex items-center justify-end gap-2 border-t border-stone-100">
                <button
                  type="button"
                  onClick={() => setIsCreatingNewStory(false)}
                  className="px-4 py-2 rounded-lg border border-stone-200 text-slate-600 hover:bg-stone-50 transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-medium transition cursor-pointer shadow-xs"
                >
                  Create Story
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
