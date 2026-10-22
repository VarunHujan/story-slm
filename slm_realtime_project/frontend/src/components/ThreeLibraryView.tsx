"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  BookOpen,
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Copy,
  Check,
  Trash2,
  Sparkles,
  Bot,
  User,
  Activity,
  Layers,
  X,
  Compass
} from "lucide-react";
import ThreeLibraryCorridor, { ThreeBookData, StoryChapter } from "./ThreeLibraryCorridor";

export const STORAGE_BOOKS_KEY = "karna_saved_books_v1";

interface ThreeLibraryViewProps {
  onNavigateToChat: () => void;
  onContinueInChat?: (messages: { id: string; sender: "user" | "slm"; text: string; timestamp: string }[]) => void;
  onOpenArchitecture?: () => void;
  profileId?: string;
}

export default function ThreeLibraryView({
  onNavigateToChat,
  onContinueInChat,
  onOpenArchitecture,
  profileId
}: ThreeLibraryViewProps) {
  // Compute user-specific storage key
  const userStorageKey = useMemo(() => {
    if (!profileId || profileId === "profile_varun") {
      return STORAGE_BOOKS_KEY;
    }
    return `karna_saved_books_${profileId}`;
  }, [profileId]);

  // Load saved books from localStorage
  const [books, setBooks] = useState<ThreeBookData[]>([]);
  const [selectedBook, setSelectedBook] = useState<ThreeBookData | null>(null);
  const [activeChapterIndex, setActiveChapterIndex] = useState<number>(0);
  const [copied, setCopied] = useState<boolean>(false);

  // Load books on mount & when profile changes
  useEffect(() => {
    try {
      const raw = localStorage.getItem(userStorageKey);
      if (raw) {
        const parsed: ThreeBookData[] = JSON.parse(raw);
        setBooks(parsed);
      } else {
        setBooks([]);
      }
    } catch (e) {
      console.error("Failed to load saved books:", e);
      setBooks([]);
    }
  }, [userStorageKey]);

  // Save books back whenever state changes
  const persistBooks = (updated: ThreeBookData[]) => {
    setBooks(updated);
    try {
      localStorage.setItem(userStorageKey, JSON.stringify(updated));
    } catch (e) {
      console.error("Failed to persist books:", e);
    }
  };

  // Delete a book
  const handleDeleteBook = (bookId: string) => {
    if (!confirm("Are you sure you want to remove this book from your 3D library shelf?")) return;
    const remaining = books.filter((b) => b.id !== bookId);
    // Re-index remaining books to fill shelf slots cleanly
    const reindexed: ThreeBookData[] = remaining.map((b, idx) => {
      if (idx < 12) {
        return {
          ...b,
          side: "left",
          tier: Math.floor(idx / 3) + 1,
          slot: (idx % 3) + 1
        };
      } else {
        const rIdx = idx - 12;
        return {
          ...b,
          side: "right",
          tier: Math.floor(rIdx / 3) + 1,
          slot: (rIdx % 3) + 1
        };
      }
    });
    persistBooks(reindexed);
    setSelectedBook(null);
  };

  // Handle Copy text
  const handleCopyChapter = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Current active chapter inside selected book
  const currentChapter: StoryChapter | null = useMemo(() => {
    if (!selectedBook || !selectedBook.chapters || selectedBook.chapters.length === 0) {
      return null;
    }
    return selectedBook.chapters[activeChapterIndex] || selectedBook.chapters[0];
  }, [selectedBook, activeChapterIndex]);

  return (
    <div className="p-4 sm:p-6 space-y-6 max-w-7xl mx-auto">
      {/* View Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white border border-stone-200 p-5 rounded-2xl shadow-xs">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h2 className="font-serif font-bold text-xl text-slate-900 tracking-tight flex items-center gap-2">
              <span className="text-amber-700">🏛️</span>
              <span>The 3D Grimoire Sanctuary</span>
            </h2>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-100 text-amber-900 border border-amber-300">
              {books.length} {books.length === 1 ? "Book" : "Books"} Stored
            </span>
          </div>
          <p className="text-xs text-slate-600 font-serif">
            A real-time 3D library corridor. Every book on these oak shelves is a story session you bound from the Chatbot.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onNavigateToChat}
            className="px-4 py-2 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-serif font-bold text-xs flex items-center gap-2 shadow-sm transition cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Open Chatbot ➔</span>
          </button>
        </div>
      </div>

      {/* Main 3D Corridor Canvas */}
      <div className="relative">
        <ThreeLibraryCorridor
          key={`three-corridor-${books.length}`}
          books={books}
          onSelectBook={(b) => {
            setSelectedBook(b);
            setActiveChapterIndex(0);
          }}
          onNavigateToChat={onNavigateToChat}
        />
      </div>

      {/* ========================================================= */}
      {/* BOOK READER MODAL / CHRONICLE READER VIEW                 */}
      {/* ========================================================= */}
      {selectedBook && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 animate-in fade-in">
          <div className="bg-[#faf7f2] border border-amber-900/30 rounded-2xl max-w-3xl w-full max-h-[90vh] flex flex-col shadow-2xl overflow-hidden animate-in zoom-in-95">
            {/* Header: Book Leather Spine Ribbon */}
            <div
              style={{ backgroundColor: selectedBook.leatherColor }}
              className="p-5 text-white flex items-center justify-between border-b border-white/20 shadow-sm"
            >
              <div className="flex items-center gap-3">
                <span className="text-3xl filter drop-shadow">{selectedBook.emblem}</span>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-serif font-bold text-lg tracking-tight text-amber-100">
                      {selectedBook.title}
                    </h3>
                    <span className="text-[10px] font-mono uppercase font-bold px-2 py-0.5 rounded bg-black/30 border border-white/20 text-amber-200">
                      {selectedBook.genre}
                    </span>
                  </div>
                  <p className="text-xs text-amber-200/80 font-serif italic">{selectedBook.subtitle}</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setSelectedBook(null)}
                  className="p-1.5 rounded-lg bg-black/20 hover:bg-black/40 text-white/80 hover:text-white transition cursor-pointer"
                  title="Close Reader"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Book Body: Parchment Page Display */}
            <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-6 bg-[#fcfbf9] text-slate-800 font-serif">
              {currentChapter ? (
                <div className="space-y-6">
                  {/* Chapter Header */}
                  <div className="flex items-center justify-between border-b border-amber-900/15 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold uppercase tracking-wider text-amber-900">
                        Chapter {activeChapterIndex + 1} of {selectedBook.chapters?.length || 1}
                      </span>
                    </div>

                    <button
                      onClick={() => handleCopyChapter(currentChapter.story)}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-lg border border-amber-900/15 text-xs text-amber-900 hover:bg-amber-100/50 transition cursor-pointer"
                    >
                      {copied ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-600" />
                          <span className="text-emerald-700 font-medium">Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5 text-amber-800" />
                          <span>Copy Story</span>
                        </>
                      )}
                    </button>
                  </div>

                  {/* Prompt Box */}
                  <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-900/15 space-y-1">
                    <div className="flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase text-amber-900">
                      <User className="w-3.5 h-3.5 text-amber-700" />
                      <span>The Chronicler's Seed (Your Prompt):</span>
                    </div>
                    <p className="text-xs text-slate-700 font-serif italic leading-relaxed">
                      "{currentChapter.prompt}"
                    </p>
                  </div>

                  {/* Story Text Box */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase text-amber-900">
                      <Bot className="w-3.5 h-3.5 text-amber-700" />
                      <span>Karna-49M Inscribed Verse:</span>
                    </div>
                    <div className="p-5 rounded-2xl bg-white border border-stone-200 shadow-xs text-sm text-slate-800 leading-relaxed font-serif whitespace-pre-wrap">
                      {currentChapter.story}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-500 font-serif">
                  <p>This book folio is clean parchment awaiting inscribed verses.</p>
                </div>
              )}
            </div>

            {/* Footer Navigation Bar */}
            <div className="p-4 bg-white border-t border-amber-900/15 flex flex-wrap items-center justify-between gap-3 text-xs">
              {/* Stepper controls */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActiveChapterIndex((prev) => Math.max(0, prev - 1))}
                  disabled={activeChapterIndex === 0}
                  className="px-3 py-1.5 rounded-lg border border-amber-900/20 text-slate-700 disabled:opacity-30 hover:bg-stone-50 transition cursor-pointer flex items-center gap-1"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span>Previous Chapter</span>
                </button>

                <span className="font-mono font-bold text-amber-900 px-2">
                  {activeChapterIndex + 1} / {selectedBook.chapters?.length || 1}
                </span>

                <button
                  onClick={() =>
                    setActiveChapterIndex((prev) =>
                      Math.min((selectedBook.chapters?.length || 1) - 1, prev + 1)
                    )
                  }
                  disabled={
                    !selectedBook.chapters ||
                    activeChapterIndex >= selectedBook.chapters.length - 1
                  }
                  className="px-3 py-1.5 rounded-lg border border-amber-900/20 text-slate-700 disabled:opacity-30 hover:bg-stone-50 transition cursor-pointer flex items-center gap-1"
                >
                  <span>Next Chapter</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              {/* Action buttons */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleDeleteBook(selectedBook.id)}
                  className="px-3 py-1.5 rounded-lg border border-rose-200 text-rose-700 hover:bg-rose-50 transition cursor-pointer flex items-center gap-1"
                  title="Delete this book from the shelf"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Delete Book</span>
                </button>

                <button
                  onClick={() => setSelectedBook(null)}
                  className="px-4 py-1.5 rounded-lg bg-stone-100 hover:bg-stone-200 text-slate-800 font-medium transition cursor-pointer"
                >
                  Back to Shelves
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
