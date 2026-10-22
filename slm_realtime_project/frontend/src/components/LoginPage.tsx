"use client";

import React, { useState } from "react";
import {
  Sparkles,
  Bot,
  Zap,
  BookOpen,
  ArrowRight,
  Shield,
  Sun,
  Feather,
  Compass,
  Check,
  Cpu,
  Layers,
  Crown
} from "lucide-react";
import confetti from "canvas-confetti";

export interface UserSession {
  username: string;
  title: string;
  emblem: string;
  house: string;
  loginTime: string;
  isGuest: boolean;
}

interface LoginPageProps {
  onLogin: (session: UserSession) => void;
}

// Preset Chronicler Archetypes for creative selection
const CHRONICLER_HOUSES = [
  {
    id: "surya",
    name: "Suryaputra Scribes",
    title: "Guardian of the Golden Armor",
    emblem: "☀️",
    color: "#6a1b24",
    tagline: "Devoted to solar valor, honor, and unyielding vows."
  },
  {
    id: "vijaya",
    name: "Bow of Vijaya",
    title: "Master of Celestial Archery",
    emblem: "🏹",
    color: "#1e4d2b",
    tagline: "Weaving tales of impossible focus and mythical weapons."
  },
  {
    id: "neural",
    name: "Neural Archmages",
    title: "Master of GQA & 49M Weights",
    emblem: "🧠",
    color: "#24334f",
    tagline: "Exploring deep transformer geometry and causal token flow."
  },
  {
    id: "astral",
    name: "Astral Lorekeepers",
    title: "Wanderer of Forgotten Realms",
    emblem: "🌌",
    color: "#45225c",
    tagline: "Uncovering cosmic odysseys across distant timelines."
  }
];

export default function LoginPage({ onLogin }: LoginPageProps) {
  const [selectedHouse, setSelectedHouse] = useState(CHRONICLER_HOUSES[0]);
  const [customName, setCustomName] = useState("");
  const [customEmblem, setCustomEmblem] = useState("☀️");
  const [isSealing, setIsSealing] = useState(false);

  // Trigger Login Animation & Submit
  const handleInscribeLogin = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsSealing(true);

    try {
      confetti({
        particleCount: 70,
        spread: 60,
        origin: { y: 0.65 }
      });
    } catch {}

    setTimeout(() => {
      const finalName = customName.trim() || selectedHouse.title;
      const session: UserSession = {
        username: finalName,
        title: selectedHouse.title,
        emblem: customEmblem || selectedHouse.emblem,
        house: selectedHouse.name,
        loginTime: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isGuest: false
      };
      onLogin(session);
    }, 600);
  };

  // 1-Click Temporary / Quick Bypass Login
  const handleQuickGuestLogin = () => {
    setIsSealing(true);
    try {
      confetti({
        particleCount: 50,
        spread: 50,
        origin: { y: 0.65 }
      });
    } catch {}

    setTimeout(() => {
      const session: UserSession = {
        username: "Guest Chronicler",
        title: "Wandering Scribe of Karna",
        emblem: "✨",
        house: "Free Scribes Alliance",
        loginTime: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isGuest: true
      };
      onLogin(session);
    }, 450);
  };

  return (
    <div className="min-h-screen bg-[#faf8f5] text-slate-800 flex flex-col items-center justify-center p-4 sm:p-6 relative overflow-hidden selection:bg-amber-200">
      {/* Background Decorative Temple Halo Rings */}
      <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-amber-100/60 blur-3xl pointer-events-none" />
      <div className="absolute -bottom-40 -right-40 w-96 h-96 rounded-full bg-amber-200/50 blur-3xl pointer-events-none" />

      {/* Main Celestial Login Card */}
      <div className="relative z-10 w-full max-w-xl bg-white/95 border border-amber-900/20 rounded-3xl p-6 sm:p-10 shadow-2xl backdrop-blur-md space-y-7 animate-in fade-in zoom-in-95 duration-500">
        {/* Top Arcane Crest & Portal Header */}
        <div className="text-center space-y-3">
          <div className="relative inline-block">
            {/* Spinning Golden Seal Ring */}
            <div className="w-20 h-20 mx-auto rounded-full bg-gradient-to-tr from-amber-600 via-amber-500 to-yellow-500 p-1 shadow-lg shadow-amber-600/30 flex items-center justify-center">
              <div className="w-full h-full rounded-full bg-amber-50 border-2 border-amber-200 flex items-center justify-center text-3xl filter drop-shadow">
                <span>{customEmblem}</span>
              </div>
            </div>
            {/* Pulsing Aura Badge */}
            <span className="absolute -bottom-1 -right-1 p-1 rounded-full bg-amber-600 text-white text-[10px] shadow-sm">
              <Crown className="w-3.5 h-3.5" />
            </span>
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-center gap-2">
              <h1 className="font-serif font-black text-2xl sm:text-3xl text-slate-900 tracking-tight">
                Sanctuary of Karna
              </h1>
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-800/20">
                49M SLM
              </span>
            </div>
            <p className="text-xs text-slate-600 font-serif max-w-md mx-auto leading-relaxed">
              Inscribe your chronicler seal to cross the threshold into the <strong>Karna-49M</strong> Neural Universe and 3D Grimoire Sanctuary.
            </p>
          </div>
        </div>

        {/* ========================================================= */}
        {/* 1-CLICK TEMPORARY QUICK LOGIN BANNER (USER REQUEST)       */}
        {/* ========================================================= */}
        <div className="p-3.5 rounded-2xl bg-gradient-to-r from-amber-50 via-amber-100/70 to-yellow-50 border border-amber-300/80 shadow-xs flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 text-left">
            <span className="p-2 rounded-xl bg-amber-600 text-white shadow-xs">
              <Zap className="w-4 h-4" />
            </span>
            <div>
              <span className="text-xs font-serif font-bold text-slate-900 block">
                Instant Guest Entry
              </span>
              <span className="text-[11px] text-slate-600 font-serif">
                Skip configuration and explore immediately.
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={handleQuickGuestLogin}
            disabled={isSealing}
            className="w-full sm:w-auto px-4 py-2 rounded-xl bg-amber-700 hover:bg-amber-600 text-white font-serif font-bold text-xs flex items-center justify-center gap-1.5 shadow-sm transition active:scale-95 cursor-pointer disabled:opacity-50"
          >
            <span>⚡ Quick Enter</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Divider */}
        <div className="flex items-center gap-3">
          <div className="flex-1 h-px bg-stone-200" />
          <span className="text-[10px] font-mono uppercase font-bold text-slate-400">
            Or Inscribe Your Own Seal
          </span>
          <div className="flex-1 h-px bg-stone-200" />
        </div>

        {/* Interactive Inscription Form */}
        <form onSubmit={handleInscribeLogin} className="space-y-5 text-xs">
          {/* Step 1: Select Chronicler House / Archetype */}
          <div className="space-y-2">
            <label className="text-slate-800 font-semibold block font-serif flex items-center justify-between">
              <span>1. Choose Your Chronicler House:</span>
              <span className="text-[10px] font-mono text-amber-800 font-normal">
                {selectedHouse.name}
              </span>
            </label>

            <div className="grid grid-cols-2 gap-2.5">
              {CHRONICLER_HOUSES.map((house) => {
                const isSelected = selectedHouse.id === house.id;
                return (
                  <button
                    key={house.id}
                    type="button"
                    onClick={() => {
                      setSelectedHouse(house);
                      setCustomEmblem(house.emblem);
                    }}
                    className={`p-3 rounded-2xl border text-left transition relative cursor-pointer ${
                      isSelected
                        ? "bg-amber-50/90 border-amber-600 shadow-sm ring-1 ring-amber-500"
                        : "bg-stone-50 border-stone-200 hover:border-amber-300"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xl filter drop-shadow">{house.emblem}</span>
                      {isSelected && (
                        <Check className="w-3.5 h-3.5 text-amber-700 font-bold" />
                      )}
                    </div>
                    <span className="font-serif font-bold text-slate-900 block text-xs truncate">
                      {house.name}
                    </span>
                    <span className="text-[10px] text-slate-500 font-serif line-clamp-1">
                      {house.title}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Step 2: Custom Name & Emblem */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="sm:col-span-2 space-y-1">
              <label className="text-slate-800 font-semibold block font-serif">
                2. Chronicler Name / Handle:
              </label>
              <input
                type="text"
                placeholder={selectedHouse.title}
                value={customName}
                onChange={(e) => setCustomName(e.target.value)}
                className="w-full bg-white border border-amber-900/20 rounded-xl px-3.5 py-2.5 text-xs text-slate-900 font-serif focus:outline-none focus:border-amber-600 shadow-2xs"
              />
            </div>

            <div className="space-y-1">
              <label className="text-slate-800 font-semibold block font-serif">
                Seal Emblem:
              </label>
              <div className="flex gap-1 overflow-x-auto p-1 bg-stone-50 rounded-xl border border-stone-200">
                {["☀️", "🏹", "👑", "🗝️", "⚡", "🌿", "🐾", "🌌", "👁️", "⚔️"].map((sym) => (
                  <button
                    key={sym}
                    type="button"
                    onClick={() => setCustomEmblem(sym)}
                    className={`p-1.5 rounded-lg text-sm transition cursor-pointer ${
                      customEmblem === sym
                        ? "bg-amber-600 text-white scale-110 shadow-xs"
                        : "hover:bg-stone-200/60"
                    }`}
                  >
                    {sym}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Golden Wax Seal Submit Action Button */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={isSealing}
              className={`w-full py-3.5 px-6 rounded-2xl bg-gradient-to-r from-amber-600 via-amber-700 to-yellow-600 hover:from-amber-500 hover:to-amber-600 text-white font-serif font-bold text-sm flex items-center justify-center gap-2 shadow-lg shadow-amber-700/25 transition transform active:scale-98 cursor-pointer disabled:opacity-50 ${
                isSealing ? "animate-pulse" : ""
              }`}
            >
              <Feather className="w-4 h-4" />
              <span>
                {isSealing ? "Pressing Golden Wax Seal..." : "Inscribe Seal & Enter Gateway ➔"}
              </span>
            </button>
          </div>
        </form>

        {/* Bottom Technical Telemetry Badge */}
        <div className="pt-3 border-t border-stone-100 flex flex-wrap items-center justify-between text-[10px] font-mono text-slate-400">
          <span>Model: Karna-49M v2 (SFT Checkpoint)</span>
          <span>GQA 8:4 &bull; RoPE &bull; SwiGLU &bull; 12 Layers</span>
        </div>
      </div>
    </div>
  );
}
