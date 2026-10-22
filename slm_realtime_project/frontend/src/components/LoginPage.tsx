"use client";

import React, { useState, useEffect } from "react";
import {
  Sparkles,
  Key,
  BookOpen,
  Plus,
  Trash2,
  Check,
  Crown,
  ArrowRight,
  Feather,
  RotateCcw,
  Compass,
  Shield,
  Sun,
  X,
  Layers,
  Bot
} from "lucide-react";
import confetti from "canvas-confetti";
import { motion, AnimatePresence } from "framer-motion";

export interface ChroniclerProfile {
  id: string;
  name: string;
  title: string;
  house: string;
  emblem: string;
  leatherColor: string;
  createdAt: string;
  isGuest?: boolean;
}

export interface UserSession {
  username: string;
  title: string;
  emblem: string;
  house: string;
  loginTime: string;
  isGuest: boolean;
  profileId: string;
}

interface LoginPageProps {
  onLogin: (session: UserSession) => void;
}

const STORAGE_PROFILES_KEY = "karna_chronicler_profiles_v1";

// Default starting profile if none exist
const DEFAULT_INITIAL_PROFILES: ChroniclerProfile[] = [
  {
    id: "profile_varun",
    name: "Master Scribe Varun",
    title: "Guardian of the Solar Bow",
    house: "Suryaputra Dynasty",
    emblem: "☀️",
    leatherColor: "#6a1b24",
    createdAt: "Ancient Origin"
  }
];

export default function LoginPage({ onLogin }: LoginPageProps) {
  const [profiles, setProfiles] = useState<ChroniclerProfile[]>(() => {
    if (typeof window === "undefined") return DEFAULT_INITIAL_PROFILES;
    try {
      const raw = localStorage.getItem(STORAGE_PROFILES_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        return parsed.length > 0 ? parsed : DEFAULT_INITIAL_PROFILES;
      }
      return DEFAULT_INITIAL_PROFILES;
    } catch {
      return DEFAULT_INITIAL_PROFILES;
    }
  });

  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null);
  const [isCraftingNewTome, setIsCraftingNewTome] = useState(false);
  const [isOpeningTome, setIsOpeningTome] = useState(false);

  // New Tome Creation Form States
  const [newScribeName, setNewScribeName] = useState("");
  const [newTitle, setNewTitle] = useState("Chronicler of Epic Lore");
  const [newHouse, setNewHouse] = useState("Solar Scholars");
  const [newEmblem, setNewEmblem] = useState("☀️");
  const [newLeatherColor, setNewLeatherColor] = useState("#6a1b24");

  // Persist profiles to localStorage
  const persistProfiles = (updated: ChroniclerProfile[]) => {
    setProfiles(updated);
    try {
      localStorage.setItem(STORAGE_PROFILES_KEY, JSON.stringify(updated));
    } catch (e) {
      console.error(e);
    }
  };

  // Helper to count how many books this profile has in their library
  const getProfileBookCount = (profileId: string): number => {
    if (typeof window === "undefined") return 0;
    try {
      // Check user-specific library first, fallback to shared v1
      const userSpecific = localStorage.getItem(`karna_saved_books_${profileId}`);
      if (userSpecific) return JSON.parse(userSpecific).length;
      if (profileId === "profile_varun") {
        const legacy = localStorage.getItem("karna_saved_books_v1");
        if (legacy) return JSON.parse(legacy).length;
      }
      return 0;
    } catch {
      return 0;
    }
  };

  // 1. Pick Up Chronicler Tome & Enter
  const handleSelectTome = (profile: ChroniclerProfile) => {
    setSelectedProfileId(profile.id);
    setIsOpeningTome(true);

    try {
      confetti({
        particleCount: 60,
        spread: 60,
        origin: { y: 0.6 }
      });
    } catch {}

    setTimeout(() => {
      const session: UserSession = {
        username: profile.name,
        title: profile.title,
        emblem: profile.emblem,
        house: profile.house,
        loginTime: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isGuest: false,
        profileId: profile.id
      };
      onLogin(session);
    }, 700);
  };

  // 2. The Golden Guest Key (1-Click Instant Bypass)
  const handlePickUpGuestKey = () => {
    setSelectedProfileId("guest_key");
    setIsOpeningTome(true);

    try {
      confetti({
        particleCount: 50,
        spread: 50,
        origin: { y: 0.6 }
      });
    } catch {}

    setTimeout(() => {
      const session: UserSession = {
        username: "Guest Explorer",
        title: "Wanderer of the 3D Sanctuary",
        emblem: "🗝️",
        house: "Free Scribes Alliance",
        loginTime: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isGuest: true,
        profileId: "guest_explorer"
      };
      onLogin(session);
    }, 500);
  };

  // 3. Craft and Inscribe a Brand New Tome onto the Lectern
  const handleCraftTome = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newScribeName.trim()) return;

    const newProfile: ChroniclerProfile = {
      id: `profile_${Date.now()}`,
      name: newScribeName.trim(),
      title: newTitle.trim() || "Chronicler of the Realm",
      house: newHouse.trim() || "Sanctuary Scholars",
      emblem: newEmblem,
      leatherColor: newLeatherColor,
      createdAt: new Date().toLocaleDateString()
    };

    const updated = [...profiles, newProfile];
    persistProfiles(updated);
    setIsCraftingNewTome(false);

    // Auto-login with newly crafted tome
    handleSelectTome(newProfile);
  };

  // Delete a profile from the lectern
  const handleDeleteProfile = (e: React.MouseEvent, profileId: string) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to remove this Chronicler Tome from the Registry Lectern?")) return;
    const remaining = profiles.filter((p) => p.id !== profileId);
    persistProfiles(remaining);
  };

  return (
    <div className="min-h-screen bg-[#f7f4ed] text-slate-800 flex flex-col items-center justify-between p-4 sm:p-8 relative overflow-hidden select-none">
      {/* Background Sunlit Cathedral Rays & Dust Motes */}
      <div className="absolute inset-0 bg-radial from-amber-100/60 via-transparent to-transparent pointer-events-none" />
      <div className="absolute top-0 left-1/4 w-96 h-96 rounded-full bg-amber-200/30 blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 rounded-full bg-yellow-100/40 blur-3xl pointer-events-none" />

      {/* Top Header: The Registry of Scribes */}
      <div className="relative z-10 text-center space-y-2 pt-2 max-w-2xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-100/80 border border-amber-800/20 text-xs font-mono font-bold text-amber-900 tracking-wider uppercase">
          <BookOpen className="w-3.5 h-3.5 text-amber-700" />
          <span>The Grand Lectern of Chroniclers</span>
        </div>
        <h1 className="font-serif font-black text-3xl sm:text-4xl text-slate-900 tracking-tight">
          Choose Your Identity Tome
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 font-serif leading-relaxed max-w-lg mx-auto">
          In this sanctuary, each traveler is bound into a physical Grimoire. Pick up your personal Tome from the lectern to enter, or take the Guest Key for instant entry.
        </p>
      </div>

      {/* ========================================================= */}
      {/* THE REGISTRY LECTERN SHELF (BOOKS OF IDENTITIES)          */}
      {/* ========================================================= */}
      <div className="relative z-10 w-full max-w-5xl my-auto py-8">
        {/* The Heavy Wooden Lectern Desk Platform */}
        <div className="relative bg-[#faf7f2]/95 border border-amber-900/20 rounded-3xl p-6 sm:p-10 shadow-2xl backdrop-blur-md">
          {/* Top Brass Plaque on the Lectern */}
          <div className="absolute -top-3.5 left-1/2 -translate-x-1/2 px-6 py-1 rounded-full bg-gradient-to-r from-amber-700 via-amber-600 to-amber-700 text-white border border-amber-400 text-[10px] font-mono uppercase font-bold tracking-widest shadow-md">
            ✦ Registered Scribes &bull; Select to Open ✦
          </div>

          {/* Quick Guest Key Banner Sitting Atop the Lectern (1-Click Temporary Bypass) */}
          <div className="mb-8 p-4 rounded-2xl bg-gradient-to-r from-amber-50 via-yellow-50/80 to-amber-100/60 border border-amber-300 shadow-xs flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-amber-600 to-yellow-600 flex items-center justify-center text-white shadow-md shadow-amber-600/20">
                <Key className="w-6 h-6 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-serif font-bold text-sm text-slate-900">
                    The Golden Guest Key
                  </h3>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-200/70 text-amber-900 font-bold border border-amber-400">
                    1-Click Temporary Access
                  </span>
                </div>
                <p className="text-xs text-slate-600 font-serif">
                  Skip tome selection and enter the 3D sanctuary immediately as an honored Guest Chronicler.
                </p>
              </div>
            </div>

            <button
              onClick={handlePickUpGuestKey}
              disabled={isOpeningTome}
              className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-amber-700 hover:bg-amber-600 text-white font-serif font-bold text-xs flex items-center justify-center gap-2 shadow-md transition transform active:scale-95 cursor-pointer disabled:opacity-50"
            >
              <Key className="w-3.5 h-3.5" />
              <span>⚡ Take Guest Key & Enter ➔</span>
            </button>
          </div>

          {/* ========================================================= */}
          {/* THE PHYSICAL TOMES STANDING ON THE OAK SHELF              */}
          {/* ========================================================= */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 pt-2">
            {/* 1. Existing Chronicler Tomes */}
            {profiles.map((profile) => {
              const bookCount = getProfileBookCount(profile.id);
              const isSelected = selectedProfileId === profile.id;

              return (
                <div
                  key={profile.id}
                  onClick={() => handleSelectTome(profile)}
                  className={`group relative cursor-pointer transition-all duration-300 ${
                    isSelected ? "scale-105" : "hover:-translate-y-2"
                  }`}
                >
                  {/* The Physical Upright Standing Book Cover */}
                  <div
                    style={{ backgroundColor: profile.leatherColor }}
                    className="relative h-72 rounded-2xl p-5 flex flex-col justify-between text-white shadow-xl transition-all duration-300 border border-white/20 overflow-hidden group-hover:shadow-2xl group-hover:ring-2 group-hover:ring-amber-400"
                  >
                    {/* Spine Ribbon Line */}
                    <div className="absolute top-0 bottom-0 left-3 w-1 bg-gradient-to-r from-amber-400 to-amber-200 opacity-60" />
                    <div className="absolute top-0 bottom-0 left-5 w-px bg-white/20" />

                    {/* Corner Brass Ornaments */}
                    <div className="absolute top-2 right-2 w-3 h-3 border-t-2 border-r-2 border-amber-300/80 rounded-tr" />
                    <div className="absolute bottom-2 left-2 w-3 h-3 border-b-2 border-l-2 border-amber-300/80 rounded-bl" />

                    {/* Delete Tome Icon on Hover (except if only 1 profile) */}
                    {profiles.length > 1 && (
                      <button
                        onClick={(e) => handleDeleteProfile(e, profile.id)}
                        className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/40 hover:bg-rose-700 text-white/80 hover:text-white opacity-0 group-hover:opacity-100 transition z-20 cursor-pointer"
                        title="Remove Tome from Registry"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}

                    {/* Top: House Tag */}
                    <div className="pl-4">
                      <span className="text-[9px] font-mono uppercase font-bold text-amber-200 tracking-wider line-clamp-1">
                        {profile.house}
                      </span>
                    </div>

                    {/* Center: Heraldic Crest & Scribe Name */}
                    <div className="flex flex-col items-center text-center space-y-2 my-auto pl-2">
                      <div className="w-16 h-16 rounded-2xl bg-black/20 border border-white/20 flex items-center justify-center text-3xl filter drop-shadow shadow-inner group-hover:scale-110 transition">
                        <span>{profile.emblem}</span>
                      </div>
                      <div className="space-y-0.5">
                        <h3 className="font-serif font-bold text-base text-amber-100 group-hover:text-white leading-tight">
                          {profile.name}
                        </h3>
                        <p className="text-[10px] text-amber-200/80 font-serif italic line-clamp-1">
                          "{profile.title}"
                        </p>
                      </div>
                    </div>

                    {/* Bottom: Library Stored Books Pill */}
                    <div className="pl-4 pt-2 border-t border-white/15 flex items-center justify-between text-[10px] font-mono">
                      <span className="text-amber-200/80">
                        {bookCount > 0 ? `📜 ${bookCount} Books` : "✧ Fresh Tome"}
                      </span>
                      <span className="text-amber-300 font-bold flex items-center gap-1 group-hover:translate-x-1 transition">
                        Open ➔
                      </span>
                    </div>
                  </div>

                  {/* Polished Shadow Underneath Tome */}
                  <div className="h-2 w-4/5 mx-auto rounded-full bg-amber-950/20 blur-xs mt-1 transition group-hover:w-full group-hover:bg-amber-950/30" />
                </div>
              );
            })}

            {/* 2. Craft A New Scribe Tome Slot (The Blank Uninscribed Tome) */}
            <div
              onClick={() => setIsCraftingNewTome(true)}
              className="group relative cursor-pointer transition-all duration-300 hover:-translate-y-2"
            >
              <div className="h-72 rounded-2xl border-2 border-dashed border-amber-800/30 bg-amber-50/50 hover:bg-amber-100/60 p-5 flex flex-col items-center justify-center text-center space-y-3 shadow-sm hover:shadow-xl transition-all">
                <div className="w-16 h-16 rounded-2xl bg-amber-200/80 border border-amber-400 text-amber-900 flex items-center justify-center text-2xl shadow-sm group-hover:scale-110 transition">
                  <Plus className="w-7 h-7 text-amber-800" />
                </div>
                <div className="space-y-1">
                  <h3 className="font-serif font-bold text-sm text-slate-900">
                    Inscribe New Chronicler
                  </h3>
                  <p className="text-[11px] text-slate-500 font-serif">
                    Craft a personal leather grimoire to hold your own library books.
                  </p>
                </div>
                <span className="px-3 py-1 rounded-full bg-amber-200/60 border border-amber-300 text-[10px] font-mono text-amber-900 font-bold">
                  + Craft Tome
                </span>
              </div>
              <div className="h-2 w-4/5 mx-auto rounded-full bg-amber-950/10 blur-xs mt-1" />
            </div>
          </div>

          {/* Heavy Oak Shelf Plank at the bottom */}
          <div className="mt-6 h-3 rounded-full bg-gradient-to-r from-[#593b26] via-[#8a5a36] to-[#593b26] shadow-md border-t border-amber-300/30" />
        </div>
      </div>

      {/* ========================================================= */}
      {/* CRAFT NEW TOME MODAL (THE INSCRIBING DESK)                */}
      {/* ========================================================= */}
      {isCraftingNewTome && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 animate-in fade-in">
          <div className="bg-[#faf7f2] border border-amber-900/30 rounded-3xl p-6 sm:p-8 max-w-lg w-full shadow-2xl space-y-5 text-slate-900 animate-in zoom-in-95">
            <div className="flex items-center justify-between pb-3 border-b border-amber-900/15">
              <div className="flex items-center gap-2">
                <Feather className="w-5 h-5 text-amber-700" />
                <h3 className="font-serif text-lg font-bold text-slate-900">
                  Craft Your Chronicler Grimoire
                </h3>
              </div>
              <button
                onClick={() => setIsCraftingNewTome(false)}
                className="p-1 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-stone-200/50 transition cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCraftTome} className="space-y-4 text-xs">
              {/* Scribe Name */}
              <div className="space-y-1">
                <label className="text-slate-800 font-semibold block font-serif">
                  Chronicler Name / Persona:
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Master Scribe Arjuna, Lady of Eldoria, Neural Explorer"
                  value={newScribeName}
                  onChange={(e) => setNewScribeName(e.target.value)}
                  className="w-full bg-white border border-amber-900/20 rounded-xl p-2.5 text-sm font-serif text-slate-900 focus:outline-none focus:border-amber-600 shadow-2xs"
                />
              </div>

              {/* Title / Role */}
              <div className="space-y-1">
                <label className="text-slate-800 font-semibold block font-serif">
                  Title / Lore Role:
                </label>
                <input
                  type="text"
                  placeholder="e.g. Guardian of the Celestial Armor"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  className="w-full bg-white border border-amber-900/20 rounded-xl p-2.5 text-xs font-serif text-slate-900 focus:outline-none focus:border-amber-600 shadow-2xs"
                />
              </div>

              {/* Heraldic Emblem Picker */}
              <div className="space-y-1">
                <label className="text-slate-800 font-semibold block font-serif">
                  Embossed Gold Emblem:
                </label>
                <div className="flex gap-2 overflow-x-auto p-1.5 bg-stone-50 rounded-xl border border-stone-200">
                  {["☀️", "🏹", "👑", "🗝️", "⚡", "🌿", "🐾", "🌌", "👁️", "⚔️", "🏛️"].map((sym) => (
                    <button
                      key={sym}
                      type="button"
                      onClick={() => setNewEmblem(sym)}
                      className={`p-2 rounded-xl text-lg transition cursor-pointer ${
                        newEmblem === sym
                          ? "bg-amber-600 text-white scale-110 shadow-sm"
                          : "hover:bg-stone-200/60"
                      }`}
                    >
                      {sym}
                    </button>
                  ))}
                </div>
              </div>

              {/* Leather Binding Tone */}
              <div className="space-y-1">
                <label className="text-slate-800 font-semibold block font-serif">
                  Leather Binding Tone:
                </label>
                <div className="flex gap-3">
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
                      onClick={() => setNewLeatherColor(col.hex)}
                      style={{ backgroundColor: col.hex }}
                      className={`w-8 h-8 rounded-full border-2 transition cursor-pointer ${
                        newLeatherColor === col.hex
                          ? "border-amber-600 scale-110 shadow-md ring-2 ring-amber-300"
                          : "border-slate-300"
                      }`}
                      title={col.label}
                    />
                  ))}
                </div>
              </div>

              {/* Live Preview Box */}
              <div
                style={{ backgroundColor: newLeatherColor }}
                className="p-4 rounded-2xl text-white flex items-center gap-3 shadow-md border border-white/20"
              >
                <span className="text-3xl filter drop-shadow">{newEmblem}</span>
                <div>
                  <h4 className="font-serif font-bold text-sm text-amber-100">
                    {newScribeName || "Untitled Chronicler"}
                  </h4>
                  <p className="text-[11px] text-amber-200/80 font-serif italic">
                    "{newTitle}"
                  </p>
                </div>
              </div>

              {/* Submit Buttons */}
              <div className="pt-2 flex justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setIsCraftingNewTome(false)}
                  className="px-4 py-2 rounded-xl bg-stone-100 hover:bg-stone-200 text-slate-700 text-xs font-medium transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-serif font-bold text-xs flex items-center gap-2 shadow-md shadow-amber-600/20 transition cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Emboss Tome & Place on Lectern ➔</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Footer Technical Note */}
      <div className="relative z-10 text-center text-[11px] font-mono text-slate-400 pb-2">
        <span>Powered by Karna-49M &bull; Multi-Chronicler Registry &bull; 3D Three.js Architecture</span>
      </div>
    </div>
  );
}
