"use client";

import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  BookOpen,
  Feather,
  Sparkles,
  ArrowLeft,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  Copy,
  Check,
  Sliders,
  Zap,
  Activity,
  Layers,
  Cpu,
  Compass,
  Bookmark,
  ChevronDown,
  Dices,
  Flame,
  Shield,
  Scroll,
  Download,
  Share2,
  Eye,
  History,
  Wand2,
  Sword,
  HeartCrack,
  Moon,
  Sun,
  Plus,
  Trash2,
  X
} from "lucide-react";

import ThreeLibraryCorridor, { ThreeBookData } from "./ThreeLibraryCorridor";
import confetti from "canvas-confetti";
import { motion, AnimatePresence } from "framer-motion";

export interface ChatTurn {
  id: string;
  role: "chronicler" | "oracle";
  text: string;
  timestamp: number;
}

export interface ChapterData {
  chapterNum: number;
  userPrompt: string;
  oracleResponse?: string;
  timestamp?: number;
}

export const toRoman = (num: number): string => {
  const romanMap: [number, string][] = [
    [1000, "M"],
    [900, "CM"],
    [500, "D"],
    [400, "CD"],
    [100, "C"],
    [90, "XC"],
    [50, "L"],
    [40, "XL"],
    [10, "X"],
    [9, "IX"],
    [5, "V"],
    [4, "IV"],
    [1, "I"],
  ];
  let res = "";
  let n = Math.max(1, Math.floor(num));
  for (const [val, sym] of romanMap) {
    while (n >= val) {
      res += sym;
      n -= val;
    }
  }
  return res || "I";
};

const STORAGE_CUSTOM_BOOKS_KEY = "karna_custom_books_v1";
const STORAGE_SESSIONS_KEY = "karna_book_chat_sessions_v1";
const STORAGE_SEALED_BOOKS_KEY = "karna_book_sealed_state_v1";

export interface BookDef {
  id: string;
  title: string;
  subtitle: string;
  tag: string;
  genre: string;
  tone: string;
  bgHex: string;
  foilHex: string;
  auraColor: string;
  sealEmblem: string;
  isCustom?: boolean;
  presets: { title: string; prompt: string }[];
  fateArchetypes: string[];
  fateCatalysts: string[];
  fateSettings: string[];
}

export const DEFAULT_GRIMOIRES: BookDef[] = [
  {
    id: "karna",
    title: "The Legend of Karna",
    subtitle: "Suryaputra & The Golden Armor",
    tag: "MAHABHARATA & SOLAR LORE",
    genre: "Mythology / Epic",
    tone: "wine",
    bgHex: "#481219",
    foilHex: "#ecd38c",
    auraColor: "rgba(245, 158, 11, 0.25)",
    sealEmblem: "☀️",
    presets: [
      {
        title: "Arena Infiltration",
        prompt: "Write a dramatic scene of Prince Karna entering the Hastinapur tournament arena with radiant kavacha and supreme confidence."
      },
      {
        title: "Surya's Dawn Blessing",
        prompt: "Describe Lord Surya appearing at dawn to bless Karna with unbreakable golden earrings and celestial armor."
      },
      {
        title: "Dialogue of Dharma",
        prompt: "Write a philosophical dialogue between Krishna and Karna on loyalty, destiny, and the burden of righteousness."
      }
    ],
    fateArchetypes: ["The Solar Archer", "The Banished Prince", "The Loyal General", "The Solar Deity"],
    fateCatalysts: ["A challenge at the tournament of kings", "Donating celestial armor at dawn", "A sacred vow that cannot be broken"],
    fateSettings: ["Under the scorching midday sun of Hastinapur", "On the sacred banks of the Ganga", "Before the assembled Kuru elders"]
  },
  {
    id: "eldoria",
    title: "Chronicles of Eldoria",
    subtitle: "High Fantasy & Ancient Magic",
    tag: "HIGH FANTASY & RUNES",
    genre: "High Fantasy",
    tone: "forest",
    bgHex: "#142f1d",
    foilHex: "#e3d191",
    auraColor: "rgba(16, 185, 129, 0.25)",
    sealEmblem: "🌿",
    presets: [
      {
        title: "The Golden Key",
        prompt: "Tell a story about a brave girl named Lily who found an ancient golden key in the whispering woods."
      },
      {
        title: "The Crystal Caverns",
        prompt: "A young sorceress awakens an ancient crystalline guardian beneath the misty mountains of Eldoria."
      }
    ],
    fateArchetypes: ["The Runesmith's Apprentice", "The Slumbering Dragon", "The Forest Ranger", "The Star Weaver"],
    fateCatalysts: ["Unlocking a door sealed for ten centuries", "Finding a pulsing dragon egg", "A comet landing in the enchanted glade"],
    fateSettings: ["Deep inside the canopy of Whispering Pines", "Beside the moonlit silver lake of Eldoria", "Within the ruins of the Sunken Spire"]
  },
  {
    id: "cyberpunk",
    title: "Cyberpunk 2099",
    subtitle: "Neon Syndicate & Neural Outlaws",
    tag: "SYNTHETIC NOIR & CYBERNETICS",
    genre: "Sci-Fi / Cyberpunk",
    tone: "teal",
    bgHex: "#103337",
    foilHex: "#6ee7b7",
    auraColor: "rgba(6, 182, 212, 0.25)",
    sealEmblem: "⚡",
    presets: [
      {
        title: "Mainframe Breach",
        prompt: "Detective Vex hacks into the orbital mainframe of Neo-Veridia as sirens pierce the rainy night."
      },
      {
        title: "Sentient Awakening",
        prompt: "An obsolete maintenance android awakens in the scrapheaps, discovering fragmented human memories."
      }
    ],
    fateArchetypes: ["The Rogue Netrunner", "The Disillusioned Cyber-Cop", "The Sentient Machine", "The Corporate Fixer"],
    fateCatalysts: ["A data chip containing stolen memories", "A city-wide neural blackout", "An AI claiming to possess an organic soul"],
    fateSettings: ["In the neon-drenched alleys of Lower Sector 4", "Atop the holographic spire of Omnicorp", "Inside a clandestine underground cyber-clinic"]
  },
  {
    id: "shadows",
    title: "The Whispering Shadows",
    subtitle: "Gothic Mysteries & Dark Manors",
    tag: "GOTHIC HORROR & SUSPENSE",
    genre: "Gothic Mystery",
    tone: "blue",
    bgHex: "#141f33",
    foilHex: "#c7d2fe",
    auraColor: "rgba(99, 102, 241, 0.25)",
    sealEmblem: "🗝️",
    presets: [
      {
        title: "Thirteenth Chime",
        prompt: "At midnight, the grandfather clock in the abandoned Blackwood Manor struck thirteen times."
      },
      {
        title: "The Shifting Portrait",
        prompt: "A traveling scholar notices that the ancestral portrait in the library changes expression each night."
      }
    ],
    fateArchetypes: ["The Solitary Heir", "The Whispering Specter", "The Midnight Archivist", "The Reluctant Medium"],
    fateCatalysts: ["An unsealed letter written in fading red ink", "The thirteenth chime of the clock tower", "A hidden staircase behind the fireplace"],
    fateSettings: ["Behind the fog-covered iron gates of Blackwood", "In the candlelit library smelling of old leather", "Beneath the vaulted cellar during a thunderstorm"]
  },
  {
    id: "woodland",
    title: "Tales of Whispering Woods",
    subtitle: "Playful Beasts & Childhood Fables",
    tag: "FABLES & TINY STORIES",
    genre: "Fables",
    tone: "ochre",
    bgHex: "#573714",
    foilHex: "#fde68a",
    auraColor: "rgba(245, 158, 11, 0.22)",
    sealEmblem: "🐾",
    presets: [
      {
        title: "Pip's Sunny Spot",
        prompt: "Pip the clever cat woke up early to find the sunniest patch of dirt in the enchanted garden."
      },
      {
        title: "The Lost Cloud Ball",
        prompt: "Max the playful puppy chased his magical bouncing ball straight past the tall oak tree into the sky."
      }
    ],
    fateArchetypes: ["The Inquisitive Kitten", "The Wise Old Badger", "The Shy Little Sparrow", "The Brave Field Mouse"],
    fateCatalysts: ["Finding a mysterious glowing acorn", "A gust of wind blowing away a favorite treasure", "A gentle stream that suddenly speaks"],
    fateSettings: ["Under the golden canopy of the Great Apple Tree", "Beside the pebble-filled whispering brook", "Inside a cozy mossy hollow on a sunny morning"]
  },
  {
    id: "ashoka",
    title: "The Empire of Ashoka",
    subtitle: "Edicts of Stone & Mauryan Crowns",
    tag: "HISTORICAL BIOGRAPHY",
    genre: "Historical",
    tone: "sage",
    bgHex: "#2a372c",
    foilHex: "#d1d5db",
    auraColor: "rgba(156, 163, 175, 0.22)",
    sealEmblem: "🏛️",
    presets: [
      {
        title: "The Banks of Daya",
        prompt: "Emperor Ashoka surveys the aftermath of the Kalinga battlefield and vows to embrace dharma and peace."
      },
      {
        title: "The Lion Capital",
        prompt: "Master sculptors chisel the four lions upon the polished sandstone pillar under the blazing sun."
      }
    ],
    fateArchetypes: ["The Conqueror Turned Monk", "The Royal Epigrapher", "The Wandering Buddhist Sage", "The Devoted Empress"],
    fateCatalysts: ["Carving the fourteenth rock edict", "Planting banyan trees along the royal highway", "Renouncing the sword for the wheel of dharma"],
    fateSettings: ["Upon the sandstone cliffs of Dhauli", "Within the sprawling palace of Pataliputra", "Under the sacred Bodhi tree at Gaya"]
  }
];

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

interface SpellbookSanctuaryProps {
  onOpenArchitecture?: () => void;
  onOpenPipeline?: () => void;
}

export default function SpellbookSanctuary({
  onOpenArchitecture,
  onOpenPipeline
}: SpellbookSanctuaryProps) {
  // Grimoire Library State (Stored in LocalStorage + Default Books)
  const [books, setBooks] = useState<BookDef[]>(DEFAULT_GRIMOIRES);
  const [selectedBook, setSelectedBook] = useState<BookDef | null>(null);
  const [libraryMode, setLibraryMode] = useState<"3d_corridor" | "shelves">("3d_corridor");

  // Persistent Chat Sessions (Book ID -> ChatTurn[])
  const [sessions, setSessions] = useState<Record<string, ChatTurn[]>>(() => {
    if (typeof window === "undefined") return {};
    try {
      const raw = localStorage.getItem(STORAGE_SESSIONS_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  });

  // Persistent Sealed Books State
  const [sealedBooks, setSealedBooks] = useState<Record<string, boolean>>(() => {
    if (typeof window === "undefined") return {};
    try {
      const raw = localStorage.getItem(STORAGE_SEALED_BOOKS_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  });

  // Active Streaming Response for current inking turn
  const [activeStreamingVerse, setActiveStreamingVerse] = useState<string>("");

  // Derived Sessions Summary for 3D Corridor and 2D Shelves
  const sessionsSummary = useMemo(() => {
    const summary: Record<string, { messageCount: number; lastSnippet?: string }> = {};
    for (const [bId, turns] of Object.entries(sessions)) {
      if (turns && turns.length > 0) {
        summary[bId] = {
          messageCount: turns.length,
          lastSnippet: turns[turns.length - 1]?.text.slice(0, 50)
        };
      }
    }
    return summary;
  }, [sessions]);

  // Active book turns
  const activeTurns = useMemo(() => {
    return selectedBook ? (sessions[selectedBook.id] || []) : [];
  }, [selectedBook, sessions]);

  // Derived Chapter Pairings for Multi-Turn Pagination
  const chapters = useMemo<ChapterData[]>(() => {
    const result: ChapterData[] = [];
    let currentPrompt = "";
    let currentTimestamp = 0;

    for (const turn of activeTurns) {
      if (turn.role === "chronicler") {
        if (currentPrompt) {
          result.push({
            chapterNum: result.length + 1,
            userPrompt: currentPrompt,
            timestamp: currentTimestamp
          });
        }
        currentPrompt = turn.text;
        currentTimestamp = turn.timestamp;
      } else if (turn.role === "oracle") {
        result.push({
          chapterNum: result.length + 1,
          userPrompt: currentPrompt || "Opening Inscription",
          oracleResponse: turn.text,
          timestamp: turn.timestamp || currentTimestamp
        });
        currentPrompt = "";
      }
    }

    if (currentPrompt) {
      result.push({
        chapterNum: result.length + 1,
        userPrompt: currentPrompt,
        timestamp: currentTimestamp
      });
    }

    return result;
  }, [activeTurns]);

  // Total folios: completed chapters + 1 blank page for next inscription
  const totalFolios = useMemo(() => {
    if (chapters.length === 0) return 1;
    const lastChapter = chapters[chapters.length - 1];
    return lastChapter?.oracleResponse ? chapters.length + 1 : chapters.length;
  }, [chapters]);

  const handleSelectFromThree = useCallback((threeBook: ThreeBookData) => {
    setBooks((prevBooks) => {
      const found = prevBooks.find((b) => b.id === threeBook.id || b.title === threeBook.title);
      if (found) {
        setSelectedBook(found);
      } else {
        const newBookDef: BookDef = {
          id: threeBook.id,
          title: threeBook.title,
          subtitle: threeBook.subtitle,
          tag: threeBook.genre.toUpperCase(),
          genre: threeBook.genre,
          tone: "custom",
          bgHex: threeBook.leatherColor,
          foilHex: "#ecd38c",
          auraColor: `${threeBook.leatherColor}66`,
          sealEmblem: threeBook.emblem,
          presets: [
            {
              title: "Chapter I Seed",
              prompt: `Write the opening chapter of ${threeBook.title}, setting the atmosphere and introducing the protagonist.`
            }
          ],
          fateArchetypes: ["The Reluctant Hero", "The Master of Runes", "The Wanderer", "The Silent Guardian"],
          fateCatalysts: ["Finding an ancient relic in the ruins", "A prophecy whispered by the stars", "A challenge at dawn"],
          fateSettings: ["Under the golden sky of an ancient kingdom", "Deep within the candlelit library vaults", "Before the assembled council"]
        };
        setSelectedBook(newBookDef);
        return [...prevBooks, newBookDef];
      }
      return prevBooks;
    });
    setPrompt("");
    setActiveStreamingVerse("");
  }, []);

  // Sync sealed status & default prompt & folio when book changes
  useEffect(() => {
    if (selectedBook) {
      setIsSealed(Boolean(sealedBooks[selectedBook.id]));
      const bookTurns = sessions[selectedBook.id] || [];
      if (!bookTurns || bookTurns.length === 0) {
        setPrompt(selectedBook.presets[0]?.prompt || `Write an epic tale of ${selectedBook.title}.`);
        setViewingChapter(1);
      } else {
        setPrompt("");
        const oracleCount = bookTurns.filter((t) => t.role === "oracle").length;
        setViewingChapter(Math.max(1, oracleCount));
      }
    }
  }, [selectedBook?.id]);

  // New Book Modal Form States
  const [isForgingNewBook, setIsForgingNewBook] = useState<boolean>(false);
  const [newTitle, setNewTitle] = useState<string>("");
  const [newSubtitle, setNewSubtitle] = useState<string>("");
  const [newGenre, setNewGenre] = useState<string>("High Fantasy");
  const [newToneColor, setNewToneColor] = useState<string>("#481219");
  const [newFoilColor, setNewFoilColor] = useState<string>("#ecd38c");
  const [newEmblem, setNewEmblem] = useState<string>("⚔️");
  const [newStartingPrompt, setNewStartingPrompt] = useState<string>("");

  // Story & Prompt States
  const [prompt, setPrompt] = useState<string>("");
  const [isCasting, setIsCasting] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [viewingChapter, setViewingChapter] = useState<number>(1);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [showMarginalia, setShowMarginalia] = useState<boolean>(true);
  const [isSealed, setIsSealed] = useState<boolean>(false);

  // Fate Dice Roller States
  const [isRollingDice, setIsRollingDice] = useState<boolean>(false);
  const [rolledArchetype, setRolledArchetype] = useState<string>("");
  const [rolledCatalyst, setRolledCatalyst] = useState<string>("");
  const [rolledSetting, setRolledSetting] = useState<string>("");

  // Hyperparameters
  const [temperature, setTemperature] = useState<number>(0.75);
  const [maxTokens, setMaxTokens] = useState<number>(350);
  const [topK, setTopK] = useState<number>(40);

  // Telemetry
  const [tokensPerSec, setTokensPerSec] = useState<number>(0);
  const [totalTokensGenerated, setTotalTokensGenerated] = useState<number>(0);
  const [currentStreamingToken, setCurrentStreamingToken] = useState<string>("");

  const leftPageScrollRef = useRef<HTMLDivElement>(null);
  const rightPageScrollRef = useRef<HTMLDivElement>(null);

  // Load custom books from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_CUSTOM_BOOKS_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          const merged = [...DEFAULT_GRIMOIRES];
          parsed.forEach((customB: BookDef) => {
            if (!merged.some((b) => b.id === customB.id)) {
              merged.push(customB);
            }
          });
          setBooks(merged);
        }
      }
    } catch (e) {
      console.error("Error reading saved books from storage", e);
    }
  }, []);

  // Auto-scroll manuscript while streaming
  useEffect(() => {
    if (isCasting) {
      rightPageScrollRef.current?.scrollTo({
        top: rightPageScrollRef.current.scrollHeight,
        behavior: "smooth"
      });
      leftPageScrollRef.current?.scrollTo({
        top: leftPageScrollRef.current.scrollHeight,
        behavior: "smooth"
      });
    }
  }, [activeStreamingVerse, isCasting]);

  // Roll the Fate Dice
  const handleRollFateDice = () => {
    if (!selectedBook || isRollingDice) return;
    setIsRollingDice(true);

    let rolls = 0;
    const archetypes = selectedBook.fateArchetypes?.length ? selectedBook.fateArchetypes : ["The Protagonist", "The Rebel", "The Seeker"];
    const catalysts = selectedBook.fateCatalysts?.length ? selectedBook.fateCatalysts : ["A mysterious awakening", "A forgotten betrayal", "An urgent quest"];
    const settings = selectedBook.fateSettings?.length ? selectedBook.fateSettings : ["In an ancient realm", "Beneath starlit skies", "Across silent lands"];

    const interval = setInterval(() => {
      const a = archetypes[Math.floor(Math.random() * archetypes.length)];
      const c = catalysts[Math.floor(Math.random() * catalysts.length)];
      const s = settings[Math.floor(Math.random() * settings.length)];

      setRolledArchetype(a);
      setRolledCatalyst(c);
      setRolledSetting(s);

      rolls++;
      if (rolls >= 8) {
        clearInterval(interval);
        setIsRollingDice(false);
        setPrompt(`Inscribe the next turn: Featuring ${a}, beginning with ${c.toLowerCase()}, set ${s.toLowerCase()}.`);

        try {
          confetti({
            particleCount: 22,
            spread: 55,
            origin: { x: 0.32, y: 0.62 },
            colors: ["#ffd700", "#f59e0b", "#d97706", "#fef08a"],
            shapes: ["star"],
            ticks: 120,
            scalar: 0.85
          });
        } catch {
          // confetti fail-safe
        }
      }
    }, 100);
  };

  // Affix Royal Wax Seal with Imperial Gold Leaf Confetti
  const handleAffixWaxSeal = (e?: React.MouseEvent) => {
    if (!selectedBook) return;
    const nextSealed = !isSealed;
    setIsSealed(nextSealed);

    const updatedSeals = { ...sealedBooks, [selectedBook.id]: nextSealed };
    setSealedBooks(updatedSeals);
    try {
      localStorage.setItem(STORAGE_SEALED_BOOKS_KEY, JSON.stringify(updatedSeals));
    } catch {}

    if (nextSealed) {
      let originX = 0.5;
      let originY = 0.5;
      if (e) {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        originX = (rect.left + rect.width / 2) / window.innerWidth;
        originY = (rect.top + rect.height / 2) / window.innerHeight;
      }

      try {
        confetti({
          particleCount: 65,
          spread: 75,
          origin: { x: originX, y: originY },
          colors: ["#d4af37", "#fef08a", "#b45309", "#991b1b", "#7f1d1d"],
          shapes: ["circle", "square"],
          scalar: 0.95,
          gravity: 0.8,
          ticks: 160
        });

        setTimeout(() => {
          confetti({
            particleCount: 28,
            spread: 90,
            origin: { x: originX, y: originY },
            colors: ["#ffd700", "#ffffff", "#fef9c3"],
            shapes: ["star"],
            scalar: 1.15,
            gravity: 0.55,
            ticks: 200
          });
        }, 100);
      } catch {}
    }
  };

  // Reset or Clear this Book's Folio
  const handleClearFolio = () => {
    if (!selectedBook) return;
    const newSessions = { ...sessions };
    delete newSessions[selectedBook.id];
    setSessions(newSessions);
    try {
      localStorage.setItem(STORAGE_SESSIONS_KEY, JSON.stringify(newSessions));
    } catch {}
    setPrompt(selectedBook.presets[0]?.prompt || `Write a chronicle of ${selectedBook.title}.`);
    setActiveStreamingVerse("");
    setViewingChapter(1);
  };

  // Forge a new custom grimoire (New Chat Session Book)
  const handleForgeBook = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) return;

    const newBookId = "tome_" + Date.now();
    const createdBook: BookDef = {
      id: newBookId,
      title: newTitle.trim(),
      subtitle: newSubtitle.trim() || "A custom chronicle penned by the author",
      tag: newGenre.toUpperCase(),
      genre: newGenre,
      tone: "custom",
      bgHex: newToneColor,
      foilHex: newFoilColor,
      auraColor: `${newToneColor}55`,
      sealEmblem: newEmblem,
      isCustom: true,
      presets: [
        {
          title: "Initial Chapter Seed",
          prompt: newStartingPrompt.trim() || `Write the opening chapter of ${newTitle.trim()} introducing the world and protagonist.`
        }
      ],
      fateArchetypes: ["The Chosen Pioneer", "The Enigmatic Wanderer", "The Ancient Scholar", "The Sovereign"],
      fateCatalysts: ["Uncovering an unsealed prophecy", "A betrayal among allies", "A celestial alignment"],
      fateSettings: ["Underneath the twin moons", "Within the grand sanctum", "Across the forgotten wilderness"]
    };

    const updated = [...books, createdBook];
    setBooks(updated);

    try {
      const customOnes = updated.filter((b) => b.isCustom);
      localStorage.setItem(STORAGE_CUSTOM_BOOKS_KEY, JSON.stringify(customOnes));
    } catch (err) {
      console.error(err);
    }

    setIsForgingNewBook(false);
    setSelectedBook(createdBook);
    setPrompt(createdBook.presets[0].prompt);
    setActiveStreamingVerse("");
    setIsSealed(false);
    setViewingChapter(1);
  };

  const handleDeleteCustomBook = (e: React.MouseEvent, bookId: string) => {
    e.stopPropagation();
    const updated = books.filter((b) => b.id !== bookId);
    setBooks(updated);
    try {
      const customOnes = updated.filter((b) => b.isCustom);
      localStorage.setItem(STORAGE_CUSTOM_BOOKS_KEY, JSON.stringify(customOnes));
    } catch (err) {
      console.error(err);
    }
  };

  // Surgical Destiny Pivot
  const handleDestinyPivot = (pivotType: "combat" | "tragedy" | "mystery" | "triumph") => {
    let pivotPrompt = "";
    if (pivotType === "combat") {
      pivotPrompt = `Continue the chronicle with an intense climax: a sudden clash of arms, divine archery, and heroic defiance!`;
    } else if (pivotType === "tragedy") {
      pivotPrompt = `Continue the chronicle with a poignant, tragic revelation that tests the deepest bonds of destiny and honor.`;
    } else if (pivotType === "mystery") {
      pivotPrompt = `Continue the chronicle by introducing an ancient, mind-bending secret that questions everything known before.`;
    } else if (pivotType === "triumph") {
      pivotPrompt = `Continue the chronicle toward radiant triumph, with the golden dawn breaking through shadows and justice prevailing.`;
    }

    setPrompt(pivotPrompt);
    handleSendInscription(pivotPrompt);
  };

  // Inscribe in Ink: Multi-Turn Chat Generation with Conversation Context
  const handleSendInscription = async (overridePrompt?: string) => {
    if (!selectedBook || isCasting) return;
    const textToSend = (overridePrompt || prompt).trim();
    if (!textToSend) return;

    const existingTurns = sessions[selectedBook.id] || [];
    const targetChapter = Math.floor(existingTurns.length / 2) + 1;
    setViewingChapter(targetChapter);

    const userTurn: ChatTurn = {
      id: "turn_u_" + Date.now(),
      role: "chronicler",
      text: textToSend,
      timestamp: Date.now()
    };

    const updatedTurnsWithUser = [...existingTurns, userTurn];

    // Immediately record user turn
    const intermediateSessions = {
      ...sessions,
      [selectedBook.id]: updatedTurnsWithUser
    };
    setSessions(intermediateSessions);
    try {
      localStorage.setItem(STORAGE_SESSIONS_KEY, JSON.stringify(intermediateSessions));
    } catch (err) {
      console.error(err);
    }

    setPrompt("");
    setIsCasting(true);
    setCurrentStreamingToken("");
    setActiveStreamingVerse("");

    // Build context with previous turns for conversation continuity
    const historySnippet = existingTurns.slice(-4).map((t) =>
      `${t.role === "chronicler" ? "Chronicler" : "Oracle"}: ${t.text}`
    ).join("\n\n");

    const promptPayload = historySnippet
      ? `${historySnippet}\n\nChronicler: ${textToSend}\nOracle:`
      : textToSend;

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
              setCurrentStreamingToken(data.token);
              setActiveStreamingVerse(accumulated);
              tokenCount++;

              if (!firstTokenTime) {
                firstTokenTime = performance.now();
              }
              const elapsedSec = (performance.now() - firstTokenTime) / 1000;
              const currentTps =
                tokenCount > 1 && elapsedSec > 0.05
                  ? Math.round(tokenCount / elapsedSec)
                  : 38;

              setTokensPerSec(currentTps);
              setTotalTokensGenerated((prev) => prev + 1);
            } catch (err) {
              console.error("JSON parse error", err);
            }
          }
        }
      }

      if (accumulated.trim()) {
        const oracleTurn: ChatTurn = {
          id: "turn_o_" + Date.now(),
          role: "oracle",
          text: accumulated.trim(),
          timestamp: Date.now()
        };
        const finalTurns = [...updatedTurnsWithUser, oracleTurn];
        const finalSessions = {
          ...sessions,
          [selectedBook.id]: finalTurns
        };
        setSessions(finalSessions);
        try {
          localStorage.setItem(STORAGE_SESSIONS_KEY, JSON.stringify(finalSessions));
        } catch (err) {
          console.error(err);
        }
      }
    } catch (err: any) {
      const errorTurn: ChatTurn = {
        id: "turn_err_" + Date.now(),
        role: "oracle",
        text: `[Arcane Connection Error: ${err.message}. Ensure Karna-49M FastAPI server is running on ${BACKEND_URL}]`,
        timestamp: Date.now()
      };
      const finalTurns = [...updatedTurnsWithUser, errorTurn];
      setSessions({ ...sessions, [selectedBook.id]: finalTurns });
    } finally {
      setIsCasting(false);
      setCurrentStreamingToken("");
      setActiveStreamingVerse("");
      if (accumulated.length > 50) {
        try {
          confetti({
            particleCount: 30,
            spread: 65,
            origin: { x: 0.65, y: 0.38 },
            colors: ["#d4af37", "#fef08a", "#b45309"],
            shapes: ["star", "circle"],
            scalar: 0.9,
            gravity: 0.7,
            ticks: 150
          });
        } catch {}
      }
    }
  };

  const copyEntireChronicle = async () => {
    if (!selectedBook) return;
    const turns = sessions[selectedBook.id] || [];
    if (turns.length === 0) return;

    const transcript = [
      `══════════════════════════════════════════════════════════════════`,
      `FOLIO: ${selectedBook.title.toUpperCase()} (${selectedBook.genre})`,
      `CHRONICLER: Scribe & Karna-49M SLM`,
      `INSCRIBED TURNS: ${turns.length}`,
      `SEALED: ${isSealed ? "YES (ROYAL WAX SEAL AFFIXED)" : "UNSEALED"}`,
      `══════════════════════════════════════════════════════════════════\n`,
      ...turns.map((t, idx) => {
        if (t.role === "chronicler") {
          return `[Turn ${Math.floor(idx / 2) + 1} • Inscription]\n${t.text}\n`;
        } else {
          return `[The Oracle of Karna]\n${t.text}\n\n✦ ✦ ✦\n`;
        }
      })
    ].join("\n");

    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="relative min-h-[calc(100vh-70px)] text-slate-900 flex flex-col justify-start bg-[#faf7f2]">
      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 py-6 flex flex-col gap-6">
        <AnimatePresence mode="wait">
          {/* ========================================================= */}
          {/* VIEW A: THE 3D WOODEN BOOKSHELVES (HOME)                  */}
          {/* ========================================================= */}
          {!selectedBook ? (
            <motion.div
              key="shelves-view"
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3 }}
              className="flex flex-col gap-8"
            >
            {/* Library Masthead */}
            <div className="text-center space-y-3 max-w-3xl mx-auto pt-2">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/90 border border-amber-800/20 text-amber-950 text-xs font-mono tracking-widest uppercase shadow-md backdrop-blur-md">
                <Sparkles className="w-3.5 h-3.5 text-amber-600" />
                The Grand Scriptoria of Karna &bull; Every Tome is a Living Conversation
              </div>

              <h1 className="text-4xl md:text-5xl font-serif font-bold tracking-tight text-slate-900 filter drop-shadow-sm">
                The Archival Chat Sanctuary
              </h1>

              <p className="text-sm text-slate-700 font-serif italic leading-relaxed max-w-xl mx-auto">
                "In this sacred hall, words are not lost to ephemeral servers. Every book on these shelves is a persistent conversation session with Karna-49M. Select any tome to resume its dialogue, or inscribe a new book to begin fresh."
              </p>

              {/* Action Buttons: Forge Book & Stats */}
              <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
                {/* View Switcher: 3D Corridor vs Flat Shelves */}
                <div className="bg-white/95 p-1 rounded-xl border border-amber-800/20 flex items-center gap-1 shadow-sm">
                  <button
                    onClick={() => setLibraryMode("3d_corridor")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-serif font-bold transition cursor-pointer flex items-center gap-1.5 ${
                      libraryMode === "3d_corridor"
                        ? "bg-amber-600 text-white shadow-md shadow-amber-600/20"
                        : "text-slate-600 hover:text-slate-950 hover:bg-amber-50"
                    }`}
                  >
                    <Compass className="w-3.5 h-3.5" />
                    <span>🌟 3D Three.js Sanctuary</span>
                  </button>

                  <button
                    onClick={() => setLibraryMode("shelves")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-serif font-bold transition cursor-pointer flex items-center gap-1.5 ${
                      libraryMode === "shelves"
                        ? "bg-amber-600 text-white shadow-md shadow-amber-600/20"
                        : "text-slate-600 hover:text-slate-950 hover:bg-amber-50"
                    }`}
                  >
                    <BookOpen className="w-3.5 h-3.5" />
                    <span>🪵 Wooden Bookshelves</span>
                  </button>
                </div>

                <button
                  onClick={() => setIsForgingNewBook(true)}
                  className="px-4 py-2 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-serif font-bold text-xs flex items-center gap-1.5 shadow-lg shadow-amber-950/20 transition cursor-pointer"
                >
                  <Plus className="w-4 h-4 stroke-[2.5]" />
                  <span>➕ Inscribe New Tome</span>
                </button>
              </div>
            </div>

            {/* CONDITIONAL DISPLAY: 3D THREE.JS CORRIDOR VS WOODEN SHELVES */}
            {libraryMode === "3d_corridor" ? (
              <div className="max-w-6xl mx-auto w-full">
                <ThreeLibraryCorridor onSelectBook={handleSelectFromThree} sessionsSummary={sessionsSummary} />
              </div>
            ) : (
            /* ======================================================= */
            /* THE ARCHITECTURAL 3D WOODEN BOOKSHELF UNIT              */
            /* ======================================================= */
            <div className="wooden-bookshelf-unit max-w-6xl mx-auto w-full">
              {/* Carved Wooden Header Cornice */}
              <div className="h-6 bg-gradient-to-r from-[#b57a46] via-[#9e6433] to-[#7a4a21] border-b border-[#5a3414] flex items-center justify-between px-6 text-[10px] font-mono text-amber-100 tracking-widest uppercase shadow-sm">
                <span>✦ UPPER ARCHIVE &bull; HISTORICAL & MYTHIC LORE ✦</span>
                <span>FOLIO SECTION I</span>
              </div>

              {/* TIER 1: UPPER WOODEN SHELF */}
              <div className="wooden-shelf-tier">
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8 px-4">
                  {books.slice(0, 3).map((book) => (
                    <div
                      key={book.id}
                      onClick={() => {
                        setSelectedBook(book);
                        if (!sessions[book.id] || sessions[book.id].length === 0) {
                          setPrompt(book.presets[0]?.prompt || `Write a chronicle of ${book.title}.`);
                        } else {
                          setPrompt("");
                        }
                        setActiveStreamingVerse("");
                      }}
                      className="tome-shelf-slot group cursor-pointer"
                    >
                      {/* The Standing Inscribed Leather Tome */}
                      <div
                        style={{ backgroundColor: book.bgHex, color: book.foilHex }}
                        className="tome-cover relative z-10"
                      >
                        {/* Sealed Wax Stamp Tag */}
                        {sealedBooks[book.id] && (
                          <div className="absolute -top-2.5 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-red-800 border border-amber-300 text-[8px] font-mono text-amber-100 shadow-md z-20">
                            ✦ SEALED ✦
                          </div>
                        )}

                        {/* Metallic Gold Corner Protectors */}
                        <div className="tome-corner-top-left" />
                        <div className="tome-corner-bottom-right" />
                        <div className="tome-border" />

                        {/* Top Foil Tag */}
                        <span className="text-[8px] uppercase tracking-widest font-mono opacity-70">
                          {book.tag}
                        </span>

                        {/* Center Inscribed Seal & Title */}
                        <div className="flex flex-col items-center gap-1.5 text-center my-auto px-1">
                          <span className="text-3xl filter drop-shadow-md mb-1">{book.sealEmblem}</span>
                          <h3 className="font-serif text-lg font-bold leading-tight text-amber-100 group-hover:text-amber-50">
                            {book.title}
                          </h3>
                          <span className="text-[9px] text-amber-200/60 font-mono tracking-wider">
                            ✧ INSCRIBED FOLIO ✧
                          </span>
                        </div>

                        {/* Bottom Genre Foil Stamp & Session Turns */}
                        <div className="w-full flex items-center justify-between text-[8px] font-mono opacity-85 border-t border-current/20 pt-1">
                          <span>{book.genre}</span>
                          <span className="font-semibold text-amber-200">
                            {sessions[book.id]?.length
                              ? `📜 ${sessions[book.id].length} Turns`
                              : "✦ Fresh Folio"}
                          </span>
                        </div>

                        {/* Delete Custom Book Badge */}
                        {book.isCustom && (
                          <button
                            onClick={(e) => handleDeleteCustomBook(e, book.id)}
                            className="absolute top-2 right-2 p-1 rounded-full bg-red-950/80 hover:bg-red-800 text-red-200 opacity-0 group-hover:opacity-100 transition z-20"
                            title="Remove this book from shelf"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        )}
                      </div>

                      {/* Tome Caption under the book */}
                      <div className="text-center space-y-0.5 pt-3 z-10">
                        <span className="text-xs font-serif italic text-slate-800 block">
                          "{book.subtitle}"
                        </span>
                        <span className="text-[11px] font-mono text-amber-900 font-semibold flex items-center justify-center gap-1 group-hover:text-amber-700 transition">
                          <BookOpen className="w-3 h-3 text-amber-800" /> Open Chronicle ➔
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* The Heavy Polished Oak Plank where the books stand */}
                <div className="wooden-shelf-plank" />
              </div>

              {/* TIER 2: LOWER WOODEN SHELF */}
              <div className="wooden-shelf-tier pt-2">
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8 px-4">
                  {books.slice(3).map((book) => (
                    <div
                      key={book.id}
                      onClick={() => {
                        setSelectedBook(book);
                        if (!sessions[book.id] || sessions[book.id].length === 0) {
                          setPrompt(book.presets[0]?.prompt || `Write a chronicle of ${book.title}.`);
                        } else {
                          setPrompt("");
                        }
                        setActiveStreamingVerse("");
                      }}
                      className="tome-shelf-slot group cursor-pointer"
                    >
                      {/* The Standing Inscribed Leather Tome */}
                      <div
                        style={{ backgroundColor: book.bgHex, color: book.foilHex }}
                        className="tome-cover relative z-10"
                      >
                        {/* Sealed Wax Stamp Tag */}
                        {sealedBooks[book.id] && (
                          <div className="absolute -top-2.5 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-red-800 border border-amber-300 text-[8px] font-mono text-amber-100 shadow-md z-20">
                            ✦ SEALED ✦
                          </div>
                        )}

                        <div className="tome-corner-top-left" />
                        <div className="tome-corner-bottom-right" />
                        <div className="tome-border" />

                        <span className="text-[8px] uppercase tracking-widest font-mono opacity-70">
                          {book.tag}
                        </span>

                        <div className="flex flex-col items-center gap-1.5 text-center my-auto px-1">
                          <span className="text-3xl filter drop-shadow-md mb-1">{book.sealEmblem}</span>
                          <h3 className="font-serif text-lg font-bold leading-tight text-amber-100 group-hover:text-amber-50">
                            {book.title}
                          </h3>
                          <span className="text-[9px] text-amber-200/60 font-mono tracking-wider">
                            ✧ INSCRIBED FOLIO ✧
                          </span>
                        </div>

                        <div className="w-full flex items-center justify-between text-[8px] font-mono opacity-85 border-t border-current/20 pt-1">
                          <span>{book.genre}</span>
                          <span className="font-semibold text-amber-200">
                            {sessions[book.id]?.length
                              ? `📜 ${sessions[book.id].length} Turns`
                              : "✦ Fresh Folio"}
                          </span>
                        </div>

                        {book.isCustom && (
                          <button
                            onClick={(e) => handleDeleteCustomBook(e, book.id)}
                            className="absolute top-2 right-2 p-1 rounded-full bg-red-950/80 hover:bg-red-800 text-red-200 opacity-0 group-hover:opacity-100 transition z-20"
                            title="Remove this book from shelf"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        )}
                      </div>

                      {/* Tome Caption under the book */}
                      <div className="text-center space-y-0.5 pt-3 z-10">
                        <span className="text-xs font-serif italic text-slate-800 block">
                          "{book.subtitle}"
                        </span>
                        <span className="text-[11px] font-mono text-amber-900 font-semibold flex items-center justify-center gap-1 group-hover:text-amber-700 transition">
                          <BookOpen className="w-3 h-3 text-amber-800" /> Open Chronicle ➔
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* The Lower Heavy Polished Oak Plank */}
                <div className="wooden-shelf-plank" />
              </div>
            </div>
            )}

        {/* ========================================================= */}
        {/* FORGE NEW GRIMOIRE MODAL (DYNAMIC BOOK CREATION)          */}
        {/* ========================================================= */}
        {isForgingNewBook && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 animate-in fade-in">
            <div className="bg-[#faf7f2] border border-amber-900/20 rounded-2xl p-6 max-w-lg w-full shadow-2xl space-y-5 text-slate-900">
              <div className="flex items-center justify-between pb-3 border-b border-amber-900/15">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-amber-600" />
                  <h3 className="font-serif text-xl font-bold text-slate-900">
                    Forge a New Grimoire
                  </h3>
                </div>
                <button
                  onClick={() => setIsForgingNewBook(false)}
                  className="p-1.5 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-amber-100/60 transition"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <form onSubmit={handleForgeBook} className="space-y-4 text-xs">
                {/* Title */}
                <div className="space-y-1">
                  <label className="text-slate-800 font-semibold block">Tome Title</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Chronicles of the Crimson Nebula"
                    value={newTitle}
                    onChange={(e) => setNewTitle(e.target.value)}
                    className="w-full bg-white border border-amber-900/20 rounded-lg p-2.5 text-sm font-serif text-slate-900 focus:outline-none focus:border-amber-600 shadow-sm"
                  />
                </div>

                {/* Subtitle */}
                <div className="space-y-1">
                  <label className="text-slate-800 font-semibold block">Subtitle / Lore Tagline</label>
                  <input
                    type="text"
                    placeholder="e.g. Ancient interstellar oaths and forgotten kings"
                    value={newSubtitle}
                    onChange={(e) => setNewSubtitle(e.target.value)}
                    className="w-full bg-white border border-amber-900/20 rounded-lg p-2.5 text-xs font-serif text-slate-900 focus:outline-none focus:border-amber-600 shadow-sm"
                  />
                </div>

                {/* Genre & Emblem Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-slate-800 font-semibold block">Genre</label>
                    <select
                      value={newGenre}
                      onChange={(e) => setNewGenre(e.target.value)}
                      className="w-full bg-white border border-amber-900/20 rounded-lg p-2 text-xs text-slate-900 focus:outline-none focus:border-amber-600 shadow-sm"
                    >
                      <option value="Epic Mythology">Epic Mythology</option>
                      <option value="High Fantasy">High Fantasy</option>
                      <option value="Cyberpunk Sci-Fi">Cyberpunk Sci-Fi</option>
                      <option value="Gothic Horror">Gothic Horror</option>
                      <option value="Historical Biography">Historical Biography</option>
                      <option value="Cosmic Wonder">Cosmic Wonder</option>
                      <option value="Fable Adventure">Fable Adventure</option>
                    </select>
                  </div>

                  <div className="space-y-1">
                    <label className="text-slate-800 font-semibold block">Heraldic Emblem</label>
                    <div className="flex gap-1.5 overflow-x-auto pb-1">
                      {["⚔️", "☀️", "🐉", "⚡", "🗝️", "🐾", "🏛️", "👑", "👁️", "🌌", "🏹"].map((icon) => (
                        <button
                          key={icon}
                          type="button"
                          onClick={() => setNewEmblem(icon)}
                          className={`p-1.5 rounded text-sm transition ${
                            newEmblem === icon ? "bg-amber-100 border border-amber-600 scale-110 shadow-sm" : "bg-white border border-amber-900/20"
                          }`}
                        >
                          {icon}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Leather Binding Tone Color */}
                <div className="space-y-1">
                  <label className="text-slate-800 font-semibold block">Leather Binding Color</label>
                  <div className="flex gap-3">
                    {[
                      { hex: "#481219", label: "Crimson Wine" },
                      { hex: "#142f1d", label: "Deep Forest" },
                      { hex: "#141f33", label: "Midnight Blue" },
                      { hex: "#103337", label: "Cyber Teal" },
                      { hex: "#2c123b", label: "Imperial Purple" },
                      { hex: "#573714", label: "Aged Ochre" },
                      { hex: "#151619", label: "Obsidian" }
                    ].map((col) => (
                      <button
                        key={col.hex}
                        type="button"
                        onClick={() => setNewToneColor(col.hex)}
                        style={{ backgroundColor: col.hex }}
                        className={`w-7 h-7 rounded-full border-2 transition ${
                          newToneColor === col.hex ? "border-amber-600 scale-110 shadow-md" : "border-slate-300"
                        }`}
                        title={col.label}
                      />
                    ))}
                  </div>
                </div>

                {/* Starting Seed Prompt */}
                <div className="space-y-1">
                  <label className="text-slate-800 font-semibold block">Opening Chapter Prompt Seed</label>
                  <textarea
                    rows={2}
                    placeholder="Enter the opening scene or instruction for this tome..."
                    value={newStartingPrompt}
                    onChange={(e) => setNewStartingPrompt(e.target.value)}
                    className="w-full bg-white border border-amber-900/20 rounded-lg p-2 text-xs font-serif text-slate-900 focus:outline-none focus:border-amber-600 resize-none shadow-sm"
                  />
                </div>

                {/* Submit button */}
                <div className="pt-2 flex justify-end gap-3">
                  <button
                    type="button"
                    onClick={() => setIsForgingNewBook(false)}
                    className="px-4 py-2 rounded-lg bg-amber-100 hover:bg-amber-200 text-slate-800 text-xs font-semibold transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-5 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-serif font-bold text-xs flex items-center gap-1.5 shadow-md shadow-amber-950/20 transition cursor-pointer"
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Inscribe & Place on Shelf</span>
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </motion.div>
    ) : (
          /* ========================================================= */
          /* VIEW B: THE OPEN DUAL-PAGE GRIMOIRE (LEFT PROMPT / RIGHT STORY) */
          /* ========================================================= */
          <motion.div
            key="grimoire-view"
            initial={{ opacity: 0, scale: 0.93, rotateX: 4 }}
            animate={{ opacity: 1, scale: 1, rotateX: 0 }}
            exit={{ opacity: 0, scale: 0.94, rotateX: -4 }}
            transition={{ type: "spring", stiffness: 130, damping: 20 }}
            className="flex flex-col gap-4"
          >
            {/* Navigation & Telemetry Bar Above Book */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 bg-white/95 border border-amber-900/20 rounded-2xl px-5 py-3 backdrop-blur-md shadow-lg text-slate-800">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setSelectedBook(null)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-900/20 text-xs font-serif font-medium text-amber-950 hover:bg-amber-100 transition cursor-pointer shadow-sm"
                >
                  <ArrowLeft className="w-3.5 h-3.5 text-amber-800" />
                  <span>Library Shelves</span>
                </button>

                <div className="h-4 w-[1px] bg-amber-900/20 hidden sm:block" />

                <div>
                  <h2 className="text-sm font-bold text-slate-900 font-serif flex items-center gap-2">
                    <span className="text-base">{selectedBook.sealEmblem}</span>
                    <span>{selectedBook.title}</span>
                    <span className="text-[11px] font-sans font-normal text-amber-800">({selectedBook.genre})</span>
                  </h2>
                </div>
              </div>

              {/* Quick Actions & Telemetry */}
              <div className="flex items-center gap-3 text-xs">
                {isCasting && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-100 border border-amber-800/30 text-amber-900 animate-pulse font-mono">
                    <Activity className="w-3.5 h-3.5" />
                    <span>Inking: {tokensPerSec} tok/s</span>
                  </div>
                )}

                <button
                  onClick={() => setShowMarginalia(!showMarginalia)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs transition cursor-pointer font-medium ${
                    showMarginalia
                      ? "bg-amber-100 border-amber-800/40 text-amber-950 shadow-sm"
                      : "bg-white border-amber-900/20 text-slate-700 hover:text-slate-950 hover:bg-amber-50"
                  }`}
                  title="Toggle Neural Marginalia & SLM Thought Lens"
                >
                  <Eye className="w-3.5 h-3.5 text-amber-800" />
                  <span>{showMarginalia ? "Marginalia On" : "Marginalia Off"}</span>
                </button>

                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-amber-900/20 text-slate-700 hover:text-slate-950 hover:bg-amber-50 transition cursor-pointer font-medium shadow-sm"
                >
                  <Sliders className="w-3.5 h-3.5 text-amber-800" />
                  <span>Quill Sliders</span>
                </button>

                {onOpenArchitecture && (
                  <button
                    onClick={onOpenArchitecture}
                    className="hidden md:flex items-center gap-1 text-[11px] text-purple-700 hover:text-purple-900 transition font-medium"
                  >
                    <Layers className="w-3.5 h-3.5" />
                    <span>Anatomy</span>
                  </button>
                )}
              </div>
            </div>

            {/* Expandable Quill Settings Modal/Drawer */}
            {showSettings && (
              <div className="p-4 bg-[#fcfaf6] border border-amber-900/20 rounded-xl grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs animate-in slide-in-from-top-2 text-slate-800 shadow-sm">
                <div className="space-y-1">
                  <div className="flex justify-between font-mono">
                    <span className="text-slate-700">Creativity (Temperature):</span>
                    <span className="text-amber-900 font-bold">{temperature}</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="1.4"
                    step="0.05"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-amber-200 rounded accent-amber-600 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-500">Lower = historical fidelity, Higher = mythic invention</p>
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between font-mono">
                    <span className="text-slate-700">Scroll Length (Max Tokens):</span>
                    <span className="text-amber-900 font-bold">{maxTokens}</span>
                  </div>
                  <input
                    type="range"
                    min="64"
                    max="1024"
                    step="32"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-amber-200 rounded accent-amber-600 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-500">1 Chapter ≈ 300 to 500 tokens</p>
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between font-mono">
                    <span className="text-slate-700">Vocabulary Focus (Top-K):</span>
                    <span className="text-amber-900 font-bold">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="80"
                    step="5"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-amber-200 rounded accent-amber-600 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-500">Filters probability pool to top candidate tokens</p>
                </div>
              </div>
            )}

            {/* ========================================================= */}
            {/* THE OPEN 3D DUAL-PAGE BOOK STAGE                          */}
            {/* ========================================================= */}
            <div className="book-stage-wrapper relative w-full rounded-lg">
              {/* Backing Book Cover Overhang */}
              <div
                style={{ backgroundColor: selectedBook.bgHex }}
                className="book-backing-cover"
              />

              {/* Ornate Gold Brass Corner Protectors */}
              <div className="absolute -top-1 -left-1 w-6 h-6 border-t-2 border-l-2 border-[#ecd38c] rounded-tl-sm pointer-events-none z-20 shadow-sm" />
              <div className="absolute -top-1 -right-1 w-6 h-6 border-t-2 border-r-2 border-[#ecd38c] rounded-tr-sm pointer-events-none z-20 shadow-sm" />
              <div className="absolute -bottom-1 -left-1 w-6 h-6 border-b-2 border-l-2 border-[#ecd38c] rounded-bl-sm pointer-events-none z-20 shadow-sm" />
              <div className="absolute -bottom-1 -right-1 w-6 h-6 border-b-2 border-r-2 border-[#ecd38c] rounded-br-sm pointer-events-none z-20 shadow-sm" />

              {/* The Two Pages */}
              <div className="book-pages-container relative rounded-md overflow-hidden border border-amber-950/40">
                {/* Center Spine Crease Gradient */}
                <div className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-8 bg-gradient-to-r from-amber-950/5 via-amber-950/20 to-amber-950/5 pointer-events-none z-20 hidden md:block" />
                {/* ----------------------------------------------------- */}
                {/* LEFT PAGE: THE SCRIBE'S FOLIO (INPUT OR RECORDED)     */}
                {/* ----------------------------------------------------- */}
                <article className="book-page-parchment left-parchment-page flex flex-col justify-between">
                  <div className="space-y-3">
                    {/* Topline */}
                    <div className="flex items-center justify-between text-[10px] text-amber-950/60 font-mono tracking-widest uppercase pb-2 border-b border-amber-950/15">
                      <span>PAGE {toRoman((viewingChapter - 1) * 2 + 1)} &bull; CHAPTER {toRoman(viewingChapter)} &bull; THE SCRIBE</span>
                      <div className="flex items-center gap-1.5 font-bold text-amber-900">
                        <Feather className="w-3.5 h-3.5 text-amber-900" strokeWidth={1.5} />
                        <span>Folio {viewingChapter} of {totalFolios}</span>
                      </div>
                    </div>

                    {/* Entry Header */}
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-[10px] font-mono tracking-wider text-amber-900/50 uppercase block">
                          FOLIO CODEX &bull; {selectedBook.tag}
                        </span>
                        <h2 className="font-serif text-xl font-bold text-amber-950 tracking-tight flex items-center gap-1.5">
                          <span>{selectedBook.sealEmblem}</span>
                          <span>{selectedBook.title}</span>
                        </h2>
                      </div>

                      {/* Oracle Fate Dice Roller Button (available when writing) */}
                      {(!chapters[viewingChapter - 1]?.oracleResponse || viewingChapter > chapters.length) && (
                        <button
                          onClick={handleRollFateDice}
                          disabled={isRollingDice || isCasting}
                          className="px-2.5 py-1.5 rounded-lg bg-amber-950/10 hover:bg-amber-950/20 text-amber-950 border border-amber-950/20 text-xs font-serif flex items-center gap-1.5 shadow-sm transition cursor-pointer"
                          title="Roll the 3 Fate Dice to synthesize a dramatic continuation prompt"
                        >
                          <Dices className={`w-4 h-4 text-amber-900 ${isRollingDice ? "animate-spin" : ""}`} />
                          <span>{isRollingDice ? "Casting..." : "Roll Fate Dice"}</span>
                        </button>
                      )}
                    </div>

                    {/* IF VIEWING AN ALREADY-COMPLETED CHAPTER */}
                    {chapters[viewingChapter - 1]?.oracleResponse && viewingChapter <= chapters.length ? (
                      <div className="space-y-3 pt-1">
                        <div className="p-3.5 rounded-xl bg-amber-900/5 border border-amber-900/15 space-y-2">
                          <div className="flex items-center justify-between font-mono text-[10px] text-amber-900/70 uppercase font-semibold">
                            <span className="flex items-center gap-1.5">
                              <Feather className="w-3.5 h-3.5 text-amber-800 -rotate-45" />
                              <span>Chronicler's Inscribed Invocation:</span>
                            </span>
                            <span>Chapter {toRoman(viewingChapter)}</span>
                          </div>
                          <p className="font-serif italic text-sm text-slate-800 leading-relaxed pl-1 whitespace-pre-wrap">
                            "{chapters[viewingChapter - 1].userPrompt}"
                          </p>
                        </div>

                        <div className="p-3 rounded-lg bg-amber-900/5 border border-amber-950/10 text-xs font-serif text-amber-900/80 space-y-1">
                          <p className="font-bold text-amber-950 flex items-center gap-1">
                            <Sparkles className="w-3.5 h-3.5 text-amber-600" />
                            <span>Folio Sealed in Story Chronicle</span>
                          </p>
                          <p className="text-[11px] leading-relaxed">
                            This chapter's prompt is forever chronicled. You can turn to the next page to continue writing, or jump to the active writing desk.
                          </p>
                        </div>

                        {viewingChapter < totalFolios && (
                          <div className="pt-2 flex flex-col gap-2">
                            <button
                              onClick={() => setViewingChapter(viewingChapter + 1)}
                              className="w-full py-2.5 px-3 rounded-lg bg-amber-900/10 hover:bg-amber-900/20 text-amber-950 font-serif text-xs font-bold flex items-center justify-center gap-2 border border-amber-950/20 transition cursor-pointer"
                            >
                              <span>Turn to Chapter {toRoman(viewingChapter + 1)} Page ➔</span>
                              <ArrowRight className="w-3.5 h-3.5" />
                            </button>

                            <button
                              onClick={() => {
                                setViewingChapter(totalFolios);
                                setPrompt("");
                              }}
                              className="w-full py-2 px-3 rounded-lg bg-amber-950 hover:bg-amber-900 text-amber-100 font-serif text-xs font-bold flex items-center justify-center gap-2 shadow-xs transition cursor-pointer"
                            >
                              <Feather className="w-3.5 h-3.5 text-amber-300" />
                              <span>Jump to Writing Folio (Chapter {toRoman(totalFolios)}) ➔</span>
                            </button>
                          </div>
                        )}
                      </div>
                    ) : (
                      /* ACTIVE WRITING CHAPTER (BLANK OR STREAMING) */
                      <div className="space-y-2.5">
                        {/* Fate Dice Result Display */}
                        {(rolledArchetype || isRollingDice) && (
                          <div className="p-2 rounded bg-amber-900/10 border border-amber-950/15 text-[11px] font-serif text-amber-950 space-y-0.5 animate-in fade-in">
                            <div className="flex items-center gap-1 text-[10px] font-mono text-amber-900 font-bold uppercase">
                              <Sparkles className="w-3 h-3 text-amber-800" />
                              <span>Oracle’s Fate Roll:</span>
                            </div>
                            <p className="italic text-amber-950/90 leading-tight">
                              "{rolledArchetype} &bull; {rolledCatalyst} &bull; {rolledSetting}"
                            </p>
                          </div>
                        )}

                        {/* Continuation Context from Previous Chapter */}
                        {viewingChapter > 1 && chapters[viewingChapter - 2] && (
                          <div className="p-2.5 rounded-lg bg-amber-900/5 border border-amber-950/15 text-xs font-serif text-amber-950 space-y-1">
                            <div className="flex items-center gap-1.5 font-mono text-[9px] text-amber-900/70 uppercase font-semibold">
                              <Scroll className="w-3 h-3 text-amber-700" />
                              <span>Continuing from Chapter {toRoman(viewingChapter - 1)}:</span>
                            </div>
                            <p className="italic text-slate-800 line-clamp-2 leading-relaxed">
                              "{chapters[viewingChapter - 2]?.oracleResponse?.slice(-120) || chapters[viewingChapter - 2]?.userPrompt}"
                            </p>
                          </div>
                        )}

                        {/* Starting Seeds for Chapter 1 */}
                        {viewingChapter === 1 && chapters.length === 0 && (
                          <div className="py-1 text-center space-y-1">
                            <span className="text-[10px] font-mono uppercase text-amber-900/60 font-semibold block text-left">
                              ✦ Starting Inspiration Seeds:
                            </span>
                            <div className="flex flex-col gap-1">
                              {selectedBook.presets.map((preset, idx) => (
                                <button
                                  key={idx}
                                  onClick={() => setPrompt(preset.prompt)}
                                  className="text-left text-xs font-serif p-1.5 rounded bg-amber-900/5 hover:bg-amber-900/10 text-amber-950 border border-amber-950/10 transition flex items-center justify-between group cursor-pointer"
                                >
                                  <span>&bull; {preset.title}</span>
                                  <span className="text-[10px] font-mono text-amber-800 opacity-0 group-hover:opacity-100 transition">
                                    Pen ➔
                                  </span>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Sacred Inscription Inkwell Textarea */}
                        <div className="space-y-1.5 pt-1">
                          <div className="flex items-center justify-between text-[10px] font-mono text-amber-900/70 uppercase">
                            <span className="flex items-center gap-1 font-bold">
                              <Feather className="w-3 h-3 text-amber-800" />
                              <span>Chapter {toRoman(viewingChapter)} Inscription</span>
                            </span>
                            <span className="italic text-amber-900/50">Enter ↵ to inscribe</span>
                          </div>

                          <div className="relative bg-white/75 rounded-lg border border-amber-950/25 p-2.5 focus-within:border-amber-700 focus-within:ring-1 focus-within:ring-amber-700/30 transition shadow-inner">
                            <textarea
                              value={prompt}
                              onChange={(e) => setPrompt(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                  e.preventDefault();
                                  handleSendInscription();
                                }
                              }}
                              placeholder={
                                viewingChapter === 1
                                  ? "Dip your quill to inscribe Chapter I... (Press Enter to inscribe, Shift+Enter for newline)"
                                  : `Inscribe Chapter ${toRoman(viewingChapter)}... What deeds or destinies unfold next?`
                              }
                              disabled={isCasting}
                              rows={4}
                              className="w-full text-xs font-serif text-amber-950 placeholder-amber-950/40 focus:outline-none resize-none bg-transparent leading-relaxed"
                            />
                          </div>

                          {/* Surgical Destiny Pivots & Send Button */}
                          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                            <div className="flex flex-wrap items-center gap-1 text-[10px] font-serif">
                              <span className="text-[9px] font-mono text-amber-900/60 uppercase mr-0.5">Pivot:</span>
                              <button
                                onClick={() => handleDestinyPivot("combat")}
                                disabled={isCasting}
                                className="px-2 py-1 rounded bg-amber-950/5 hover:bg-amber-950/15 border border-amber-950/15 text-amber-950 flex items-center gap-1 transition cursor-pointer"
                                title="Steer story toward an intense clash of arms"
                              >
                                <Sword className="w-3 h-3 text-amber-800" />
                                <span>Clash</span>
                              </button>
                              <button
                                onClick={() => handleDestinyPivot("tragedy")}
                                disabled={isCasting}
                                className="px-2 py-1 rounded bg-amber-950/5 hover:bg-amber-950/15 border border-amber-950/15 text-amber-950 flex items-center gap-1 transition cursor-pointer"
                                title="Steer story toward a poignant sacrifice"
                              >
                                <HeartCrack className="w-3 h-3 text-rose-800" />
                                <span>Tragedy</span>
                              </button>
                              <button
                                onClick={() => handleDestinyPivot("mystery")}
                                disabled={isCasting}
                                className="px-2 py-1 rounded bg-amber-950/5 hover:bg-amber-950/15 border border-amber-950/15 text-amber-950 flex items-center gap-1 transition cursor-pointer"
                                title="Steer story toward an ancient secret"
                              >
                                <Moon className="w-3 h-3 text-indigo-800" />
                                <span>Secret</span>
                              </button>
                              <button
                                onClick={() => handleDestinyPivot("triumph")}
                                disabled={isCasting}
                                className="px-2 py-1 rounded bg-amber-950/5 hover:bg-amber-950/15 border border-amber-950/15 text-amber-950 flex items-center gap-1 transition cursor-pointer"
                                title="Steer story toward radiant dawn and victory"
                              >
                                <Sun className="w-3 h-3 text-amber-800" />
                                <span>Triumph</span>
                              </button>
                            </div>

                            <button
                              onClick={() => handleSendInscription()}
                              disabled={isCasting || !prompt.trim()}
                              className="px-4 py-2 rounded-lg bg-amber-950 hover:bg-amber-900 text-amber-100 font-serif text-xs font-bold flex items-center gap-1.5 shadow-md shadow-amber-950/25 disabled:opacity-40 transition cursor-pointer"
                            >
                              {isCasting ? (
                                <>
                                  <Feather className="w-3.5 h-3.5 animate-bounce" />
                                  <span>Inking Folio...</span>
                                </>
                              ) : (
                                <>
                                  <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                                  <span>Inscribe in Ink ↵</span>
                                </>
                              )}
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Left Page Footer */}
                  <div className="mt-3 pt-2 border-t border-amber-950/15 flex items-center justify-between text-[11px] font-serif text-amber-950/70">
                    <button
                      onClick={handleClearFolio}
                      disabled={isCasting || activeTurns.length === 0}
                      className="flex items-center gap-1 text-amber-900/70 hover:text-red-800 disabled:opacity-30 transition cursor-pointer"
                      title="Clear all inscriptions in this book and start fresh"
                    >
                      <RotateCcw className="w-3 h-3" />
                      <span>Purge Folio</span>
                    </button>

                    <span className="font-mono text-[10px] text-amber-950/50">
                      Folio {viewingChapter} &bull; Karna-49M SLM
                    </span>
                  </div>
                </article>

                {/* ----------------------------------------------------- */}
                {/* RIGHT PAGE: THE FULL SLM STORY MANUSCRIPT             */}
                {/* ----------------------------------------------------- */}
                <article className="book-page-parchment right-parchment-page flex flex-col justify-between relative">
                  {/* Decorative Gold Ribbon Bookmark */}
                  <div className="bookmark-gold-ribbon" />

                  <div className="flex flex-col h-full justify-between">
                    <div>
                      {/* Topline */}
                      <div className="flex items-center justify-between text-[10px] text-amber-950/60 font-mono tracking-widest uppercase pb-2 border-b border-amber-950/15">
                        <span>PAGE {toRoman((viewingChapter - 1) * 2 + 2)} &bull; CHAPTER {toRoman(viewingChapter)} &bull; THE LIVING MANUSCRIPT</span>
                        <span className="font-serif italic font-bold text-amber-950/70">Oracle Chamber</span>
                      </div>

                      {/* Chapter Title & Action Controls */}
                      <div className="mt-3 mb-2 flex items-center justify-between">
                        <div>
                          <span className="text-[10px] font-mono tracking-wider text-amber-900/50 uppercase block">
                            CHRONICLE MANUSCRIPT &bull; CHAPTER {toRoman(viewingChapter)}
                          </span>
                          <h3 className="font-serif text-lg font-bold text-amber-950 flex items-center gap-1.5">
                            <span>{selectedBook.title}</span>
                            {isSealed && <span className="text-xs text-amber-800 font-normal">✧ Sealed with Wax</span>}
                          </h3>
                        </div>

                        <div className="flex items-center gap-2">
                          {chapters[viewingChapter - 1]?.oracleResponse && (
                            <button
                              onClick={() => {
                                const text = chapters[viewingChapter - 1]?.oracleResponse || "";
                                if (text) {
                                  navigator.clipboard.writeText(text);
                                  setCopied(true);
                                  setTimeout(() => setCopied(false), 2000);
                                }
                              }}
                              className="p-1.5 rounded hover:bg-amber-950/10 text-amber-900 transition cursor-pointer"
                              title="Copy this Chapter Verse"
                            >
                              {copied ? <Check className="w-4 h-4 text-emerald-800" /> : <Copy className="w-4 h-4" />}
                            </button>
                          )}

                          {/* Wax Seal Button */}
                          <button
                            onClick={(e) => handleAffixWaxSeal(e)}
                            className={`wax-seal transition cursor-pointer ${isSealed ? "ring-2 ring-amber-400 scale-105 shadow-amber-500/30" : ""}`}
                            title="Click to stamp chapter with royal wax seal & confetti"
                          >
                            <span className="text-xl filter drop-shadow">{selectedBook.sealEmblem}</span>
                          </button>
                        </div>
                      </div>

                      {/* Main Story Manuscript Chamber */}
                      <div
                        ref={rightPageScrollRef}
                        className="mt-2 overflow-y-auto max-h-[360px] pr-2 space-y-3"
                      >
                        {/* CASE 1: CHAPTER IS COMPLETED */}
                        {chapters[viewingChapter - 1]?.oracleResponse ? (
                          <div className="space-y-3">
                            <div className="flex items-center justify-between text-[10px] font-mono text-amber-900/60 uppercase">
                              <span className="flex items-center gap-1 font-semibold">
                                <Sparkles className="w-3 h-3 text-amber-600" />
                                <span>Oracle Verses &bull; Chapter {toRoman(viewingChapter)}</span>
                              </span>
                            </div>
                            <div className="font-serif text-sm leading-relaxed text-amber-950 whitespace-pre-wrap select-text drop-cap pl-1">
                              {chapters[viewingChapter - 1].oracleResponse}
                            </div>
                            <div className="flex items-center justify-center my-3 text-amber-900/25 text-[10px] font-mono select-none">
                              ✦ &bull; ✦ &bull; ✦
                            </div>
                          </div>
                        ) : isCasting ? (
                          /* CASE 2: CURRENTLY INKING LIVE CALLIGRAPHY */
                          <div className="p-3.5 rounded-xl bg-amber-50/85 border border-amber-900/15 shadow-sm space-y-2">
                            <div className="flex items-center justify-between text-[10px] font-mono text-amber-800 font-bold uppercase">
                              <span className="flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-amber-600 animate-ping inline-block" />
                                <span>Inking Chapter {toRoman(viewingChapter)} Verses...</span>
                              </span>
                              {tokensPerSec > 0 && <span>{tokensPerSec} tok/s</span>}
                            </div>
                            <div className="font-serif text-sm leading-relaxed text-amber-950 whitespace-pre-wrap select-text">
                              {activeStreamingVerse}
                              <span className="quill-inking-tip ml-1.5 px-2 py-0.5 rounded-full bg-amber-100 border border-amber-400/50 shadow-sm text-amber-900 font-mono text-[11px]">
                                <span className="wet-ink-token font-serif font-bold">{currentStreamingToken || "inking..."}</span>
                                <Feather className="w-3.5 h-3.5 text-amber-800 stroke-[2] -rotate-45" />
                              </span>
                            </div>
                          </div>
                        ) : (
                          /* CASE 3: BLANK STATE WATERMARK */
                          <div className="h-60 flex flex-col items-center justify-center text-center p-6 space-y-2">
                            <Feather className="w-16 h-16 text-amber-950/15 stroke-1" />
                            <h4 className="font-serif text-lg text-amber-950/60 font-medium">
                              Chapter {toRoman(viewingChapter)} Awaits Your Ink
                            </h4>
                            <p className="font-serif italic text-xs text-amber-950/50 max-w-sm">
                              Compose your invocation on the left page. The Oracle of Karna-49M will scribe Chapter {toRoman(viewingChapter)} onto this parchment in living ink.
                            </p>
                          </div>
                        )}

                        {/* Royal Sealed Wax Certificate Stamp */}
                        {isSealed && chapters[viewingChapter - 1]?.oracleResponse && (
                          <div className="mt-4 p-3 rounded-xl bg-amber-100/50 border border-amber-900/20 flex items-center justify-between gap-3 animate-in fade-in">
                            <div className="flex items-center gap-3">
                              <div className="wax-seal scale-90 shadow-md">
                                <span className="text-xl filter drop-shadow">{selectedBook.sealEmblem}</span>
                              </div>
                              <div>
                                <h5 className="font-serif font-bold text-xs text-amber-950 flex items-center gap-1.5">
                                  <span>Imperial Royal Seal Affixed</span>
                                  <span className="text-amber-700 font-mono text-[10px]">✧ Validated</span>
                                </h5>
                                <p className="font-serif italic text-[11px] text-amber-900/70">
                                  This chapter is permanently sanctified in the imperial grimoires.
                                </p>
                              </div>
                            </div>
                            <span className="wax-ribbon-tag px-3 py-1 rounded text-[10px] font-mono uppercase font-bold tracking-wider">
                              Archived &bull; Sealed
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* ACTION FOOTER ON RIGHT PAGE: THE NEXT PAGE BUTTON (CRITICAL) */}
                    <div>
                      {/* NEXT PAGE NAVIGATION CALL TO ACTION */}
                      {chapters[viewingChapter - 1]?.oracleResponse && !isCasting && (
                        <div className="mt-3 pt-2.5 border-t border-amber-950/15 flex flex-col sm:flex-row items-center justify-between gap-2.5 bg-amber-900/5 p-2.5 rounded-lg border border-amber-900/10">
                          <span className="text-[11px] font-serif italic text-amber-900/70">
                            Chapter {toRoman(viewingChapter)} concluded in ink.
                          </span>

                          {viewingChapter < chapters.length ? (
                            <button
                              onClick={() => setViewingChapter(viewingChapter + 1)}
                              className="w-full sm:w-auto px-4 py-2 rounded-lg bg-amber-900 hover:bg-amber-950 text-amber-100 font-serif font-bold text-xs flex items-center justify-center gap-1.5 shadow-sm transition cursor-pointer"
                            >
                              <span>Turn to Next Page (Chapter {toRoman(viewingChapter + 1)})</span>
                              <ArrowRight className="w-3.5 h-3.5" />
                            </button>
                          ) : (
                            <button
                              onClick={() => {
                                setViewingChapter(chapters.length + 1);
                                setPrompt("");
                              }}
                              className="w-full sm:w-auto px-5 py-2.5 rounded-lg bg-gradient-to-r from-amber-900 via-amber-850 to-amber-900 hover:from-amber-950 hover:to-amber-900 text-amber-100 font-serif font-bold text-xs flex items-center justify-center gap-2 shadow-md hover:shadow-lg transition cursor-pointer border border-[#ecd38c]/40 group"
                            >
                              <Feather className="w-3.5 h-3.5 text-amber-300 group-hover:rotate-12 transition transform" />
                              <span>Turn to Next Page & Inscribe Chapter {toRoman(chapters.length + 1)} ➔</span>
                            </button>
                          )}
                        </div>
                      )}

                      {/* Right Page Baseline Info */}
                      <div className="mt-2.5 pt-2 border-t border-amber-950/15 flex items-center justify-between text-[11px] font-serif text-amber-950/70">
                        <div className="flex items-center gap-2">
                          <span>Folio {viewingChapter} &bull; Manuscript Output</span>
                          {tokensPerSec > 0 && (
                            <span className="font-mono text-[10px] text-amber-900/60">
                              &bull; {tokensPerSec} tok/s
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-2 font-mono text-[10px] text-amber-900/60">
                          <span>Model: Karna-49M (GQA 8:4)</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </article>
              </div>

              {/* Folio Bottom Page Turner Ribbon Bar */}
              <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 bg-amber-950/10 border-t border-amber-950/20 text-xs font-serif text-amber-950 select-none rounded-b-md">
                {/* Prev Folio Button */}
                <button
                  onClick={() => setViewingChapter((p) => Math.max(1, p - 1))}
                  disabled={viewingChapter <= 1 || isCasting}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-white/70 hover:bg-white text-amber-950 border border-amber-950/20 disabled:opacity-30 disabled:pointer-events-none transition cursor-pointer font-serif font-medium shadow-xs"
                >
                  <ChevronLeft className="w-4 h-4 text-amber-800" />
                  <span>&larr; Prev Folio (Ch. {toRoman(Math.max(1, viewingChapter - 1))})</span>
                </button>

                {/* Folio / Chapter Indicators */}
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] font-mono uppercase text-amber-900/70 font-semibold mr-1 hidden sm:inline">
                    Folios ({totalFolios}):
                  </span>
                  {Array.from({ length: totalFolios }).map((_, idx) => {
                    const chNum = idx + 1;
                    const isCurrent = chNum === viewingChapter;
                    const hasStory = Boolean(chapters[idx]?.oracleResponse);
                    return (
                      <button
                        key={chNum}
                        onClick={() => setViewingChapter(chNum)}
                        disabled={isCasting}
                        className={`px-2.5 py-1 rounded text-xs font-serif transition cursor-pointer flex items-center gap-1 ${
                          isCurrent
                            ? "bg-amber-900 text-amber-100 font-bold shadow-xs border border-amber-950"
                            : hasStory
                            ? "bg-white/80 hover:bg-white text-amber-950 border border-amber-950/20"
                            : "border border-dashed border-amber-900/50 bg-amber-900/5 text-amber-900/80 hover:bg-amber-900/10"
                        }`}
                        title={`Chapter ${toRoman(chNum)} (Pages ${(chNum - 1) * 2 + 1}-${(chNum - 1) * 2 + 2})`}
                      >
                        <span>Ch. {toRoman(chNum)}</span>
                        {hasStory ? (
                          <span className="text-[9px] text-emerald-800 font-bold">✓</span>
                        ) : (
                          <span className="text-[9px] text-amber-800 font-bold">✎</span>
                        )}
                      </button>
                    );
                  })}
                </div>

                {/* Next Folio Button */}
                <button
                  onClick={() => setViewingChapter((p) => Math.min(totalFolios, p + 1))}
                  disabled={viewingChapter >= totalFolios || isCasting}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-white/70 hover:bg-white text-amber-950 border border-amber-950/20 disabled:opacity-30 disabled:pointer-events-none transition cursor-pointer font-serif font-medium shadow-xs"
                >
                  <span>Next Folio (Ch. {toRoman(Math.min(totalFolios, viewingChapter + 1))}) &rarr;</span>
                  <ChevronRight className="w-4 h-4 text-amber-800" />
                </button>
              </div>

              {/* Realistic Page Edges under the book */}
              <div className="page-edge-layers" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      </div>
    </div>
  );
}
