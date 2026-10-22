"use client";

import React, { useRef, useEffect, useState, useMemo } from "react";
import * as THREE from "three";
import {
  ArrowUp,
  ArrowDown,
  Sparkles,
  BookOpen,
  Compass,
  Layers,
  ChevronRight,
  Info,
  RotateCcw,
  Zap,
  HelpCircle
} from "lucide-react";

export interface StoryChapter {
  prompt: string;
  story: string;
  timestamp?: string;
  tokensPerSec?: number;
}

export interface ThreeBookData {
  id: string;
  side: "left" | "right";
  tier: number; // 1 to 4
  slot: number; // 1 to 3
  title: string;
  subtitle: string;
  genre: string;
  leatherColor: string;
  emblem: string;
  isInscribed: boolean;
  chapters?: StoryChapter[];
  createdAt?: string;
  meshRef?: THREE.Mesh;
}

// Generate the 24 initial books (12 on Left across 4 shelves, 12 on Right across 4 shelves)
export function generate24Books(): ThreeBookData[] {
  const books: ThreeBookData[] = [];

  const leftGenres = [
    { name: "Epic Mythology", color: "#6a1b24", emblem: "☀️", title: "The Legend of Karna", sub: "Suryaputra & Solar Armor", inscribed: true },
    { name: "High Fantasy", color: "#1e4d2b", emblem: "🌿", title: "Chronicles of Eldoria", sub: "Runes of the Whispering Oak", inscribed: true },
    { name: "Gothic Mystery", color: "#24334f", emblem: "🗝️", title: "Whispering Shadows", sub: "Midnight Chimes & Silent Manors", inscribed: true },
    { name: "Ancient History", color: "#3d4e3e", emblem: "🏛️", title: "Empire of Ashoka", sub: "Edicts of Sandstone", inscribed: true }
  ];

  const rightGenres = [
    { name: "Cyberpunk Noir", color: "#194f55", emblem: "⚡", title: "Cyberpunk 2099", sub: "Neon Syndicate & Neural Ghosts", inscribed: true },
    { name: "Animal Fables", color: "#7a4c1c", emblem: "🐾", title: "Whispering Woods", sub: "Pip and the Golden Star", inscribed: true },
    { name: "Cosmic Odyssey", color: "#45225c", emblem: "🌌", title: "Echoes of the Void", sub: "Starlight upon dead worlds", inscribed: false },
    { name: "Dark Sorcery", color: "#2f3136", emblem: "👁️", title: "The Black Grimoire", sub: "Spells of the Nether Realm", inscribed: false }
  ];

  // Palette of warm vintage leather bindings for empty/unwritten tomes in light library
  const emptyTomePalette = [
    "#8a5a36", // Vintage saddle leather
    "#9e683f", // Warm caramel
    "#593b26", // Deep walnut
    "#3e4a5b", // Slate blue
    "#344b3c", // Forest moss
    "#73383e"  // Muted burgundy
  ];

  // 4 Shelves on Left, each having 3 books = 12
  for (let tier = 1; tier <= 4; tier++) {
    const tierTemplate = leftGenres[tier - 1];
    for (let slot = 1; slot <= 3; slot++) {
      const isFirst = slot === 1 && tierTemplate.inscribed;
      const paletteIdx = ((tier - 1) * 3 + slot) % emptyTomePalette.length;
      books.push({
        id: `L-T${tier}-S${slot}`,
        side: "left",
        tier,
        slot,
        title: isFirst ? tierTemplate.title : `Unbound Tome L${tier}-${slot}`,
        subtitle: isFirst ? tierTemplate.sub : "Empty parchment awaiting ink",
        genre: isFirst ? tierTemplate.name : "Unwritten Lore",
        leatherColor: isFirst ? tierTemplate.color : emptyTomePalette[paletteIdx],
        emblem: isFirst ? tierTemplate.emblem : "✧",
        isInscribed: isFirst
      });
    }
  }

  // 4 Shelves on Right, each having 3 books = 12
  for (let tier = 1; tier <= 4; tier++) {
    const tierTemplate = rightGenres[tier - 1];
    for (let slot = 1; slot <= 3; slot++) {
      const isFirst = slot === 1 && tierTemplate.inscribed;
      const paletteIdx = ((tier + 2) * 3 + slot) % emptyTomePalette.length;
      books.push({
        id: `R-T${tier}-S${slot}`,
        side: "right",
        tier,
        slot,
        title: isFirst ? tierTemplate.title : `Unbound Tome R${tier}-${slot}`,
        subtitle: isFirst ? tierTemplate.sub : "Empty parchment awaiting ink",
        genre: isFirst ? tierTemplate.name : "Unwritten Lore",
        leatherColor: isFirst ? tierTemplate.color : emptyTomePalette[paletteIdx],
        emblem: isFirst ? tierTemplate.emblem : "✧",
        isInscribed: isFirst
      });
    }
  }

  return books;
}

export interface ThreeLibraryCorridorProps {
  books?: ThreeBookData[];
  onSelectBook: (book: ThreeBookData) => void;
  onNavigateToChat?: () => void;
  sessionsSummary?: Record<string, { messageCount: number; lastSnippet?: string }>;
}

export default function ThreeLibraryCorridor({
  books = [],
  onSelectBook,
  onNavigateToChat,
  sessionsSummary
}: ThreeLibraryCorridorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mountRef = useRef<HTMLDivElement>(null);

  const booksList = useMemo(() => books, [books]);
  const [hoveredBook, setHoveredBook] = useState<ThreeBookData | null>(null);

  // References for Three.js render loop and interaction
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const bookMeshesRef = useRef<Map<string, { mesh: THREE.Mesh; originalPos: THREE.Vector3; bookData: ThreeBookData }>>(new Map());
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2(-999, -999));
  const targetCameraZ = useRef<number>(5.5);

  // Smooth camera navigation helpers
  const moveForward = () => {
    targetCameraZ.current = Math.max(targetCameraZ.current - 2.5, -2.5);
  };

  const moveBackward = () => {
    targetCameraZ.current = Math.min(targetCameraZ.current + 2.5, 8.5);
  };

  const resetCamera = () => {
    targetCameraZ.current = 5.5;
  };

  // Keyboard controls for moving in 3D
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "w" || e.key === "ArrowUp") moveForward();
      if (e.key === "s" || e.key === "ArrowDown") moveBackward();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Initialize Three.js Scene (runs ONCE on mount for silky-smooth 60fps performance)
  useEffect(() => {
    if (!mountRef.current) return;

    const mount = mountRef.current;
    while (mount.firstChild) {
      mount.removeChild(mount.firstChild);
    }

    const width = mount.clientWidth || 800;
    const height = mount.clientHeight || 620;

    // 1. Scene & Atmospheric Sunlit Morning Mist
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0xf5ede0, 0.014); // Soft distant atmosphere
    sceneRef.current = scene;

    // 2. Perspective Camera looking down the library corridor aisle
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 60);
    camera.position.set(0, 2.2, targetCameraZ.current);
    camera.lookAt(0, 2.4, -15.8); // Framed on the cathedral rose window at the end of the aisle
    cameraRef.current = camera;

    // 3. WebGL Renderer with High Quality Shadows & Tone Mapping
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        powerPreference: "high-performance"
      });
    } catch (e) {
      console.error("Failed to initialize WebGLRenderer:", e);
      return;
    }

    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
    renderer.setClearColor(0xf7f3ea, 1);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    rendererRef.current = renderer;

    const canvas = renderer.domElement;
    canvas.className = "w-full h-full block cursor-pointer";
    mount.appendChild(canvas);

    // 4. Lights: Sunlit Skylight & Hall Daylights
    const ambientLight = new THREE.AmbientLight(0xfff8ed, 1.25);
    scene.add(ambientLight);

    // Warm Sunbeam from library clerestory windows (single fast directional shadow map)
    const sunLight = new THREE.DirectionalLight(0xffedd5, 1.6);
    sunLight.position.set(4, 9, 6);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 1024;
    sunLight.shadow.mapSize.height = 1024;
    sunLight.shadow.bias = -0.0005;
    scene.add(sunLight);

    // Warm Brass Lantern along Left Wall (castShadow false to eliminate 6-face cubemap overhead)
    const leftLantern = new THREE.PointLight(0xffc56e, 1.5, 16, 1.1);
    leftLantern.position.set(-2.5, 3.8, 1.5);
    leftLantern.castShadow = false;
    scene.add(leftLantern);

    // Warm Brass Lantern along Right Wall (castShadow false to eliminate 6-face cubemap overhead)
    const rightLantern = new THREE.PointLight(0xffc56e, 1.5, 16, 1.1);
    rightLantern.position.set(2.5, 3.8, 1.5);
    rightLantern.castShadow = false;
    scene.add(rightLantern);

    // Distance Nave Glow
    const distanceSconce = new THREE.PointLight(0xfcd34d, 1.2, 22, 1.3);
    distanceSconce.position.set(0, 3.5, -6);
    scene.add(distanceSconce);

    // =========================================================================
    // 5. GRAND ARCHITECTURAL GAME ENVIRONMENT (HALL, WALLS, CEILING, ARCHES, WINDOWS)
    // =========================================================================

    // Shared Architectural Materials
    const woodOakDarkMat = new THREE.MeshStandardMaterial({ color: 0x543015, roughness: 0.65, metalness: 0.1 });
    const woodOakMediumMat = new THREE.MeshStandardMaterial({ color: 0x7a4722, roughness: 0.6, metalness: 0.1 });
    const wallStoneMat = new THREE.MeshStandardMaterial({ color: 0xded2bf, roughness: 0.85, metalness: 0.02 });
    const ceilingBeamMat = new THREE.MeshStandardMaterial({ color: 0x381f10, roughness: 0.75 });
    const goldTrimMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, roughness: 0.35, metalness: 0.65 });
    const brassMat = new THREE.MeshStandardMaterial({ color: 0xc8963e, roughness: 0.3, metalness: 0.8 });

    // A. FLOOR & GOLD-EMBROIDERED CARPET RUNNER
    // Honey-Parquet Wooden Floor
    const floorGeo = new THREE.PlaneGeometry(16, 40);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0xd6b388,
      roughness: 0.55,
      metalness: 0.05
    });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.position.y = 0;
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    // Royal Crimson Runner Carpet in Center Aisle
    const carpetGeo = new THREE.PlaneGeometry(2.4, 38);
    const carpetMat = new THREE.MeshStandardMaterial({
      color: 0x8a2430,
      roughness: 0.85
    });
    const carpetMesh = new THREE.Mesh(carpetGeo, carpetMat);
    carpetMesh.rotation.x = -Math.PI / 2;
    carpetMesh.position.set(0, 0.01, 0);
    carpetMesh.receiveShadow = true;
    scene.add(carpetMesh);

    // Gold Leaf Border Trim Strips along Carpet
    const goldStripGeo = new THREE.PlaneGeometry(0.06, 38);
    const leftGoldStrip = new THREE.Mesh(goldStripGeo, goldTrimMat);
    leftGoldStrip.rotation.x = -Math.PI / 2;
    leftGoldStrip.position.set(-1.18, 0.015, 0);
    scene.add(leftGoldStrip);

    const rightGoldStrip = new THREE.Mesh(goldStripGeo, goldTrimMat);
    rightGoldStrip.rotation.x = -Math.PI / 2;
    rightGoldStrip.position.set(1.18, 0.015, 0);
    scene.add(rightGoldStrip);

    // B. CORRIDOR WALLS (LEFT & RIGHT ARCHITECTURAL WALLS)
    const wallLength = 36;

    // LEFT WALL (x = -4.2)
    // Lower Wood Wainscoting
    const wainscotGeo = new THREE.PlaneGeometry(wallLength, 2.2);
    const leftWainscot = new THREE.Mesh(wainscotGeo, woodOakDarkMat);
    leftWainscot.position.set(-4.2, 1.1, -2);
    leftWainscot.rotation.y = Math.PI / 2;
    leftWainscot.receiveShadow = true;
    scene.add(leftWainscot);

    // Upper Stone Cathedral Wall
    const upperWallGeo = new THREE.PlaneGeometry(wallLength, 4.6);
    const leftUpperWall = new THREE.Mesh(upperWallGeo, wallStoneMat);
    leftUpperWall.position.set(-4.2, 4.5, -2);
    leftUpperWall.rotation.y = Math.PI / 2;
    leftUpperWall.receiveShadow = true;
    scene.add(leftUpperWall);

    // Chair rail & baseboard along Left Wall
    const trimRailGeo = new THREE.BoxGeometry(0.12, 0.1, wallLength);
    const leftRail = new THREE.Mesh(trimRailGeo, woodOakMediumMat);
    leftRail.position.set(-4.15, 2.2, -2);
    scene.add(leftRail);

    const skirtingGeo = new THREE.BoxGeometry(0.14, 0.22, wallLength);
    const leftSkirting = new THREE.Mesh(skirtingGeo, woodOakDarkMat);
    leftSkirting.position.set(-4.14, 0.11, -2);
    scene.add(leftSkirting);

    // RIGHT WALL (x = +4.2)
    const rightWainscot = new THREE.Mesh(wainscotGeo, woodOakDarkMat);
    rightWainscot.position.set(4.2, 1.1, -2);
    rightWainscot.rotation.y = -Math.PI / 2;
    rightWainscot.receiveShadow = true;
    scene.add(rightWainscot);

    const rightUpperWall = new THREE.Mesh(upperWallGeo, wallStoneMat);
    rightUpperWall.position.set(4.2, 4.5, -2);
    rightUpperWall.rotation.y = -Math.PI / 2;
    rightUpperWall.receiveShadow = true;
    scene.add(rightUpperWall);

    const rightRail = new THREE.Mesh(trimRailGeo, woodOakMediumMat);
    rightRail.position.set(4.15, 2.2, -2);
    scene.add(rightRail);

    const rightSkirting = new THREE.Mesh(skirtingGeo, woodOakDarkMat);
    rightSkirting.position.set(4.14, 0.11, -2);
    scene.add(rightSkirting);

    // C. VAULTED CEILING & TIMBER CROSSBEAMS
    const ceilingGeo = new THREE.PlaneGeometry(8.4, wallLength);
    const ceilingMat = new THREE.MeshStandardMaterial({ color: 0x361e0e, roughness: 0.8 });
    const ceilingMesh = new THREE.Mesh(ceilingGeo, ceilingMat);
    ceilingMesh.position.set(0, 6.4, -2);
    ceilingMesh.rotation.x = Math.PI / 2;
    scene.add(ceilingMesh);

    // Gothic Overhead Crossbeams & Wall Pilasters (Spaced along corridor)
    const beamZPositions = [-14, -9.5, -5, -0.5, 4, 8.5];
    beamZPositions.forEach((bz) => {
      // Horizontal Crossbeam across ceiling
      const beamGeo = new THREE.BoxGeometry(8.4, 0.35, 0.4);
      const beamMesh = new THREE.Mesh(beamGeo, ceilingBeamMat);
      beamMesh.position.set(0, 6.2, bz);
      scene.add(beamMesh);

      // Left Wall Stone Pilaster / Column
      const pilasterGeo = new THREE.BoxGeometry(0.35, 6.4, 0.45);
      const leftPilaster = new THREE.Mesh(pilasterGeo, wallStoneMat);
      leftPilaster.position.set(-4.05, 3.2, bz);
      leftPilaster.receiveShadow = true;
      scene.add(leftPilaster);

      // Right Wall Stone Pilaster / Column
      const rightPilaster = new THREE.Mesh(pilasterGeo, wallStoneMat);
      rightPilaster.position.set(4.05, 3.2, bz);
      rightPilaster.receiveShadow = true;
      scene.add(rightPilaster);

      // Left Arch Support Strut
      const strutGeo = new THREE.BoxGeometry(0.25, 1.2, 0.3);
      const leftStrut = new THREE.Mesh(strutGeo, ceilingBeamMat);
      leftStrut.position.set(-3.4, 5.7, bz);
      leftStrut.rotation.z = Math.PI / 4;
      scene.add(leftStrut);

      // Right Arch Support Strut
      const rightStrut = new THREE.Mesh(strutGeo, ceilingBeamMat);
      rightStrut.position.set(3.4, 5.7, bz);
      rightStrut.rotation.z = -Math.PI / 4;
      scene.add(rightStrut);
    });

    // D. GRAND END-OF-HALL SANCTUARY & CATHEDRAL ROSE/ARCHED WINDOW (z = -15.8)
    // Back Stone Wall
    const backWallGeo = new THREE.PlaneGeometry(8.4, 6.4);
    const backWallMesh = new THREE.Mesh(backWallGeo, wallStoneMat);
    backWallMesh.position.set(0, 3.2, -15.8);
    backWallMesh.receiveShadow = true;
    scene.add(backWallMesh);

    // Glowing Cathedral Sunlit Glass Panel
    const windowGlassGeo = new THREE.PlaneGeometry(3.6, 4.4);
    const windowGlassMat = new THREE.MeshStandardMaterial({
      color: 0xfffae8,
      emissive: 0xffe6ab,
      emissiveIntensity: 0.9,
      roughness: 0.15
    });
    const windowGlass = new THREE.Mesh(windowGlassGeo, windowGlassMat);
    windowGlass.position.set(0, 3.6, -15.75);
    scene.add(windowGlass);

    // Window Mullions (Gothic Stone Tracery & Crossbars)
    const mullionMat = new THREE.MeshStandardMaterial({ color: 0x3d281a, roughness: 0.7 });
    [-0.9, 0, 0.9].forEach((mx) => {
      const vMullion = new THREE.Mesh(new THREE.BoxGeometry(0.08, 4.4, 0.1), mullionMat);
      vMullion.position.set(mx, 3.6, -15.72);
      scene.add(vMullion);
    });
    [2.2, 3.6, 4.8].forEach((my) => {
      const hTransom = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.08, 0.1), mullionMat);
      hTransom.position.set(0, my, -15.72);
      scene.add(hTransom);
    });
    const archRing = new THREE.Mesh(new THREE.TorusGeometry(1.2, 0.08, 8, 24), mullionMat);
    archRing.position.set(0, 4.2, -15.71);
    scene.add(archRing);

    // E. VOLUMETRIC SUNLIGHT GOD-RAYS (Cinematic Game Light Beams)
    const beamGeo = new THREE.PlaneGeometry(3.2, 14.0);
    const beamMat = new THREE.MeshBasicMaterial({
      color: 0xfff0cc,
      transparent: true,
      opacity: 0.15,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const sunbeam = new THREE.Mesh(beamGeo, beamMat);
    sunbeam.position.set(0, 2.6, -8.5);
    sunbeam.rotation.x = Math.PI / 4.5;
    scene.add(sunbeam);

    // F. HANGING BRASS CHANDELIERS WITH CANDLES
    const candleFlames: THREE.Mesh[] = [];
    [-2.0, -8.5].forEach((cz) => {
      const chain = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1.8), brassMat);
      chain.position.set(0, 5.2, cz);
      scene.add(chain);

      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.75, 0.04, 8, 24), brassMat);
      ring.position.set(0, 4.3, cz);
      ring.rotation.x = Math.PI / 2;
      scene.add(ring);

      for (let i = 0; i < 6; i++) {
        const angle = (i / 6) * Math.PI * 2;
        const cx = Math.cos(angle) * 0.75;
        const czPos = cz + Math.sin(angle) * 0.75;

        const candleBody = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.16), new THREE.MeshStandardMaterial({ color: 0xfffae8 }));
        candleBody.position.set(cx, 4.4, czPos);
        scene.add(candleBody);

        const flameMat = new THREE.MeshBasicMaterial({ color: 0xffb732 });
        const flame = new THREE.Mesh(new THREE.SphereGeometry(0.035, 6, 6), flameMat);
        flame.position.set(cx, 4.52, czPos);
        scene.add(flame);
        candleFlames.push(flame);
      }
    });

    // G. MAGICAL FLOATING DUST MOTES (Particles shimmering in sunlight)
    const dustCount = 220;
    const dustGeo = new THREE.BufferGeometry();
    const dustPos = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount; i++) {
      dustPos[i * 3] = (Math.random() - 0.5) * 6.5;
      dustPos[i * 3 + 1] = Math.random() * 4.8 + 0.6;
      dustPos[i * 3 + 2] = Math.random() * 24 - 15;
    }
    dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
    const dustMat = new THREE.PointsMaterial({
      color: 0xffe6a8,
      size: 0.06,
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending
    });
    const dustParticles = new THREE.Points(dustGeo, dustMat);
    scene.add(dustParticles);

    // H. ARCANE RUNIC GLYPH HALO (Spins around active hovered book)
    const runeHaloGroup = new THREE.Group();
    const runeRingGeo = new THREE.TorusGeometry(0.44, 0.015, 8, 36);
    const runeRingMat = new THREE.MeshStandardMaterial({
      color: 0xffe066,
      emissive: 0xd4af37,
      emissiveIntensity: 0.9,
      roughness: 0.2,
      metalness: 0.8,
      transparent: true,
      opacity: 0
    });
    const runeRingMesh = new THREE.Mesh(runeRingGeo, runeRingMat);
    runeHaloGroup.add(runeRingMesh);

    const innerRingGeo = new THREE.RingGeometry(0.3, 0.33, 24);
    const innerRingMat = new THREE.MeshBasicMaterial({
      color: 0xffd700,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending
    });
    const innerRingMesh = new THREE.Mesh(innerRingGeo, innerRingMat);
    runeHaloGroup.add(innerRingMesh);

    runeHaloGroup.visible = false;
    scene.add(runeHaloGroup);

    // =========================================================================
    // 6. BUILD THE MASTER-CRAFTED BOOKCASE CABINET UNITS (LEFT & RIGHT)
    // =========================================================================
    const shelfHeights = [0.8, 1.7, 2.6, 3.5];
    const shelfWidth = 2.6;
    const shelfDepth = 0.8;
    const shelfThickness = 0.12;

    const shelfMat = new THREE.MeshStandardMaterial({
      color: 0x9e6433, // Warm Honey Oak
      roughness: 0.6,
      metalness: 0.1
    });

    // Left Bookcase Cabinet Enclosure (Backboard, Side Uprights, Plinth, Cornice)
    const bookcaseBackGeo = new THREE.BoxGeometry(0.06, 4.3, 2.76);
    const leftBackboard = new THREE.Mesh(bookcaseBackGeo, woodOakDarkMat);
    leftBackboard.position.set(-2.98, 2.15, 0.2);
    leftBackboard.receiveShadow = true;
    scene.add(leftBackboard);

    const uprightGeo = new THREE.BoxGeometry(shelfDepth + 0.06, 4.3, 0.09);
    const leftUprightNorth = new THREE.Mesh(uprightGeo, woodOakDarkMat);
    leftUprightNorth.position.set(-2.6, 2.15, 0.2 - 1.35);
    scene.add(leftUprightNorth);

    const leftUprightSouth = new THREE.Mesh(uprightGeo, woodOakDarkMat);
    leftUprightSouth.position.set(-2.6, 2.15, 0.2 + 1.35);
    scene.add(leftUprightSouth);

    const crownGeo = new THREE.BoxGeometry(shelfDepth + 0.14, 0.26, 2.9);
    const leftCrown = new THREE.Mesh(crownGeo, woodOakMediumMat);
    leftCrown.position.set(-2.6, 4.25, 0.2);
    scene.add(leftCrown);

    const plinthGeo = new THREE.BoxGeometry(shelfDepth + 0.1, 0.2, 2.84);
    const leftPlinth = new THREE.Mesh(plinthGeo, woodOakDarkMat);
    leftPlinth.position.set(-2.6, 0.1, 0.2);
    scene.add(leftPlinth);

    // Right Bookcase Cabinet Enclosure (Backboard, Side Uprights, Plinth, Cornice)
    const rightBackboard = new THREE.Mesh(bookcaseBackGeo, woodOakDarkMat);
    rightBackboard.position.set(2.98, 2.15, 0.2);
    rightBackboard.receiveShadow = true;
    scene.add(rightBackboard);

    const rightUprightNorth = new THREE.Mesh(uprightGeo, woodOakDarkMat);
    rightUprightNorth.position.set(2.6, 2.15, 0.2 - 1.35);
    scene.add(rightUprightNorth);

    const rightUprightSouth = new THREE.Mesh(uprightGeo, woodOakDarkMat);
    rightUprightSouth.position.set(2.6, 2.15, 0.2 + 1.35);
    scene.add(rightUprightSouth);

    const rightCrown = new THREE.Mesh(crownGeo, woodOakMediumMat);
    rightCrown.position.set(2.6, 4.25, 0.2);
    scene.add(rightCrown);

    const rightPlinth = new THREE.Mesh(plinthGeo, woodOakDarkMat);
    rightPlinth.position.set(2.6, 0.1, 0.2);
    scene.add(rightPlinth);

    const bookMeshesMap = new Map<string, { mesh: THREE.Mesh; originalPos: THREE.Vector3; bookData: ThreeBookData }>();

    // Shared geometry for book tomes standing upright
    const bookGeo = new THREE.BoxGeometry(0.24, 0.68, 0.52);

    // Build Left Shelf Structure (x = -2.6)
    shelfHeights.forEach((shelfY, tierIdx) => {
      const tierNum = tierIdx + 1;

      // Wooden Plank Mesh
      const plankGeo = new THREE.BoxGeometry(shelfDepth, shelfThickness, shelfWidth);
      const plankMesh = new THREE.Mesh(plankGeo, shelfMat);
      plankMesh.position.set(-2.6, shelfY, 0.2);
      plankMesh.receiveShadow = true;
      plankMesh.castShadow = true;
      scene.add(plankMesh);

      // Place 3 books standing upright on this shelf tier
      // Slots 1, 2, 3 placed along Z axis: z = -0.65, 0.15, 0.95
      const zOffsets = [-0.65, 0.15, 0.95];
      zOffsets.forEach((slotZ, slotIdx) => {
        const slotNum = slotIdx + 1;
        const bookData = booksList.find((b) => b.side === "left" && b.tier === tierNum && b.slot === slotNum);
        if (!bookData) return;

        // Custom leather material
        const tomeMat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(bookData.leatherColor),
          roughness: 0.55,
          metalness: bookData.isInscribed ? 0.35 : 0.05
        });

        const tomeMesh = new THREE.Mesh(bookGeo, tomeMat);
        const posX = -2.55;
        const posY = shelfY + shelfThickness / 2 + 0.34;
        const posZ = slotZ;

        tomeMesh.position.set(posX, posY, posZ);
        tomeMesh.rotation.y = Math.PI / 2; // Facing the center aisle
        tomeMesh.castShadow = true;
        tomeMesh.receiveShadow = true;

        // Add small golden foil spine ribbon
        if (bookData.isInscribed) {
          const foilGeo = new THREE.BoxGeometry(0.04, 0.55, 0.02);
          const foilMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.8, roughness: 0.2 });
          const foilMesh = new THREE.Mesh(foilGeo, foilMat);
          foilMesh.position.set(0, 0, 0.26);
          tomeMesh.add(foilMesh);
        }

        scene.add(tomeMesh);
        bookMeshesMap.set(bookData.id, {
          mesh: tomeMesh,
          originalPos: new THREE.Vector3(posX, posY, posZ),
          bookData
        });
      });
    });

    // Build Right Shelf Structure (x = +2.6)
    shelfHeights.forEach((shelfY, tierIdx) => {
      const tierNum = tierIdx + 1;

      // Wooden Plank Mesh
      const plankGeo = new THREE.BoxGeometry(shelfDepth, shelfThickness, shelfWidth);
      const plankMesh = new THREE.Mesh(plankGeo, shelfMat);
      plankMesh.position.set(2.6, shelfY, 0.2);
      plankMesh.receiveShadow = true;
      plankMesh.castShadow = true;
      scene.add(plankMesh);

      // Place 3 books standing upright on this shelf tier
      const zOffsets = [-0.65, 0.15, 0.95];
      zOffsets.forEach((slotZ, slotIdx) => {
        const slotNum = slotIdx + 1;
        const bookData = booksList.find((b) => b.side === "right" && b.tier === tierNum && b.slot === slotNum);
        if (!bookData) return;

        const tomeMat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(bookData.leatherColor),
          roughness: 0.55,
          metalness: bookData.isInscribed ? 0.35 : 0.05
        });

        const tomeMesh = new THREE.Mesh(bookGeo, tomeMat);
        const posX = 2.55;
        const posY = shelfY + shelfThickness / 2 + 0.34;
        const posZ = slotZ;

        tomeMesh.position.set(posX, posY, posZ);
        tomeMesh.rotation.y = -Math.PI / 2; // Facing the center aisle
        tomeMesh.castShadow = true;
        tomeMesh.receiveShadow = true;

        if (bookData.isInscribed) {
          const foilGeo = new THREE.BoxGeometry(0.04, 0.55, 0.02);
          const foilMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.8, roughness: 0.2 });
          const foilMesh = new THREE.Mesh(foilGeo, foilMat);
          foilMesh.position.set(0, 0, 0.26);
          tomeMesh.add(foilMesh);
        }

        scene.add(tomeMesh);
        bookMeshesMap.set(bookData.id, {
          mesh: tomeMesh,
          originalPos: new THREE.Vector3(posX, posY, posZ),
          bookData
        });
      });
    });

    bookMeshesRef.current = bookMeshesMap;

    // Pre-cache an array of interactive meshes once (avoids Array.from allocations on every frame)
    const interactiveMeshes: THREE.Mesh[] = [];
    bookMeshesMap.forEach((entry) => interactiveMeshes.push(entry.mesh));

    // Dirty flags and throttles for 60fps performance
    let mouseMoved = false;
    let lastRaycast = 0;
    const hoveredBookIdRef = { current: null as string | null };

    // 7. Mouse Interaction Raycasting
    const onPointerMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      mouseMoved = true;
    };

    const onClick = () => {
      if (!cameraRef.current) return;
      raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
      const intersects = raycasterRef.current.intersectObjects(interactiveMeshes, false);

      if (intersects.length > 0) {
        const clickedMesh = intersects[0].object as THREE.Mesh;
        for (const entry of bookMeshesMap.values()) {
          if (entry.mesh === clickedMesh) {
            onSelectBook(entry.bookData);
            break;
          }
        }
      }
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.deltaY > 0) {
        targetCameraZ.current = Math.min(targetCameraZ.current + 0.8, 8.5);
      } else {
        targetCameraZ.current = Math.max(targetCameraZ.current - 0.8, -2.5);
      }
    };

    canvas.addEventListener("mousemove", onPointerMove, { passive: true });
    canvas.addEventListener("click", onClick);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    // 8. Animation Render Loop with Smooth Camera & Hover Physics
    let animationId: number;
    let hoveredMeshRecord: { mesh: THREE.Mesh; originalPos: THREE.Vector3; bookData: ThreeBookData } | null = null;

    const animate = (now: number) => {
      animationId = requestAnimationFrame(animate);
      const currentTime = typeof now === "number" && !isNaN(now) ? now : performance.now();

      // Smooth camera interpolation
      let cameraMoving = false;
      if (cameraRef.current) {
        const diff = targetCameraZ.current - cameraRef.current.position.z;
        if (Math.abs(diff) > 0.002) {
          cameraRef.current.position.z += diff * 0.1;
          cameraMoving = true;
        }
      }

      // Throttled Raycast: runs ONLY when mouse or camera moved, max ~33Hz to save GPU/CPU
      if (cameraRef.current && (mouseMoved || cameraMoving) && currentTime - lastRaycast > 30) {
        lastRaycast = currentTime;
        mouseMoved = false;

        raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
        const intersects = raycasterRef.current.intersectObjects(interactiveMeshes, false);

        if (intersects.length > 0) {
          const hitMesh = intersects[0].object as THREE.Mesh;
          let foundRecord: { mesh: THREE.Mesh; originalPos: THREE.Vector3; bookData: ThreeBookData } | null = null;
          for (const entry of bookMeshesMap.values()) {
            if (entry.mesh === hitMesh) {
              foundRecord = entry;
              break;
            }
          }

          if (foundRecord && foundRecord !== hoveredMeshRecord) {
            // Restore previous mesh
            if (hoveredMeshRecord) {
              hoveredMeshRecord.mesh.position.copy(hoveredMeshRecord.originalPos);
              (hoveredMeshRecord.mesh.material as THREE.MeshStandardMaterial).emissive.setHex(0x000000);
            }
            // Slide current hovered book slightly out toward the center aisle
            const slideDir = foundRecord.bookData.side === "left" ? 0.35 : -0.35;
            foundRecord.mesh.position.x = foundRecord.originalPos.x + slideDir;
            (foundRecord.mesh.material as THREE.MeshStandardMaterial).emissive.setHex(0x553300);
            hoveredMeshRecord = foundRecord;

            if (hoveredBookIdRef.current !== foundRecord.bookData.id) {
              hoveredBookIdRef.current = foundRecord.bookData.id;
              setHoveredBook(foundRecord.bookData);
            }
          }
        } else {
          if (hoveredMeshRecord) {
            hoveredMeshRecord.mesh.position.copy(hoveredMeshRecord.originalPos);
            (hoveredMeshRecord.mesh.material as THREE.MeshStandardMaterial).emissive.setHex(0x000000);
            hoveredMeshRecord = null;
            if (hoveredBookIdRef.current !== null) {
              hoveredBookIdRef.current = null;
              setHoveredBook(null);
            }
          }
        }
      }

      // Living Candle Flame Physics (organic flicker)
      for (let i = 0; i < candleFlames.length; i++) {
        const flame = candleFlames[i];
        const flicker = 1.0 + Math.sin(currentTime * 0.012 + i * 0.8) * 0.18 + Math.cos(currentTime * 0.024 + i) * 0.08;
        flame.scale.set(flicker, flicker * 1.25, flicker);
      }

      // Soft breathing warmth on wall lanterns
      const lanternFlicker = Math.sin(currentTime * 0.007) * 0.06;
      leftLantern.intensity = 1.5 + lanternFlicker;
      rightLantern.intensity = 1.5 - lanternFlicker;

      // Animate Arcane Runic Glyph Halo around the hovered book
      if (hoveredMeshRecord) {
        const bookMesh = hoveredMeshRecord.mesh;
        const isLeft = hoveredMeshRecord.bookData.side === "left";
        runeHaloGroup.position.set(
          bookMesh.position.x + (isLeft ? 0.09 : -0.09),
          bookMesh.position.y,
          bookMesh.position.z
        );
        runeHaloGroup.rotation.y = Math.PI / 2;
        runeRingMesh.rotation.z += 0.022;
        innerRingMesh.rotation.z -= 0.016;

        runeHaloGroup.visible = true;
        if (runeRingMat.opacity < 0.92) runeRingMat.opacity += 0.08;
        if (innerRingMat.opacity < 0.75) innerRingMat.opacity += 0.06;
      } else {
        if (runeRingMat.opacity > 0.02) {
          runeRingMat.opacity -= 0.08;
          innerRingMat.opacity -= 0.08;
          runeRingMesh.rotation.z += 0.022;
          innerRingMesh.rotation.z -= 0.016;
        } else {
          runeHaloGroup.visible = false;
        }
      }

      // Shimmer floating dust motes
      dustParticles.rotation.y += 0.0003;
      dustParticles.position.y = Math.sin(currentTime * 0.0008) * 0.04;

      if (cameraRef.current) {
        try {
          renderer.render(scene, cameraRef.current);
        } catch {
          // Render fail-safe
        }
      }
    };

    animationId = requestAnimationFrame(animate);

    // 9. Resize Listener
    const onResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      if (!w || !h) return;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };

    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", onResize);
      canvas.removeEventListener("mousemove", onPointerMove);
      canvas.removeEventListener("click", onClick);
      canvas.removeEventListener("wheel", onWheel);
      if (mount && mount.contains(canvas)) {
        mount.removeChild(canvas);
      }
      try {
        runeRingGeo.dispose();
        runeRingMat.dispose();
        innerRingGeo.dispose();
        innerRingMat.dispose();
        dustGeo.dispose();
        dustMat.dispose();
        renderer.dispose();
        renderer.forceContextLoss();
      } catch {
        // cleanup safe
      }
    };
  }, [booksList, onSelectBook]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[620px] rounded-2xl overflow-hidden border border-amber-900/20 shadow-xl bg-[#faf7f2]"
    >
      {/* Three.js WebGL Canvas Mount Container */}
      <div ref={mountRef} className="absolute inset-0 w-full h-full cursor-pointer" />

      {/* Empty Library Atmospheric Welcome Card */}
      {booksList.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-30 p-4">
          <div className="bg-white/95 border border-amber-900/20 p-6 sm:p-8 rounded-2xl shadow-2xl max-w-md text-center pointer-events-auto space-y-4 backdrop-blur-md">
            <div className="w-14 h-14 mx-auto rounded-2xl bg-amber-100 border border-amber-200 flex items-center justify-center text-amber-800 shadow-sm">
              <BookOpen className="w-7 h-7" />
            </div>
            <div className="space-y-1.5">
              <h3 className="font-serif font-bold text-xl text-slate-900">Your 3D Library is Empty</h3>
              <p className="text-xs text-slate-600 leading-relaxed font-serif">
                The grand oak shelves are currently empty, waiting for your stories. Chat with Karna-49M in the Chatbot, then click <strong>"Bind Chat to 3D Library"</strong> to place your first written book here!
              </p>
            </div>
            {onNavigateToChat && (
              <button
                onClick={onNavigateToChat}
                className="w-full py-2.5 px-4 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-serif font-bold text-xs flex items-center justify-center gap-2 shadow-md shadow-amber-600/20 transition cursor-pointer"
              >
                <Sparkles className="w-4 h-4" />
                <span>Open Chatbot to Generate a Story</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* 3D Corridor Top HUD Overlay */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none z-20">
        <div className="bg-white/90 border border-amber-800/20 px-3.5 py-1.5 rounded-xl backdrop-blur-md flex items-center gap-2 text-xs font-serif text-slate-800 shadow-md">
          <Sparkles className="w-3.5 h-3.5 text-amber-600" />
          <span>
            3D Sanctuary &bull; {booksList.length === 0 ? "Empty Shelves (Awaiting Inscription)" : `${booksList.length} Bound Book${booksList.length === 1 ? '' : 's'} on Shelves`}
          </span>
        </div>

        {/* Hovered Book Card Overlay */}
        {hoveredBook && (
          <div className="bg-white/95 border border-amber-700/30 px-4 py-2.5 rounded-xl backdrop-blur-md shadow-2xl text-xs flex items-center gap-3 animate-in zoom-in-95 pointer-events-auto">
            <span className="text-2xl filter drop-shadow">{hoveredBook.emblem}</span>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-serif font-bold text-slate-900">{hoveredBook.title}</span>
                <span className="text-[10px] font-mono text-amber-800 font-semibold">({hoveredBook.genre})</span>
              </div>
              <p className="text-[11px] text-slate-600 font-serif italic">{hoveredBook.subtitle}</p>
              <div className="flex items-center gap-1.5 mt-1">
                <span className="text-[10px] font-mono text-amber-900 bg-amber-100/80 px-2 py-0.5 rounded border border-amber-800/15 font-medium">
                  {sessionsSummary && sessionsSummary[hoveredBook.id]?.messageCount
                    ? `📜 Session: ${sessionsSummary[hoveredBook.id].messageCount} Inscriptions`
                    : "✧ Fresh Folio: Awaiting Inscription"}
                </span>
              </div>
            </div>
            <button
              onClick={() => onSelectBook(hoveredBook)}
              className="ml-2 px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-serif font-bold text-xs flex items-center gap-1 transition shadow-md shadow-amber-950/20 cursor-pointer"
            >
              <span>Open Chronicle</span>
              <ChevronRight className="w-3 h-3" />
            </button>
          </div>
        )}
      </div>

      {/* Navigation Controls Floating Dock */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-white/95 border border-amber-800/20 px-4 py-2 rounded-2xl backdrop-blur-md shadow-xl z-20 text-xs">
        <span className="text-[11px] font-mono text-amber-950/70 font-semibold uppercase tracking-wider">
          Aisle Navigation:
        </span>

        <button
          onClick={moveForward}
          className="px-3 py-1.5 rounded-xl bg-amber-50 hover:bg-amber-100 text-amber-950 border border-amber-800/20 flex items-center gap-1 transition cursor-pointer font-serif font-medium shadow-sm"
          title="Move camera forward down the library corridor (W or Up Arrow)"
        >
          <ArrowUp className="w-3.5 h-3.5 text-amber-800" />
          <span>Glide Forward</span>
        </button>

        <button
          onClick={moveBackward}
          className="px-3 py-1.5 rounded-xl bg-amber-50 hover:bg-amber-100 text-amber-950 border border-amber-800/20 flex items-center gap-1 transition cursor-pointer font-serif font-medium shadow-sm"
          title="Move camera backward down the library corridor (S or Down Arrow)"
        >
          <ArrowDown className="w-3.5 h-3.5 text-amber-800" />
          <span>Glide Backward</span>
        </button>

        <button
          onClick={resetCamera}
          className="p-1.5 rounded-xl bg-amber-50 hover:bg-amber-100 border border-amber-800/20 text-slate-700 hover:text-slate-900 transition cursor-pointer"
          title="Reset camera viewpoint"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Floating Instructions Pill */}
      <div className="absolute bottom-4 right-4 hidden md:flex items-center gap-1.5 bg-white/85 border border-amber-800/20 px-3 py-1.5 rounded-xl text-[10px] text-slate-700 font-mono backdrop-blur-md pointer-events-none shadow-sm">
        <Info className="w-3 h-3 text-amber-600" />
        <span>Click any book to open &bull; Mouse wheel / W/S to walk aisle</span>
      </div>
    </div>
  );
}
