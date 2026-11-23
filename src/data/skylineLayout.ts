export interface TowerSpec {
  id: string;
  name: string;
  x: number;
  z: number;
  width: number;
  depth: number;
  height: number;
  rotationY: number;
  style:
    | 'obelisk'
    | 'emerald-cluster'
    | 'chamfered-twin'
    | 'curved-slab'
    | 'stepped'
    | 'box';
  isHero: boolean;
  mullionDensity: number;
}

export interface SailboatSpec {
  id: string;
  baseX: number;
  baseZ: number;
  orbitRadius: number;
  speed: number;
  phase: number;
  heading: number;
  scale: number;
}

export interface StreetlightSpec {
  x: number;
  z: number;
}

export type LightingPresetId = 'overhead-studio' | 'grazing-rim' | 'harbor-dusk';

export interface LightingPreset {
  id: LightingPresetId;
  label: string;
  subtitle: string;
  softboxKeyElevation: number;
  softboxKeyAzimuth: number;
  keyIntensity: number;
  rimIntensity: number;
  fillIntensity: number;
  waterReflectivity: number;
  glassEdgeGlow: number;
}

export const BASIN_RADIUS = 13.4;

/**
 * Returns the Z coordinate of the curved San Diego Embarcadero seawall
 * at a given X coordinate. Land/city sits at z <= seawallZ(x); bay water
 * sits at z > seawallZ(x).
 */
export function seawallZ(x: number): number {
  return 2.6 + 2.35 * Math.cos(x * 0.24 + 0.35) - 0.14 * x;
}

export function isInsideBasin(x: number, z: number, margin = 0): boolean {
  const r = BASIN_RADIUS - margin;
  return x * x + z * z <= r * r;
}

export function isOnLand(x: number, z: number, setback = 0.35): boolean {
  return isInsideBasin(x, z, 0.7) && z <= seawallZ(x) - setback;
}

export function isOnBayWater(x: number, z: number, margin = 0.85): boolean {
  return isInsideBasin(x, z, margin) && z >= seawallZ(x) + 0.75;
}

/**
 * Deterministic PRNG (Mulberry32) for reproducible architectural blocks.
 */
export function createSeededRng(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export const HERO_LANDMARKS: TowerSpec[] = [
  {
    id: 'one-america-plaza',
    name: 'One America Plaza',
    x: -7.6,
    z: -4.4,
    width: 1.55,
    depth: 1.55,
    height: 8.6,
    rotationY: 0.22,
    style: 'obelisk',
    isHero: true,
    mullionDensity: 18,
  },
  {
    id: 'emerald-plaza',
    name: 'Emerald Plaza',
    x: -4.9,
    z: -4.2,
    width: 1.9,
    depth: 1.8,
    height: 6.8,
    rotationY: 0.18,
    style: 'emerald-cluster',
    isHero: true,
    mullionDensity: 16,
  },
  {
    id: 'bayside-tower-west',
    name: 'Bayside Tower',
    x: -7.1,
    z: -1.1,
    width: 1.3,
    depth: 1.3,
    height: 6.1,
    rotationY: 0.22,
    style: 'stepped',
    isHero: true,
    mullionDensity: 15,
  },
  {
    id: 'harbor-club-west',
    name: 'Harbor Club West',
    x: -3.2,
    z: 2.3,
    width: 1.35,
    depth: 1.35,
    height: 6.4,
    rotationY: 0.24,
    style: 'chamfered-twin',
    isHero: true,
    mullionDensity: 16,
  },
  {
    id: 'harbor-club-east',
    name: 'Harbor Club East',
    x: -1.45,
    z: 2.65,
    width: 1.35,
    depth: 1.35,
    height: 6.0,
    rotationY: 0.24,
    style: 'chamfered-twin',
    isHero: true,
    mullionDensity: 16,
  },
  {
    id: 'manchester-hyatt-west',
    name: 'Manchester Grand Hyatt North',
    x: 1.45,
    z: 1.95,
    width: 1.5,
    depth: 1.05,
    height: 5.8,
    rotationY: -0.26,
    style: 'curved-slab',
    isHero: true,
    mullionDensity: 14,
  },
  {
    id: 'manchester-hyatt-east',
    name: 'Manchester Grand Hyatt South',
    x: 2.95,
    z: 1.55,
    width: 1.45,
    depth: 1.0,
    height: 5.4,
    rotationY: -0.26,
    style: 'curved-slab',
    isHero: true,
    mullionDensity: 14,
  },
  {
    id: 'symphony-tower',
    name: 'Symphony Towers',
    x: 0.2,
    z: -4.8,
    width: 1.6,
    depth: 1.4,
    height: 5.5,
    rotationY: 0.22,
    style: 'stepped',
    isHero: true,
    mullionDensity: 13,
  },
  {
    id: 'pinnacle-marina',
    name: 'Pinnacle Marina',
    x: 4.5,
    z: -1.4,
    width: 1.2,
    depth: 1.2,
    height: 5.6,
    rotationY: -0.22,
    style: 'chamfered-twin',
    isHero: true,
    mullionDensity: 14,
  },
  {
    id: 'electra-tower',
    name: 'Electra Residences',
    x: -5.3,
    z: -0.2,
    width: 1.25,
    depth: 1.25,
    height: 5.2,
    rotationY: 0.22,
    style: 'stepped',
    isHero: true,
    mullionDensity: 14,
  },
];

/**
 * Generates background downtown blocks placed strictly on land inside the circular
 * diorama, avoiding hero landmark footprints and the Convention Center waterfront parcel.
 */
export function generateCityGridBlocks(seed = 2026): TowerSpec[] {
  const rng = createSeededRng(seed);
  const blocks: TowerSpec[] = [];
  const stepX = 1.65;
  const stepZ = 1.65;

  for (let gx = -11.2; gx <= 11.2; gx += stepX) {
    for (let gz = -11.2; gz <= 4.5; gz += stepZ) {
      const jitterX = (rng() - 0.5) * 0.28;
      const jitterZ = (rng() - 0.5) * 0.28;
      const x = Number((gx + jitterX).toFixed(3));
      const z = Number((gz + jitterZ).toFixed(3));

      if (!isOnLand(x, z, 0.95)) continue;

      // Reserve Convention Center footprint (east bayfront)
      if (x > 4.2 && x < 11.4 && z > -2.3 && z < 2.6) continue;

      // Avoid colliding with hero landmarks
      const overlapsHero = HERO_LANDMARKS.some(
        (hero) => Math.hypot(hero.x - x, hero.z - z) < 1.55,
      );
      if (overlapsHero) continue;

      const distFromCore = Math.hypot(x + 2.0, z + 3.0);
      const heightFalloff = Math.max(0.28, 1 - distFromCore / 13.5);
      const baseHeight = 1.0 + rng() * 2.8 * heightFalloff;
      const tallAccent = rng() > 0.84 ? 1.45 + rng() * 1.4 : 1.0;

      blocks.push({
        id: `city-block-${blocks.length}`,
        name: `Downtown Parcel ${blocks.length + 1}`,
        x,
        z,
        width: 0.82 + rng() * 0.52,
        depth: 0.82 + rng() * 0.52,
        height: Number((baseHeight * tallAccent).toFixed(3)),
        rotationY: 0.22 + (rng() - 0.5) * 0.08,
        style: rng() > 0.65 ? 'stepped' : 'box',
        isHero: false,
        mullionDensity: 7 + Math.floor(rng() * 5),
      });
    }
  }

  return blocks;
}

export interface MidRiseArchitectureSpec {
  facadeMaterial: 'curtain-glass' | 'masonry-slate';
  parapetHeight: number;
  hasMechanicalPenthouse: boolean;
  penthouseHeight: number;
  hvacCount: 1 | 2;
  hasMetallicHvac: boolean;
}

/**
 * Deterministically assigns material separation (glossy curtain-wall glass vs dead-matte
 * dark slate/limestone masonry), varied parapet heights, and rooftop mechanical units
 * across mid-rise blocks so adjacent parcels break up repetitive silhouettes.
 */
export function classifyMidRiseBlock(index: number, block: TowerSpec): MidRiseArchitectureSpec {
  const rng = createSeededRng(2026 + index * 7919);
  const matRoll = rng();
  const parapetRoll = rng();
  const penthouseRoll = rng();
  const hvacRoll = rng();
  const metalRoll = rng();

  const facadeMaterial: 'curtain-glass' | 'masonry-slate' =
    block.height >= 2.85 || matRoll > 0.46 ? 'curtain-glass' : 'masonry-slate';

  const parapetHeight =
    facadeMaterial === 'masonry-slate'
      ? Math.min(0.145, Math.max(0.068, block.height * (0.042 + parapetRoll * 0.032)))
      : Math.min(0.095, Math.max(0.042, block.height * (0.025 + parapetRoll * 0.020)));

  const hasMechanicalPenthouse = block.style === 'stepped' || penthouseRoll > 0.62;
  const penthouseHeight = hasMechanicalPenthouse
    ? Math.min(0.42, Math.max(0.16, block.height * (0.11 + parapetRoll * 0.06)))
    : 0;

  return {
    facadeMaterial,
    parapetHeight,
    hasMechanicalPenthouse,
    penthouseHeight,
    hvacCount: hvacRoll > 0.34 ? 2 : 1,
    hasMetallicHvac: metalRoll > 0.52,
  };
}

export function generateSeawallLights(count = 30): StreetlightSpec[] {
  const lights: StreetlightSpec[] = [];
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    const x = -10.8 + t * 21.4;
    const z = seawallZ(x) + 0.08;
    if (isInsideBasin(x, z, 1.05)) {
      lights.push({
        x: Number(x.toFixed(3)),
        z: Number(z.toFixed(3)),
      });
    }
  }
  return lights;
}

export const HARBOR_SAILBOATS: SailboatSpec[] = [
  { id: 'sail-1', baseX: -6.2, baseZ: 7.4, orbitRadius: 0.55, speed: 0.22, phase: 0.2, heading: 0.35, scale: 1.0 },
  { id: 'sail-2', baseX: -2.4, baseZ: 9.1, orbitRadius: 0.7, speed: 0.18, phase: 1.4, heading: 0.65, scale: 1.12 },
  { id: 'sail-3', baseX: 0.6, baseZ: 7.8, orbitRadius: 0.45, speed: 0.25, phase: 2.3, heading: 0.48, scale: 0.88 },
  { id: 'sail-4', baseX: 3.9, baseZ: 8.6, orbitRadius: 0.65, speed: 0.2, phase: 3.6, heading: -0.45, scale: 1.05 },
  { id: 'sail-5', baseX: 7.1, baseZ: 5.6, orbitRadius: 0.5, speed: 0.24, phase: 4.9, heading: -0.65, scale: 0.92 },
  { id: 'sail-6', baseX: 8.6, baseZ: 3.4, orbitRadius: 0.4, speed: 0.19, phase: 5.5, heading: -0.8, scale: 0.8 },
];

export const LIGHTING_PRESETS: Record<LightingPresetId, LightingPreset> = {
  'overhead-studio': {
    id: 'overhead-studio',
    label: 'Overhead Softbox',
    subtitle: 'Raking 39° upper-left softbox key & crisp diagonal cast shadows',
    softboxKeyElevation: 0.52,
    softboxKeyAzimuth: 2.66,
    keyIntensity: 1.34,
    rimIntensity: 0.94,
    fillIntensity: 0.34,
    waterReflectivity: 0.96,
    glassEdgeGlow: 0.92,
  },
  'grazing-rim': {
    id: 'grazing-rim',
    label: 'Grazing Rim Sweep',
    subtitle: 'Low-angle raking key accentuating brushed metal & long shadows',
    softboxKeyElevation: 0.34,
    softboxKeyAzimuth: 2.72,
    keyIntensity: 1.18,
    rimIntensity: 1.38,
    fillIntensity: 0.34,
    waterReflectivity: 1.12,
    glassEdgeGlow: 1.06,
  },
  'harbor-dusk': {
    id: 'harbor-dusk',
    label: 'Harbor Dusk',
    subtitle: 'High-contrast harbor silhouette with sculpted curtain-wall reflections',
    softboxKeyElevation: 0.44,
    softboxKeyAzimuth: 1.28,
    keyIntensity: 0.98,
    rimIntensity: 1.15,
    fillIntensity: 0.36,
    waterReflectivity: 1.20,
    glassEdgeGlow: 1.18,
  },
};
