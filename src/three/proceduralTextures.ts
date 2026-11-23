import * as THREE from 'three';
import {
  BASIN_RADIUS,
  TowerSpec,
  seawallZ,
} from '../data/skylineLayout';

export interface ProceduralTextureBundle {
  brushedMetalMap: THREE.CanvasTexture;
  architecturalGrainMap: THREE.CanvasTexture;
  softboxDiffuserMap: THREE.CanvasTexture;
  waterMicroRippleMap: THREE.DataTexture;
  microImperfectionMap: THREE.DataTexture;
  groundContactAoMap: THREE.DataTexture;
  dispose: () => void;
}

function hash2D(x: number, y: number): number {
  const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
  return n - Math.floor(n);
}

function hash2DVec(ix: number, iy: number, period: number): [number, number] {
  const px = ((ix % period) + period) % period;
  const py = ((iy % period) + period) % period;
  return [hash2D(px, py), hash2D(px + 37.3, py + 91.7)];
}

function smoothNoiseTileable(x: number, y: number, period: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);

  const p00 = hash2D(((xi % period) + period) % period, ((yi % period) + period) % period);
  const p10 = hash2D((((xi + 1) % period) + period) % period, ((yi % period) + period) % period);
  const p01 = hash2D(((xi % period) + period) % period, (((yi + 1) % period) + period) % period);
  const p11 = hash2D((((xi + 1) % period) + period) % period, (((yi + 1) % period) + period) % period);

  const nx0 = p00 + (p10 - p00) * u;
  const nx1 = p01 + (p11 - p01) * u;
  return nx0 + (nx1 - nx0) * v;
}

function fbmTileable(u: number, v: number, baseFreq: number, octaves = 4): number {
  let value = 0;
  let amp = 0.5;
  let freq = baseFreq;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    value += amp * smoothNoiseTileable(u * freq, v * freq, Math.round(freq));
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return value / norm;
}

function worleyTileable(u: number, v: number, cells: number): number {
  const gx = u * cells;
  const gy = v * cells;
  const ix = Math.floor(gx);
  const iy = Math.floor(gy);
  const fx = gx - ix;
  const fy = gy - iy;

  let d1 = 10;
  let d2 = 10;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const [rx, ry] = hash2DVec(ix + dx, iy + dy, cells);
      const px = dx + rx - fx;
      const py = dy + ry - fy;
      const d = Math.sqrt(px * px + py * py);
      if (d < d1) {
        d2 = d1;
        d1 = d;
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  // Normalized edge/cellular contrast
  return Math.min(1, Math.max(0, (d2 - d1) * 1.45));
}

/**
 * Generates an offscreen HTML5 Canvas brushed-metal anisotropic streak texture
 * with subtle lathe/brushing micro-variance for metallic fins, tower crowns,
 * and the machined aluminum diorama bezel.
 */
function createBrushedMetalTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);

  const imgData = ctx.getImageData(0, 0, size, size);
  const data = imgData.data;

  for (let y = 0; y < size; y++) {
    const vNorm = y / size;
    const rowStreak = (hash2D(0, y * 1.37) - 0.5) * 42;
    const latheWave = Math.sin(vNorm * Math.PI * 18) * 7;
    for (let x = 0; x < size; x++) {
      const uNorm = x / size;
      const idx = (y * size + x) * 4;
      const fbmMod = (fbmTileable(uNorm, vNorm, 6, 3) - 0.5) * 18;
      const fineStreak =
        Math.sin(y * 0.65 + hash2D(Math.floor(x / 16), y) * 2.1) * 14 +
        (hash2D(x * 0.25, y * 1.9) - 0.5) * 22;
      const v = Math.max(0, Math.min(255, Math.round(128 + rowStreak + fineStreak + latheWave + fbmMod)));
      data[idx] = v;
      data[idx + 1] = v;
      data[idx + 2] = v;
      data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates an offscreen HTML5 Canvas ultra-fine bead-blast aggregate micro-grain map
 * so dead-matte architectural concrete, slate setbacks, and roadways read as physical scale-model material.
 */
function createArchitecturalGrainTexture(maxAnisotropy: number): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  const imgData = ctx.createImageData(size, size);
  const data = imgData.data;

  for (let y = 0; y < size; y++) {
    const vn = y / size;
    for (let x = 0; x < size; x++) {
      const un = x / size;
      const idx = (y * size + x) * 4;
      const coarse = fbmTileable(un, vn, 6, 3);
      const medium = smoothNoiseTileable(un * 24, vn * 24, 24);
      const fine = hash2D(x, y);
      // Tight physical bead-blast range (~116..158) to avoid noisy salt-and-pepper grain
      const grain = coarse * 0.45 + medium * 0.35 + fine * 0.20;
      const v = Math.max(0, Math.min(255, Math.round(108 + grain * 54)));
      data[idx] = v;
      data[idx + 1] = v;
      data[idx + 2] = v;
      data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a realistic studio softbox diffuser silk texture with feathered edge falloff
 * and subtle woven diffusion scrim micro-texture.
 */
function createSoftboxDiffuserTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#050506';
  ctx.fillRect(0, 0, size, size);

  const grad = ctx.createRadialGradient(
    size * 0.5,
    size * 0.5,
    size * 0.08,
    size * 0.5,
    size * 0.5,
    size * 0.56,
  );
  grad.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
  grad.addColorStop(0.55, 'rgba(232, 236, 242, 0.88)');
  grad.addColorStop(0.85, 'rgba(130, 136, 146, 0.38)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0.0)');

  ctx.fillStyle = grad;
  ctx.fillRect(8, 8, size - 16, size - 16);

  // Subtle woven ripstop nylon diffuser grid
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.06)';
  ctx.lineWidth = 1;
  for (let i = 0; i < size; i += 8) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(size, i);
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a seamless 256x256 procedural high-frequency capillary wind ripple normal map
 * for polished dark smoked harbor resin/water (zero blurry low-frequency smudges).
 */
function createWaterMicroRippleTexture(maxAnisotropy: number): THREE.DataTexture {
  const size = 256;
  const data = new Uint8Array(size * size * 4);

  for (let y = 0; y < size; y++) {
    const vn = y / size;
    for (let x = 0; x < size; x++) {
      const un = x / size;
      const u = un * Math.PI * 2;
      const v = vn * Math.PI * 2;

      // Multi-directional crisp capillary wavelets (prime harmonic ratios prevent tiling streaks)
      const dx =
        Math.cos(u * 5 + v * 3) * 0.36 +
        Math.cos(u * 11 - v * 7) * 0.28 +
        Math.cos(u * 17 + v * 13) * 0.20 +
        Math.cos(u * 23 - v * 19) * 0.14;
      const dz =
        Math.sin(v * 7 - u * 4) * 0.36 +
        Math.sin(v * 13 + u * 9) * 0.28 +
        Math.sin(v * 19 - u * 15) * 0.20 +
        Math.sin(v * 29 + u * 21) * 0.14;

      const idx = (y * size + x) * 4;
      data[idx] = Math.max(0, Math.min(255, Math.round((dx * 0.5 + 0.5) * 255)));
      data[idx + 1] = Math.max(0, Math.min(255, Math.round((dz * 0.5 + 0.5) * 255)));
      data[idx + 2] = 240;
      data[idx + 3] = 255;
    }
  }

  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a tileable 256x256 procedural micro-imperfection map:
 * - R: Low/mid-frequency FBM (oleophobic glass smudge & anodized finish variance)
 * - G: Worley cellular micro-structure (satin bead-blast & aggregate pores)
 * - B: High-frequency fine lathe/machining micro-scratches
 */
function createMicroImperfectionTexture(maxAnisotropy: number): THREE.DataTexture {
  const size = 256;
  const data = new Uint8Array(size * size * 4);

  for (let y = 0; y < size; y++) {
    const vn = y / size;
    for (let x = 0; x < size; x++) {
      const un = x / size;
      const fbm = fbmTileable(un, vn, 4, 4);
      const worley = worleyTileable(un, vn, 16);
      const fineLathe =
        0.5 +
        0.28 * Math.sin(vn * Math.PI * 64 + fbm * 4.5) +
        0.22 * (smoothNoiseTileable(un * 12, vn * 48, 48) - 0.5);

      const idx = (y * size + x) * 4;
      data[idx] = Math.max(0, Math.min(255, Math.round(fbm * 255)));
      data[idx + 1] = Math.max(0, Math.min(255, Math.round(worley * 255)));
      data[idx + 2] = Math.max(0, Math.min(255, Math.round(fineLathe * 255)));
      data[idx + 3] = 255;
    }
  }

  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return texture;
}

function signedBoxDistance2D(
  px: number,
  pz: number,
  cx: number,
  cz: number,
  hw: number,
  hd: number,
  rotY: number,
): number {
  const c = Math.cos(-rotY);
  const s = Math.sin(-rotY);
  const dx = px - cx;
  const dz = pz - cz;
  const lx = Math.abs(dx * c - dz * s) - hw;
  const lz = Math.abs(dx * s + dz * c) - hd;
  const outX = Math.max(lx, 0);
  const outZ = Math.max(lz, 0);
  return Math.sqrt(outX * outX + outZ * outZ) + Math.min(Math.max(lx, lz), 0);
}

/**
 * Bakes a 512x512 Spatial Optical Map across the circular San Diego diorama footprint:
 * - R: Tight base crevice Contact AO (seawall, tower bases, podiums, street curbs)
 * - G: Deep urban street-canyon Ambient Occlusion (accumulated from surrounding towers)
 * - B: Architectural blocker heightfield (normalized 0..10 world units) for real-time
 *      GLSL3 directional soft raymarched shadows across rooftops, streets & water!
 */
function createGroundContactAoTexture(
  landmarks: TowerSpec[],
  cityBlocks: TowerSpec[],
  maxAnisotropy: number,
): THREE.DataTexture {
  const size = 512;
  const data = new Uint8Array(size * size * 4);
  const extent = BASIN_RADIUS * 1.05;

  const blockers: Array<{
    cx: number;
    cz: number;
    hw: number;
    hd: number;
    rot: number;
    height: number;
    weight: number;
  }> = [];

  for (const t of landmarks) {
    // Podium base
    blockers.push({
      cx: t.x,
      cz: t.z,
      hw: t.width * 0.58,
      hd: t.depth * 0.58,
      rot: t.rotationY,
      height: 0.53,
      weight: 0.85,
    });
    // Main hero tower massing
    if (t.style === 'obelisk') {
      blockers.push({
        cx: t.x,
        cz: t.z,
        hw: t.width * 0.50,
        hd: t.depth * 0.50,
        rot: t.rotationY,
        height: 0.08 + 0.45 + t.height * 0.82,
        weight: 1.0,
      });
      blockers.push({
        cx: t.x,
        cz: t.z,
        hw: t.width * 0.24,
        hd: t.depth * 0.24,
        rot: t.rotationY,
        height: 0.08 + 0.45 + t.height * 0.98,
        weight: 1.0,
      });
    } else if (t.style === 'emerald-cluster') {
      blockers.push({
        cx: t.x,
        cz: t.z,
        hw: t.width * 0.46,
        hd: t.depth * 0.46,
        rot: t.rotationY,
        height: 0.08 + t.height * 0.92,
        weight: 1.0,
      });
    } else {
      blockers.push({
        cx: t.x,
        cz: t.z,
        hw: t.width * 0.48,
        hd: t.depth * 0.48,
        rot: t.rotationY,
        height: 0.08 + 0.45 + t.height * 0.90,
        weight: 1.0,
      });
    }
  }

  for (const b of cityBlocks) {
    blockers.push({
      cx: b.x,
      cz: b.z,
      hw: b.width * 0.50,
      hd: b.depth * 0.50,
      rot: b.rotationY,
      height: 0.08 + b.height,
      weight: Math.min(0.95, 0.45 + b.height * 0.14),
    });
  }

  // Convention Center main hall + Sails Pavilion roof ridge
  blockers.push({
    cx: 7.4,
    cz: 0.45,
    hw: 2.8,
    hd: 1.3,
    rot: 0.28,
    height: 0.96,
    weight: 0.92,
  });
  blockers.push({
    cx: 7.4,
    cz: 0.35,
    hw: 2.45,
    hd: 0.82,
    rot: 0.28,
    height: 1.48,
    weight: 0.95,
  });

  // USS Midway & Navy Pier
  blockers.push({
    cx: -10.5,
    cz: 2.55,
    hw: 1.9,
    hd: 0.55,
    rot: 0.55,
    height: 0.52,
    weight: 0.82,
  });

  for (let y = 0; y < size; y++) {
    const wz = ((y / (size - 1)) * 2 - 1) * extent;
    for (let x = 0; x < size; x++) {
      const wx = ((x / (size - 1)) * 2 - 1) * extent;

      let tightCreviceAo = 1.0;
      let canyonAccum = 0.0;
      let maxH = 0.0;

      for (let i = 0; i < blockers.length; i++) {
        const b = blockers[i];
        const maxR = Math.max(b.hw, b.hd) + 3.2;
        if (Math.abs(wx - b.cx) > maxR || Math.abs(wz - b.cz) > maxR) continue;

        const sdf = signedBoxDistance2D(wx, wz, b.cx, b.cz, b.hw, b.hd, b.rot);
        if (sdf <= 0.03) {
          if (b.height > maxH) maxH = b.height;
        }
        const dist = Math.max(0, sdf);
        const tight = 1.0 - Math.exp(-dist * 11.5) * 0.68 * b.weight;
        tightCreviceAo = Math.min(tightCreviceAo, tight);

        // Street canyon accumulation: taller nearby walls darken canyon floor/streets noticeably
        const heightFactor = Math.min(1.0, b.height / 5.5);
        canyonAccum += Math.exp(-dist * 1.15) * (0.14 + 0.36 * heightFactor);
      }

      const canyonAo = Math.max(0.22, 1.0 - Math.min(0.78, canyonAccum * 0.52));

      // Perimeter plinth rim & seawall curb crevice AO
      const rad = Math.sqrt(wx * wx + wz * wz);
      const rimDist = Math.abs(BASIN_RADIUS - 0.16 - rad);
      const rimAo = 1.0 - Math.exp(-rimDist * 9.0) * 0.42;

      const swDist = Math.abs(wz - seawallZ(wx));
      const swAo = 1.0 - Math.exp(-swDist * 10.0) * 0.38;

      tightCreviceAo = Math.min(tightCreviceAo, rimAo, swAo);

      const idx = (y * size + x) * 4;
      data[idx] = Math.max(0, Math.min(255, Math.round(tightCreviceAo * 255)));
      data[idx + 1] = Math.max(0, Math.min(255, Math.round(canyonAo * 255)));
      data[idx + 2] = Math.max(0, Math.min(255, Math.round(Math.min(1.0, maxH / 10.0) * 255)));
      data[idx + 3] = 255;
    }
  }

  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return texture;
}

export function createProceduralTextureBundle(
  renderer: THREE.WebGLRenderer,
  landmarks: TowerSpec[],
  cityBlocks: TowerSpec[],
): ProceduralTextureBundle {
  const maxAnisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy() || 1);
  const brushedMetalMap = createBrushedMetalTexture(maxAnisotropy);
  const architecturalGrainMap = createArchitecturalGrainTexture(maxAnisotropy);
  const softboxDiffuserMap = createSoftboxDiffuserTexture();
  const waterMicroRippleMap = createWaterMicroRippleTexture(maxAnisotropy);
  const microImperfectionMap = createMicroImperfectionTexture(maxAnisotropy);
  const groundContactAoMap = createGroundContactAoTexture(
    landmarks,
    cityBlocks,
    maxAnisotropy,
  );

  return {
    brushedMetalMap,
    architecturalGrainMap,
    softboxDiffuserMap,
    waterMicroRippleMap,
    microImperfectionMap,
    groundContactAoMap,
    dispose: () => {
      brushedMetalMap.dispose();
      architecturalGrainMap.dispose();
      softboxDiffuserMap.dispose();
      waterMicroRippleMap.dispose();
      microImperfectionMap.dispose();
      groundContactAoMap.dispose();
    },
  };
}
