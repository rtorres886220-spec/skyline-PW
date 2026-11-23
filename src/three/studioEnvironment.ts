import * as THREE from 'three';
import { RectAreaLightUniformsLib } from 'three/examples/jsm/lights/RectAreaLightUniformsLib.js';
import { LightingPreset } from '../data/skylineLayout';

export interface StudioLightingRig {
  environmentTexture: THREE.Texture;
  studioIblMap: THREE.DataTexture;
  keySoftbox: THREE.RectAreaLight | null;
  rimSoftbox: THREE.RectAreaLight | null;
  fillSoftbox: THREE.RectAreaLight | null;
  hemiLight: THREE.HemisphereLight;
  shadowKeyLight: THREE.DirectionalLight | null;
  applyPreset: (preset: LightingPreset) => void;
  dispose: () => void;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

/**
 * Evaluates a rectangular studio softbox in direction space using angular coordinates
 * on a tangent frame around `centerDir`, returning feathered silk intensity.
 */
function evalSoftboxDir(
  dx: number,
  dy: number,
  dz: number,
  cx: number,
  cy: number,
  cz: number,
  halfW: number,
  halfH: number,
  feather: number,
): number {
  const lenC = Math.hypot(cx, cy, cz) || 1;
  const fx = cx / lenC;
  const fy = cy / lenC;
  const fz = cz / lenC;

  const denom = dx * fx + dy * fy + dz * fz;
  if (denom <= 0.08) return 0;

  // Orthonormal basis around softbox normal
  const upRefX = Math.abs(fy) > 0.94 ? 0 : 0;
  const upRefY = Math.abs(fy) > 0.94 ? 0 : 1;
  const upRefZ = Math.abs(fy) > 0.94 ? 1 : 0;

  // right = normalize(cross(upRef, forward))
  let rx = upRefY * fz - upRefZ * fy;
  let ry = upRefZ * fx - upRefX * fz;
  let rz = upRefX * fy - upRefY * fx;
  const rLen = Math.hypot(rx, ry, rz) || 1;
  rx /= rLen;
  ry /= rLen;
  rz /= rLen;

  // up = cross(forward, right)
  const ux = fy * rz - fz * ry;
  const uy = fz * rx - fx * rz;
  const uz = fx * ry - fy * rx;

  const px = (dx * rx + dy * ry + dz * rz) / denom;
  const py = (dx * ux + dy * uy + dz * uz) / denom;

  const sx = 1 - smoothstep(halfW - feather, halfW + feather, Math.abs(px));
  const sy = 1 - smoothstep(halfH - feather, halfH + feather, Math.abs(py));
  // Subtle diffused scrim hot-spot roll-off
  const scrimFalloff =
    1 - 0.24 * Math.min(1, (px * px) / (halfW * halfW + 0.01) + (py * py) / (halfH * halfH + 0.01));

  return sx * sy * Math.max(0.55, scrimFalloff);
}

/**
 * Generates a procedurally baked 512x256 equirectangular studio IBL map once at startup
 * with full hardware mipmaps (`LinearMipmapLinearFilter`) so shaders sample blurred
 * studio reflections by roughness via `textureLod`.
 *
 * Includes:
 * 1. Overhead diffused multi-panel softbox bank (with dark louver dividers)
 * 2. Two vertical rim strip lights (back-left architectural key rim & opposite kick strip)
 * 3. Dark negative-fill studio flags for crisp reflection contrast
 * 4. Subtle low horizon bounce band for bay water & sail grazing sheen
 */
function createProceduralStudioIblMap(maxAnisotropy: number): THREE.DataTexture {
  const width = 512;
  const height = 256;
  const data = new Uint8Array(width * height * 4);
  const maxHdr = 2.8;

  for (let y = 0; y < height; y++) {
    // v = 0 at bottom (DataTexture flipY = false by default) -> v = 1 at zenith
    const v = (y + 0.5) / height;
    const phi = (1 - v) * Math.PI; // 0 at zenith (+Y), PI at nadir (-Y)
    const sinPhi = Math.sin(phi);
    const dy = Math.cos(phi);

    for (let x = 0; x < width; x++) {
      const u = (x + 0.5) / width;
      const theta = u * Math.PI * 2; // azimuth around studio
      const dx = -Math.sin(theta) * sinPhi;
      const dz = -Math.cos(theta) * sinPhi;

      // Ambient dark studio cyclorama floor & ceiling base (subtle cool slate bias)
      const skyGrad = smoothstep(-0.35, 0.75, dy);
      let r = 0.022 + 0.029 * skyGrad;
      let g = 0.025 + 0.034 * skyGrad;
      let b = 0.034 + 0.046 * skyGrad;

      // 1. Overhead-left/front primary diffused softbox bank (3:1 crisp neutral-warm key source)
      const overheadRaw = evalSoftboxDir(dx, dy, dz, -0.44, 0.76, 0.48, 0.85, 0.56, 0.14);
      if (overheadRaw > 0.001) {
        // Subtle dark structural louver cross-bars in the softbox bank
        const louverX = smoothstep(0.02, 0.065, Math.abs(dx + 0.44));
        const louverZ = smoothstep(0.02, 0.065, Math.abs(dz - 0.48));
        const bankMask = overheadRaw * (0.48 + 0.52 * louverX * louverZ);
        r += bankMask * 2.26;
        g += bankMask * 2.21;
        b += bankMask * 2.13;
      }

      // 2a. Left-side vertical architectural strip light (crisp warm-neutral edge kick)
      const rimStripA = evalSoftboxDir(dx, dy, dz, -0.82, 0.34, 0.44, 0.16, 0.88, 0.08);
      if (rimStripA > 0.001) {
        r += rimStripA * 2.10;
        g += rimStripA * 2.08;
        b += rimStripA * 2.04;
      }

      // 2b. Subtle cool opposite fill strip (restrained ~3:1 contrast)
      const rimStripB = evalSoftboxDir(dx, dy, dz, 0.78, 0.28, -0.56, 0.18, 0.72, 0.10);
      if (rimStripB > 0.001) {
        r += rimStripB * 0.44;
        g += rimStripB * 0.51;
        b += rimStripB * 0.65;
      }

      // 3. Low grazing horizon bounce strip (catches bay ripples & Sails Pavilion curves)
      const horizonStrip = evalSoftboxDir(dx, dy, dz, -0.48, 0.20, 0.85, 1.15, 0.18, 0.09);
      if (horizonStrip > 0.001) {
        r += horizonStrip * 0.89;
        g += horizonStrip * 0.90;
        b += horizonStrip * 0.94;
      }

      // 4. Dark negative-fill studio flags on opposite shadow side for rich 3:1 contrast
      const negFlagRight = evalSoftboxDir(dx, dy, dz, 0.86, 0.08, -0.50, 0.62, 0.72, 0.14);
      const negFlagBack = evalSoftboxDir(dx, dy, dz, 0.28, 0.04, -0.96, 0.55, 0.65, 0.14);
      const negFloor = smoothstep(0.05, -0.45, dy);
      const negMask = Math.max(negFlagRight * 0.84, negFlagBack * 0.76, negFloor * 0.72);

      r *= 1 - negMask * 0.85;
      g *= 1 - negMask * 0.85;
      b *= 1 - negMask * 0.85;

      // Gamma-like perceptual compression (power 1/1.6) so dark negative fill stays crisp in RGBA8
      // while preserving HDR softbox peaks up to `maxHdr` via hardware mipmapping
      const encR = Math.pow(clamp01(r / maxHdr), 1 / 1.6);
      const encG = Math.pow(clamp01(g / maxHdr), 1 / 1.6);
      const encB = Math.pow(clamp01(b / maxHdr), 1 / 1.6);

      const idx = (y * width + x) * 4;
      data[idx] = Math.round(encR * 255);
      data[idx + 1] = Math.round(encG * 255);
      data[idx + 2] = Math.round(encB * 255);
      data[idx + 3] = 255;
    }
  }

  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.NoColorSpace;
  texture.anisotropy = maxAnisotropy;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Bakes the procedural equirectangular studio environment into a Three.js PMREM cubemap
 * for built-in PBR materials, sharing the exact same softbox & negative-fill layout.
 */
function buildPmremFromStudioIbl(
  renderer: THREE.WebGLRenderer,
  softboxDiffuserMap: THREE.Texture,
): THREE.Texture {
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x08080a);

  const floorGeo = new THREE.PlaneGeometry(60, 60);
  const floorMat = new THREE.MeshBasicMaterial({ color: 0x0a0a0c });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -8;
  envScene.add(floor);

  const panels: Array<{
    w: number;
    h: number;
    pos: [number, number, number];
    lookAt: [number, number, number];
    color: number;
  }> = [
    // Overhead diffused softbox bank
    { w: 18, h: 11, pos: [-5, 16, 4], lookAt: [0, 0, 0], color: 0xf4f6fa },
    // Back-left vertical rim strip
    { w: 4.0, h: 22, pos: [-16, 7, -14], lookAt: [0, 2, 0], color: 0xffffff },
    // Front-right vertical rim strip
    { w: 3.4, h: 18, pos: [15, 6, 12], lookAt: [0, 1, 0], color: 0xdce2ec },
    // Grazing horizon strip
    { w: 22, h: 3.5, pos: [4, 2.8, -18], lookAt: [0, 0, 4], color: 0xe6e9ef },
  ];

  const createdGeos: THREE.BufferGeometry[] = [floorGeo];
  const createdMats: THREE.Material[] = [floorMat];

  for (const panel of panels) {
    const geo = new THREE.PlaneGeometry(panel.w, panel.h);
    const mat = new THREE.MeshBasicMaterial({
      color: panel.color,
      map: softboxDiffuserMap,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(...panel.pos);
    mesh.lookAt(...panel.lookAt);
    envScene.add(mesh);
    createdGeos.push(geo);
    createdMats.push(mat);
  }

  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileCubemapShader();
  const renderTarget = pmrem.fromScene(envScene, 0.04);

  for (const g of createdGeos) g.dispose();
  for (const m of createdMats) m.dispose();
  pmrem.dispose();

  return renderTarget.texture;
}

export function createStudioLightingRig(
  scene: THREE.Scene,
  renderer: THREE.WebGLRenderer,
  softboxDiffuserMap: THREE.Texture,
  isSoftwareGl: boolean,
  initialPreset: LightingPreset,
): StudioLightingRig {
  const maxAnisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy() || 1);
  const studioIblMap = createProceduralStudioIblMap(maxAnisotropy);
  const environmentTexture = buildPmremFromStudioIbl(renderer, softboxDiffuserMap);
  scene.environment = environmentTexture;
  scene.environmentIntensity = 1.08;

  // Cool sky / obsidian studio bounce fill (never flat white AmbientLight)
  const hemiLight = new THREE.HemisphereLight(0xd8dde8, 0x0b0b0d, 0.25);
  scene.add(hemiLight);

  let keySoftbox: THREE.RectAreaLight | null = null;
  let rimSoftbox: THREE.RectAreaLight | null = null;
  let fillSoftbox: THREE.RectAreaLight | null = null;
  let shadowKeyLight: THREE.DirectionalLight | null = null;

  if (!isSoftwareGl) {
    RectAreaLightUniformsLib.init();

    keySoftbox = new THREE.RectAreaLight(0xfaf7f2, 4.2, 16, 10);
    keySoftbox.position.set(-7, 16, 6);
    keySoftbox.lookAt(0, 1.5, 0);
    scene.add(keySoftbox);

    rimSoftbox = new THREE.RectAreaLight(0xfcfaf6, 5.5, 4.5, 20);
    rimSoftbox.position.set(-14, 8, -12);
    rimSoftbox.lookAt(0, 2.5, 0);
    scene.add(rimSoftbox);

    fillSoftbox = new THREE.RectAreaLight(0xc2ccd9, 2.2, 4.0, 16);
    fillSoftbox.position.set(14, 6, 11);
    fillSoftbox.lookAt(0, 1.0, 0);
    scene.add(fillSoftbox);

    // Subtle soft shadow caster only for grounding shadows on GPU
    shadowKeyLight = new THREE.DirectionalLight(0xf8f5ef, 0.42);
    shadowKeyLight.position.set(-10, 18, 8);
    shadowKeyLight.castShadow = true;
    shadowKeyLight.shadow.mapSize.set(1024, 1024);
    shadowKeyLight.shadow.camera.near = 2;
    shadowKeyLight.shadow.camera.far = 45;
    const d = 15;
    shadowKeyLight.shadow.camera.left = -d;
    shadowKeyLight.shadow.camera.right = d;
    shadowKeyLight.shadow.camera.top = d;
    shadowKeyLight.shadow.camera.bottom = -d;
    shadowKeyLight.shadow.bias = -0.0005;
    shadowKeyLight.shadow.normalBias = 0.025;
    scene.add(shadowKeyLight);
  }

  const applyPreset = (preset: LightingPreset) => {
    const r = 18;
    const elevAngle = preset.softboxKeyElevation * (Math.PI * 0.42);
    const y = Math.sin(elevAngle) * r;
    const horiz = Math.cos(elevAngle) * r;
    const x = Math.cos(preset.softboxKeyAzimuth) * horiz;
    const z = Math.sin(preset.softboxKeyAzimuth) * horiz;

    hemiLight.intensity = 0.22 * preset.fillIntensity + 0.08;

    if (keySoftbox) {
      keySoftbox.position.set(x, Math.max(5, y), z);
      keySoftbox.lookAt(0, 1.5, 0);
      keySoftbox.intensity = 4.2 * preset.keyIntensity;
    }
    if (rimSoftbox) {
      rimSoftbox.intensity = 5.5 * preset.rimIntensity;
    }
    if (fillSoftbox) {
      fillSoftbox.intensity = 2.2 * preset.fillIntensity;
    }
    if (shadowKeyLight) {
      shadowKeyLight.position.set(x, Math.max(6, y), z);
      shadowKeyLight.intensity = 0.42 * preset.keyIntensity;
    }
  };

  applyPreset(initialPreset);

  return {
    environmentTexture,
    studioIblMap,
    keySoftbox,
    rimSoftbox,
    fillSoftbox,
    hemiLight,
    shadowKeyLight,
    applyPreset,
    dispose: () => {
      environmentTexture.dispose();
      studioIblMap.dispose();
    },
  };
}
