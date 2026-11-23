import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  BASIN_RADIUS,
  HARBOR_SAILBOATS,
  HERO_LANDMARKS,
  LIGHTING_PRESETS,
  LightingPresetId,
  TowerSpec,
  classifyMidRiseBlock,
  generateCityGridBlocks,
  generateSeawallLights,
  seawallZ,
} from '../data/skylineLayout';
import { createProceduralTextureBundle, ProceduralTextureBundle } from './proceduralTextures';
import { createHdrPostProcessor, HdrPostProcessor } from './postProcessing';
import { createStudioLightingRig, StudioLightingRig } from './studioEnvironment';
import {
  SPRING_PRESETS,
  createSpringState,
  globalHardwareTelemetry,
  stepSpringRK4,
} from '../physics/springSolver';
import {
  createAnisotropicMetalMaterial,
  createCycloramaFloorMaterial,
  createHarborWaterMaterial,
  createLandmassObsidianMaterial,
  createMatteSlateMaterial,
  createSailFabricMaterial,
  createSdfGlassMaterial,
  createShadowDepthMaterial,
} from './shaders/skylineShaders';

export interface SkylineController {
  setLightingPreset: (presetId: LightingPresetId) => void;
  resetCamera: () => void;
  dispose: () => void;
  isSoftwareRenderer: boolean;
}

export interface SkylineCallbacks {
  onReady?: () => void;
  onContextLost?: () => void;
  onContextRestored?: () => void;
}

const DEFAULT_CAMERA_POS = new THREE.Vector3(24.6, 22.4, 34.9);
const DEFAULT_CAMERA_TARGET = new THREE.Vector3(-0.35, 1.15, 0.35);
const HERO_FOCAL_POINT = new THREE.Vector3(-2.1, 2.2, -0.2);

function computeFovForAspect(aspect: number): number {
  const baseFov = 31;
  if (aspect >= 1.22) return baseFov;
  const halfRad = THREE.MathUtils.degToRad(baseFov * 0.5);
  const adaptedHalf = Math.atan(Math.tan(halfRad) * (1.22 / Math.max(0.55, aspect)));
  return Math.min(48, THREE.MathUtils.radToDeg(adaptedHalf) * 2);
}

// Pre-allocated scratch helpers (zero GC inside render loop)
const _scratchDir = new THREE.Vector3();
const _scratchMat4 = new THREE.Matrix4();
const _scratchPos = new THREE.Vector3();
const _scratchQuat = new THREE.Quaternion();
const _scratchScale = new THREE.Vector3();
const _scratchEuler = new THREE.Euler();
const _savedCamPos = new THREE.Vector3();
const _savedCamQuat = new THREE.Quaternion();
const _parallaxOffset = new THREE.Vector3();
const _parallaxRight = new THREE.Vector3();
const _parallaxUpAxis = new THREE.Vector3(0, 1, 0);

function detectSoftwareWebGL(): boolean {
  try {
    const probeCanvas = document.createElement('canvas');
    const gl = probeCanvas.getContext('webgl2');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param) || '');
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

const _blackBg = new THREE.Color(0x000000);
const _shadowClearColor = new THREE.Color(0xffffff);
const _shadowMatrix = new THREE.Matrix4();

function addRoofParapetAndHvac(
  group: THREE.Group,
  w: number,
  d: number,
  topY: number,
  slateMaterial: THREE.ShaderMaterial,
  metalMaterial: THREE.ShaderMaterial,
  parapetH = 0.085,
  wallT = 0.065,
) {
  const fbGeo = new THREE.BoxGeometry(w, parapetH, wallT);
  const lrGeo = new THREE.BoxGeometry(wallT, parapetH, Math.max(0.05, d - wallT * 2));

  const frontWall = new THREE.Mesh(fbGeo, slateMaterial);
  frontWall.position.set(0, topY + parapetH * 0.5, d * 0.5 - wallT * 0.5);
  const backWall = new THREE.Mesh(fbGeo, slateMaterial);
  backWall.position.set(0, topY + parapetH * 0.5, -d * 0.5 + wallT * 0.5);
  const leftWall = new THREE.Mesh(lrGeo, slateMaterial);
  leftWall.position.set(-w * 0.5 + wallT * 0.5, topY + parapetH * 0.5, 0);
  const rightWall = new THREE.Mesh(lrGeo, slateMaterial);
  rightWall.position.set(w * 0.5 - wallT * 0.5, topY + parapetH * 0.5, 0);
  group.add(frontWall, backWall, leftWall, rightWall);

  // Clustered rooftop HVAC / mechanical unit inside the recessed parapet tray
  const hvacA = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.24, parapetH * 1.45, d * 0.22),
    slateMaterial,
  );
  hvacA.position.set(-w * 0.17, topY + parapetH * 0.72, d * 0.14);
  const hvacCap = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.16, parapetH * 0.55, d * 0.15),
    metalMaterial,
  );
  hvacCap.position.set(w * 0.18, topY + parapetH * 0.28, -d * 0.14);
  group.add(hvacA, hvacCap);
}

function createUnitParapetGeometry(): THREE.ExtrudeGeometry {
  const outer = new THREE.Shape();
  outer.moveTo(-0.5, -0.5);
  outer.lineTo(0.5, -0.5);
  outer.lineTo(0.5, 0.5);
  outer.lineTo(-0.5, 0.5);
  outer.closePath();

  const inner = new THREE.Path();
  inner.moveTo(-0.41, -0.41);
  inner.lineTo(-0.41, 0.41);
  inner.lineTo(0.41, 0.41);
  inner.lineTo(0.41, -0.41);
  inner.closePath();
  outer.holes.push(inner);

  const geo = new THREE.ExtrudeGeometry(outer, {
    depth: 1,
    bevelEnabled: false,
    steps: 1,
  });
  geo.rotateX(-Math.PI / 2);
  return geo;
}

function createCamberedSailGeometry(
  luffHeight: number,
  footLength: number,
  camber: number,
  isJib: boolean,
): THREE.BufferGeometry {
  const rows = 8;
  const cols = 4;
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  for (let r = 0; r <= rows; r++) {
    const v = r / rows; // 0 = foot, 1 = head
    const y = v * luffHeight;
    const chord = footLength * (1.0 - v * 0.94);
    for (let c = 0; c <= cols; c++) {
      const u = c / cols; // 0 = mast/luff, 1 = leech
      const z = isJib ? u * chord : -u * chord;
      // Smooth aerodynamic sail belly catching wind
      const belly = Math.sin(u * Math.PI) * Math.sin(v * Math.PI * 0.85 + 0.15) * camber;
      positions.push(belly, y, z);
      uvs.push(u, v);
    }
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const a = r * (cols + 1) + c;
      const b = a + 1;
      const d = a + (cols + 1);
      const e = d + 1;
      indices.push(a, b, d, b, e, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Smoothly chamfered architectural pyramid-to-cone crown for One America Plaza.
 * Transitions gradually from a rounded squircle (p=3.1) at the setback collar to
 * a circular machined apex collar, with averaged radial seam normals for silky
 * continuous light-to-shadow wrap instead of a stark two-tone facet split.
 */
function createSmoothPyramidCrownGeometry(
  baseRadius: number,
  height: number,
): THREE.BufferGeometry {
  const radialSegs = 40;
  const heightSegs = 10;
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const topRadius = 0.048;

  for (let r = 0; r <= heightSegs; r++) {
    const v = r / heightSegs;
    const y = v * height;
    const taper = Math.pow(1 - v, 0.92);
    const p = 3.1 - v * 1.1; // smooth transition from chamfered square to circular collar

    for (let c = 0; c <= radialSegs; c++) {
      const u = c / radialSegs;
      const theta = u * Math.PI * 2;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);
      const superDenom =
        Math.pow(Math.abs(cosT), p) + Math.pow(Math.abs(sinT), p);
      const superR = Math.pow(Math.max(1e-5, superDenom), -1 / p);
      const rad = (topRadius * v + baseRadius * taper * superR) / (1 + v * 0.02);

      positions.push(cosT * rad, y, sinT * rad);
      uvs.push(u, v);
    }
  }

  const stride = radialSegs + 1;
  for (let r = 0; r < heightSegs; r++) {
    for (let c = 0; c < radialSegs; c++) {
      const a = r * stride + c;
      const b = a + 1;
      const d = a + stride;
      const e = d + 1;
      indices.push(a, d, b, b, d, e);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();

  // Weld seam normals (c = 0 and c = radialSegs) so circular wrap has zero seam crease
  const normAttr = geo.getAttribute('normal') as THREE.BufferAttribute;
  for (let r = 0; r <= heightSegs; r++) {
    const i0 = r * stride;
    const i1 = i0 + radialSegs;
    const nx = (normAttr.getX(i0) + normAttr.getX(i1)) * 0.5;
    const ny = (normAttr.getY(i0) + normAttr.getY(i1)) * 0.5;
    const nz = (normAttr.getZ(i0) + normAttr.getZ(i1)) * 0.5;
    const len = Math.hypot(nx, ny, nz) || 1;
    normAttr.setXYZ(i0, nx / len, ny / len, nz / len);
    normAttr.setXYZ(i1, nx / len, ny / len, nz / len);
  }
  normAttr.needsUpdate = true;

  return geo;
}

function buildCompositeHeroTower(
  spec: TowerSpec,
  glassMaterial: THREE.ShaderMaterial,
  metalMaterial: THREE.ShaderMaterial,
  slateMaterial: THREE.ShaderMaterial,
  lobbyGlassMaterial: THREE.ShaderMaterial = glassMaterial,
): THREE.Group {
  const group = new THREE.Group();
  group.position.set(spec.x, 0.08, spec.z);
  group.rotation.y = spec.rotationY;

  const { width, depth, height, style } = spec;

  // 1. Grounded dead-matte slate podium frame with recessed warm-lit double-height lobby glazing
  const podiumH = 0.45;
  const podiumBase = new THREE.Mesh(
    new THREE.BoxGeometry(width * 1.16, 0.14, depth * 1.16),
    slateMaterial,
  );
  podiumBase.position.y = 0.07;
  group.add(podiumBase);

  const lobbyCore = new THREE.Mesh(
    new THREE.BoxGeometry(width * 1.11, 0.24, depth * 1.11),
    lobbyGlassMaterial,
  );
  lobbyCore.position.y = 0.26;
  group.add(lobbyCore);

  const pierGeo = new THREE.BoxGeometry(0.11, 0.245, 0.11);
  for (const px of [-1, 1]) {
    for (const pz of [-1, 1]) {
      const pier = new THREE.Mesh(pierGeo, slateMaterial);
      pier.position.set(
        px * (width * 0.58 - 0.055),
        0.26,
        pz * (depth * 0.58 - 0.055),
      );
      group.add(pier);
    }
  }

  const podiumCrown = new THREE.Mesh(
    new THREE.BoxGeometry(width * 1.16, 0.075, depth * 1.16),
    slateMaterial,
  );
  podiumCrown.position.y = 0.4125;
  group.add(podiumCrown);

  if (style === 'obelisk') {
    // One America Plaza / Brookfield style: chamfered glass shaft + vertical pilasters + matte collar + smoothly machined crown + detailed antenna collar & mast
    const shaftH = height * 0.74;
    const shaftGeo = new THREE.CylinderGeometry(
      width * 0.54,
      width * 0.62,
      shaftH,
      8,
      1,
    );
    const shaft = new THREE.Mesh(shaftGeo, glassMaterial);
    shaft.position.y = podiumH + shaftH * 0.5;
    shaft.rotation.y = Math.PI / 8;
    group.add(shaft);

    // Stepped vertical window mullions / structural ribs around octagonal faces
    const pilasterGeo = new THREE.BoxGeometry(0.048, shaftH * 0.99, 0.048);
    for (let k = 0; k < 8; k++) {
      const ang = (k * Math.PI) / 4 + Math.PI / 8;
      const pilaster = new THREE.Mesh(pilasterGeo, slateMaterial);
      const rad = width * 0.57;
      pilaster.position.set(Math.cos(ang) * rad, podiumH + shaftH * 0.5, Math.sin(ang) * rad);
      pilaster.rotation.y = -ang;
      group.add(pilaster);
    }

    // Dead-matte anodized setback collar
    const collarH = 0.52;
    const collarGeo = new THREE.CylinderGeometry(width * 0.44, width * 0.54, collarH, 8);
    const collar = new THREE.Mesh(collarGeo, slateMaterial);
    collar.position.y = podiumH + shaftH + collarH * 0.5;
    collar.rotation.y = Math.PI / 8;
    group.add(collar);

    // Smoothly chamfered brushed metal/dark-glass pyramid crown (silky continuous light wrap)
    const crownH = height * 0.19;
    const crownGeo = createSmoothPyramidCrownGeometry(width * 0.375, crownH);
    const crown = new THREE.Mesh(crownGeo, metalMaterial);
    crown.position.y = podiumH + shaftH + collarH;
    group.add(crown);

    // Multi-tiered precision-machined upper antenna collar & spire (One America Plaza signature)
    const crownTop = podiumH + shaftH + collarH + crownH;
    const lowerFlange = new THREE.Mesh(
      new THREE.CylinderGeometry(0.088, 0.106, 0.068, 24),
      slateMaterial,
    );
    lowerFlange.position.y = crownTop - 0.025;
    group.add(lowerFlange);

    const shadowGap = new THREE.Mesh(
      new THREE.CylinderGeometry(0.064, 0.064, 0.030, 24),
      slateMaterial,
    );
    shadowGap.position.y = crownTop + 0.022;
    group.add(shadowGap);

    const upperCollar = new THREE.Mesh(
      new THREE.CylinderGeometry(0.046, 0.082, 0.092, 24),
      metalMaterial,
    );
    upperCollar.position.y = crownTop + 0.082;
    group.add(upperCollar);

    const mastFerrule = new THREE.Mesh(
      new THREE.CylinderGeometry(0.036, 0.046, 0.042, 16),
      slateMaterial,
    );
    mastFerrule.position.y = crownTop + 0.148;
    group.add(mastFerrule);

    const mastH = 1.12;
    const mast = new THREE.Mesh(
      new THREE.CylinderGeometry(0.008, 0.035, mastH, 16),
      metalMaterial,
    );
    mast.position.y = crownTop + 0.168 + mastH * 0.5;
    group.add(mast);

    const beaconCollar = new THREE.Mesh(
      new THREE.CylinderGeometry(0.017, 0.017, 0.045, 12),
      slateMaterial,
    );
    beaconCollar.position.y = crownTop + 0.168 + mastH * 0.62;
    group.add(beaconCollar);

    const needle = new THREE.Mesh(
      new THREE.CylinderGeometry(0.003, 0.008, 0.28, 12),
      metalMaterial,
    );
    needle.position.y = crownTop + 0.168 + mastH + 0.14;
    group.add(needle);
  } else if (style === 'emerald-cluster') {
    // Emerald Plaza: 6 clustered hexagonal glass columns at staggered heights + metallic bevel caps & corner pilasters
    const offsets: Array<[number, number, number]> = [
      [0, 0, 1.0],
      [-0.42, 0.26, 0.86],
      [0.42, 0.24, 0.92],
      [-0.4, -0.28, 0.74],
      [0.4, -0.26, 0.8],
      [0, -0.45, 0.66],
    ];
    for (const [ox, oz, scaleH] of offsets) {
      const colH = (height - podiumH) * scaleH;
      const colGeo = new THREE.CylinderGeometry(width * 0.24, width * 0.24, colH, 6);
      const col = new THREE.Mesh(colGeo, glassMaterial);
      const cx = ox * width * 0.68;
      const cz = oz * depth * 0.68;
      col.position.set(cx, podiumH + colH * 0.5, cz);
      group.add(col);

      // Recessed parapet ring & satin brushed metallic bevel cap
      const ringGeo = new THREE.CylinderGeometry(width * 0.252, width * 0.252, 0.09, 6);
      const ring = new THREE.Mesh(ringGeo, slateMaterial);
      ring.position.set(cx, podiumH + colH + 0.045, cz);
      group.add(ring);

      const capGeo = new THREE.CylinderGeometry(width * 0.17, width * 0.23, 0.20, 6);
      const cap = new THREE.Mesh(capGeo, metalMaterial);
      cap.position.set(cx, podiumH + colH + 0.18, cz);
      group.add(cap);
    }
  } else if (style === 'chamfered-twin') {
    // Harbor Club / Pinnacle style: faceted vertical tower with satin metallic fins, recessed mullions & lantern
    const mainH = height * 0.84;
    const shaftGeo = new THREE.CylinderGeometry(width * 0.56, width * 0.58, mainH, 10);
    const shaft = new THREE.Mesh(shaftGeo, glassMaterial);
    shaft.position.y = podiumH + mainH * 0.5;
    group.add(shaft);

    // Anisotropic vertical ribs & stepped recessed vertical mullions
    const ribGeo = new THREE.BoxGeometry(width * 0.14, mainH * 1.02, depth * 1.06);
    const ribA = new THREE.Mesh(ribGeo, metalMaterial);
    ribA.position.y = podiumH + mainH * 0.5;
    group.add(ribA);

    const mullGeo = new THREE.BoxGeometry(0.042, mainH * 0.98, 0.042);
    for (const mx of [-width * 0.34, width * 0.34]) {
      for (const mz of [-depth * 0.46, depth * 0.46]) {
        const m = new THREE.Mesh(mullGeo, slateMaterial);
        m.position.set(mx, podiumH + mainH * 0.5, mz);
        group.add(m);
      }
    }

    // Crown lantern + slender finial mast
    const lanternH = height * 0.14;
    const lanternGeo = new THREE.CylinderGeometry(width * 0.34, width * 0.48, lanternH, 10);
    const lantern = new THREE.Mesh(lanternGeo, metalMaterial);
    lantern.position.y = podiumH + mainH + lanternH * 0.5;
    group.add(lantern);

    const finial = new THREE.Mesh(
      new THREE.CylinderGeometry(0.008, 0.022, 0.52, 6),
      metalMaterial,
    );
    finial.position.y = podiumH + mainH + lanternH + 0.26;
    group.add(finial);
  } else if (style === 'curved-slab') {
    // Manchester Grand Hyatt: wide architectural slab with stepped setback, vertical mullions, recessed parapet & HVAC
    const lowerH = height * 0.68;
    const upperH = height * 0.26;
    const lowerGeo = new THREE.BoxGeometry(width, lowerH, depth);
    const lower = new THREE.Mesh(lowerGeo, glassMaterial);
    lower.position.y = podiumH + lowerH * 0.5;
    group.add(lower);

    // Stepped vertical window mullions across front & back curtain walls
    const lowerMullGeo = new THREE.BoxGeometry(0.045, lowerH * 0.99, depth + 0.055);
    for (const mx of [-width * 0.28, 0, width * 0.28]) {
      const m = new THREE.Mesh(lowerMullGeo, slateMaterial);
      m.position.set(mx, podiumH + lowerH * 0.5, 0);
      group.add(m);
    }

    // Setback terrace parapet lip around lower roof
    addRoofParapetAndHvac(
      group,
      width,
      depth,
      podiumH + lowerH,
      slateMaterial,
      metalMaterial,
      0.075,
      0.058,
    );

    const upperGeo = new THREE.BoxGeometry(width * 0.78, upperH, depth * 0.78);
    const upper = new THREE.Mesh(upperGeo, glassMaterial);
    upper.position.y = podiumH + lowerH + upperH * 0.5;
    group.add(upper);

    // Upper slab vertical mullions + recessed top parapet & HVAC penthouse
    const upperMullGeo = new THREE.BoxGeometry(0.04, upperH * 0.99, depth * 0.78 + 0.05);
    for (const mx of [-width * 0.20, width * 0.20]) {
      const m = new THREE.Mesh(upperMullGeo, metalMaterial);
      m.position.set(mx, podiumH + lowerH + upperH * 0.5, 0);
      group.add(m);
    }

    addRoofParapetAndHvac(
      group,
      width * 0.78,
      depth * 0.78,
      podiumH + lowerH + upperH,
      slateMaterial,
      metalMaterial,
      0.085,
      0.06,
    );
  } else {
    // Stepped classic tower with recessed front vertical mullions, parapet lips & spire
    const tier1H = height * 0.65;
    const tier2H = height * 0.28;
    const tier1Geo = new THREE.BoxGeometry(width, tier1H, depth);
    const tier1 = new THREE.Mesh(tier1Geo, glassMaterial);
    tier1.position.y = podiumH + tier1H * 0.5;
    group.add(tier1);

    const tier1MullGeo = new THREE.BoxGeometry(0.042, tier1H * 0.99, depth + 0.05);
    for (const mx of [-width * 0.26, width * 0.26]) {
      const m = new THREE.Mesh(tier1MullGeo, slateMaterial);
      m.position.set(mx, podiumH + tier1H * 0.5, 0);
      group.add(m);
    }

    const tier2Geo = new THREE.BoxGeometry(width * 0.74, tier2H, depth * 0.74);
    const tier2 = new THREE.Mesh(tier2Geo, glassMaterial);
    tier2.position.y = podiumH + tier1H + tier2H * 0.5;
    group.add(tier2);

    addRoofParapetAndHvac(
      group,
      width * 0.74,
      depth * 0.74,
      podiumH + tier1H + tier2H,
      slateMaterial,
      metalMaterial,
      0.075,
      0.055,
    );

    const spireGeo = new THREE.CylinderGeometry(0.012, 0.048, 0.72, 8);
    const spire = new THREE.Mesh(spireGeo, metalMaterial);
    spire.position.y = podiumH + tier1H + tier2H + 0.36;
    group.add(spire);
  }

  return group;
}

function buildConventionCenter(
  glassMaterial: THREE.ShaderMaterial,
  metalMaterial: THREE.ShaderMaterial,
  slateMaterial: THREE.ShaderMaterial,
  terminalArchGlassMaterial: THREE.ShaderMaterial = glassMaterial,
): THREE.Group {
  const group = new THREE.Group();
  group.position.set(7.4, 0.08, 0.45);
  group.rotation.y = 0.28;

  // Main long exhibition hall base
  const hallW = 5.6;
  const hallD = 2.6;
  const hallH = 0.88;
  const hallGeo = new THREE.BoxGeometry(hallW, hallH, hallD);
  const hall = new THREE.Mesh(hallGeo, glassMaterial);
  hall.position.y = hallH * 0.5;
  group.add(hall);

  // Structural vertical facade mullions along waterfront exhibition glass wall
  const facadeMullGeo = new THREE.BoxGeometry(0.045, hallH * 0.96, 0.055);
  for (let i = -5; i <= 5; i++) {
    const mx = i * 0.48;
    const frontM = new THREE.Mesh(facadeMullGeo, slateMaterial);
    frontM.position.set(mx, hallH * 0.5, hallD * 0.5 + 0.02);
    group.add(frontM);
  }

  // Dead-matte slate waterfront colonnade overhang + recessed rooftop parapet
  const canopyGeo = new THREE.BoxGeometry(hallW * 1.04, 0.14, 0.55);
  const canopy = new THREE.Mesh(canopyGeo, slateMaterial);
  canopy.position.set(0, hallH * 0.72, hallD * 0.5 + 0.24);
  group.add(canopy);

  addRoofParapetAndHvac(
    group,
    hallW * 0.98,
    hallD * 0.96,
    hallH,
    slateMaterial,
    metalMaterial,
    0.07,
    0.055,
  );

  // 5 iconic vaulted Sails Pavilion roofs along the upper deck + recessed softly lit vaulted bays
  const sailShape = new THREE.Shape();
  sailShape.moveTo(-0.48, 0);
  sailShape.quadraticCurveTo(0, 0.92, 0.48, 0);
  sailShape.lineTo(0.42, 0);
  sailShape.quadraticCurveTo(0, 0.78, -0.42, 0);
  sailShape.closePath();

  const sailGeo = new THREE.ExtrudeGeometry(sailShape, {
    depth: 1.75,
    curveSegments: 24,
    bevelEnabled: true,
    bevelSegments: 3,
    steps: 1,
    bevelSize: 0.03,
    bevelThickness: 0.03,
  });
  sailGeo.center();

  const archCoreShape = new THREE.Shape();
  archCoreShape.moveTo(-0.41, 0);
  archCoreShape.quadraticCurveTo(0, 0.76, 0.41, 0);
  archCoreShape.closePath();

  const archCoreGeo = new THREE.ExtrudeGeometry(archCoreShape, {
    depth: 1.44,
    curveSegments: 20,
    bevelEnabled: false,
    steps: 1,
  });
  archCoreGeo.center();

  const archRibGeo = new THREE.BoxGeometry(0.032, 0.78, 0.04);

  for (let i = 0; i < 5; i++) {
    const xOffset = -2.05 + i * 1.02;
    const sailMesh = new THREE.Mesh(sailGeo, metalMaterial);
    sailMesh.position.set(xOffset, hallH + 0.44, -0.1);
    group.add(sailMesh);

    const archBayMesh = new THREE.Mesh(archCoreGeo, terminalArchGlassMaterial);
    archBayMesh.position.set(xOffset, hallH + 0.39, -0.1);
    group.add(archBayMesh);

    const centerMullion = new THREE.Mesh(archRibGeo, slateMaterial);
    centerMullion.position.set(xOffset, hallH + 0.40, 0.78);
    group.add(centerMullion);
  }

  return group;
}

function buildMidwayPier(
  metalMaterial: THREE.ShaderMaterial,
  slateMaterial: THREE.ShaderMaterial,
): THREE.Group {
  const group = new THREE.Group();
  group.position.set(-10.5, 0.05, 2.55);
  group.rotation.y = 0.55;

  // Dead-matte Navy Pier concrete apron
  const pierGeo = new THREE.BoxGeometry(3.8, 0.18, 1.05);
  const pier = new THREE.Mesh(pierGeo, slateMaterial);
  pier.position.set(-0.6, 0.09, 0);
  group.add(pier);

  // Satin metal carrier hull
  const hullShape = new THREE.Shape();
  hullShape.moveTo(-1.7, -0.26);
  hullShape.lineTo(1.5, -0.26);
  hullShape.lineTo(1.85, 0);
  hullShape.lineTo(1.5, 0.26);
  hullShape.lineTo(-1.7, 0.26);
  hullShape.closePath();

  const hullGeo = new THREE.ExtrudeGeometry(hullShape, {
    depth: 0.28,
    bevelEnabled: true,
    bevelSegments: 2,
    steps: 1,
    bevelSize: 0.022,
    bevelThickness: 0.022,
  });
  hullGeo.rotateX(-Math.PI / 2);

  const hull = new THREE.Mesh(hullGeo, metalMaterial);
  hull.position.set(-0.95, 0.14, 0.58);
  group.add(hull);

  // Angled flight deck
  const deckGeo = new THREE.BoxGeometry(3.6, 0.08, 0.72);
  const deck = new THREE.Mesh(deckGeo, slateMaterial);
  deck.position.set(-0.95, 0.44, 0.58);
  group.add(deck);

  // Island bridge superstructure
  const islandGeo = new THREE.BoxGeometry(0.45, 0.36, 0.18);
  const island = new THREE.Mesh(islandGeo, metalMaterial);
  island.position.set(-0.7, 0.65, 0.34);
  group.add(island);

  return group;
}

function buildLandmassGeometry(): THREE.ExtrudeGeometry {
  const shape = new THREE.Shape();
  const r = BASIN_RADIUS + 0.08;
  const segments = 160;

  // Trace seawall from left rim to right rim, then arc smoothly around the north/east/west perimeter
  const xStart = -r * 0.992;
  const xEnd = r * 0.992;

  const firstZ = Math.min(seawallZ(xStart), Math.sqrt(Math.max(0.05, r * r - xStart * xStart)));
  shape.moveTo(xStart, -firstZ);

  for (let i = 1; i <= segments; i++) {
    const x = xStart + (i / segments) * (xEnd - xStart);
    const maxZ = Math.sqrt(Math.max(0.05, r * r - x * x));
    const z = Math.min(seawallZ(x), maxZ);
    // Note: Shape is in XY and rotated -PI/2 around X, so shape.y = -worldZ
    shape.lineTo(x, -z);
  }

  // Arc cleanly around the rear/right perimeter flush inside the plinth basin wall
  const rightAngle = Math.atan2(-Math.min(seawallZ(xEnd), Math.sqrt(Math.max(0.05, r * r - xEnd * xEnd))), xEnd);
  let leftAngle = Math.atan2(-firstZ, xStart);
  if (leftAngle < rightAngle) {
    leftAngle += Math.PI * 2;
  }
  const arcSteps = 128;
  for (let i = 1; i <= arcSteps; i++) {
    const t = i / arcSteps;
    const angle = rightAngle + t * (leftAngle - rightAngle);
    shape.lineTo(Math.cos(angle) * r, Math.sin(angle) * r);
  }
  shape.closePath();

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: 0.16,
    bevelEnabled: true,
    bevelSegments: 2,
    steps: 1,
    bevelSize: 0.04,
    bevelThickness: 0.04,
  });
  geo.rotateX(-Math.PI / 2);
  return geo;
}

/**
 * Clean swept dark architectural seawall & beveled concrete coping ribbon
 * along the Embarcadero waterfront curve, plus an inner perimeter curb
 * where the city terrain meets the plinth ring so zero jagged grid edges appear.
 */
function buildSeawallCopingGeometry(): THREE.BufferGeometry {
  const rWall = BASIN_RADIUS + 0.12;
  const xSpan = rWall * 0.988;
  const steps = 160;

  // Cross-section profile in (outwardOffset, y): land-side edge -> beveled coping cap -> battered seawall toe
  const profile: Array<[number, number]> = [
    [-0.25, 0.145],
    [-0.22, 0.184],
    [0.055, 0.184],
    [0.080, 0.156],
    [0.088, -0.025],
  ];
  const profLen = profile.length;

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const pts: Array<{ x: number; z: number }> = [];
  for (let i = 0; i <= steps; i++) {
    const x = -xSpan + (i / steps) * (xSpan * 2);
    const maxZ = Math.sqrt(Math.max(0.02, rWall * rWall - x * x));
    const z = Math.min(seawallZ(x), maxZ);
    pts.push({ x, z });
  }

  for (let i = 0; i <= steps; i++) {
    const prev = pts[Math.max(0, i - 1)];
    const next = pts[Math.min(steps, i + 1)];
    let tx = next.x - prev.x;
    let tz = next.z - prev.z;
    const tLen = Math.hypot(tx, tz) || 1;
    tx /= tLen;
    tz /= tLen;

    // Outward normal facing bay water (+Z side)
    let nx = -tz;
    let nz = tx;
    if (nz < 0) {
      nx = -nx;
      nz = -nz;
    }

    const u = i / steps;
    const cx = pts[i].x;
    const cz = pts[i].z;

    for (let k = 0; k < profLen; k++) {
      const [off, py] = profile[k];
      const wx = cx + nx * off;
      const wz = cz + nz * off;
      // Clamp outer vertices strictly inside basin rim wall
      const rad = Math.hypot(wx, wz);
      const scale = rad > rWall + 0.02 ? (rWall + 0.02) / rad : 1;
      positions.push(wx * scale, py, wz * scale);
      uvs.push(u * 24, k / (profLen - 1));
    }
  }

  for (let i = 0; i < steps; i++) {
    for (let k = 0; k < profLen - 1; k++) {
      const a = i * profLen + k;
      const b = a + 1;
      const d = a + profLen;
      const e = d + 1;
      indices.push(a, d, b, b, d, e);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Clean beveled concrete retaining curb tracing ONLY the land arc along the
 * rear/side perimeter inside the plinth rim, meeting both ends of the Embarcadero seawall coping.
 */
function buildPerimeterCurbGeometry(): THREE.BufferGeometry {
  const r = BASIN_RADIUS + 0.10;
  const xStart = -r * 0.992;
  const xEnd = r * 0.992;
  const firstZ = Math.min(seawallZ(xStart), Math.sqrt(Math.max(0.05, r * r - xStart * xStart)));
  const lastZ = Math.min(seawallZ(xEnd), Math.sqrt(Math.max(0.05, r * r - xEnd * xEnd)));

  const rightAngle = Math.atan2(-lastZ, xEnd);
  let leftAngle = Math.atan2(-firstZ, xStart);
  if (leftAngle < rightAngle) {
    leftAngle += Math.PI * 2;
  }

  const steps = 128;
  // Cross-section in (radialOffsetFromR, y): inward on land -> beveled cap -> outer plinth wall
  const profile: Array<[number, number]> = [
    [-0.24, 0.145],
    [-0.20, 0.184],
    [0.015, 0.184],
    [0.04, 0.155],
    [0.045, 0.015],
  ];
  const profLen = profile.length;

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const angle = rightAngle + t * (leftAngle - rightAngle);
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    for (let k = 0; k < profLen; k++) {
      const [radOff, py] = profile[k];
      const rad = r + radOff;
      positions.push(cosA * rad, py, -sinA * rad);
      uvs.push(t * 28, k / (profLen - 1));
    }
  }

  for (let i = 0; i < steps; i++) {
    for (let k = 0; k < profLen - 1; k++) {
      const a = i * profLen + k;
      const b = a + 1;
      const d = a + profLen;
      const e = d + 1;
      indices.push(a, d, b, b, d, e);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

export function createSanDiegoSkylineScene(
  container: HTMLElement,
  callbacks: SkylineCallbacks = {},
): SkylineController {
  const isSoftwareGl = detectSoftwareWebGL();
  const reducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

  const scene = new THREE.Scene();
  scene.background = null;
  scene.fog = new THREE.FogExp2(0x131315, 0.0068);

  const width = Math.max(container.clientWidth, 320);
  const height = Math.max(container.clientHeight, 320);

  const camera = new THREE.PerspectiveCamera(
    computeFovForAspect(width / height),
    width / height,
    0.1,
    160,
  );
  camera.position.copy(DEFAULT_CAMERA_POS);

  const renderer = new THREE.WebGLRenderer({
    antialias: !isSoftwareGl,
    powerPreference: 'high-performance',
    alpha: true,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  });
  const initialDpr = isSoftwareGl ? 1 : Math.min(window.devicePixelRatio || 1, 2);
  renderer.setSize(width, height);
  renderer.setPixelRatio(initialDpr);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // Tone mapping & linear->sRGB gamma are executed in the RGBA16F AgX post-processing pass
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.shadowMap.enabled = !isSoftwareGl;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const postProcessor: HdrPostProcessor = createHdrPostProcessor(
    renderer,
    width,
    height,
    renderer.getPixelRatio(),
    isSoftwareGl,
  );

  renderer.domElement.style.display = 'block';
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  container.appendChild(renderer.domElement);

  const onLost = (e: Event) => {
    e.preventDefault();
    callbacks.onContextLost?.();
  };
  const onRestored = () => {
    callbacks.onContextRestored?.();
    requestRender();
  };
  renderer.domElement.addEventListener('webglcontextlost', onLost);
  renderer.domElement.addEventListener('webglcontextrestored', onRestored);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(DEFAULT_CAMERA_TARGET);
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.enableDamping = !isSoftwareGl;
  controls.dampingFactor = 0.06;
  controls.rotateSpeed = 0.55;
  controls.minPolarAngle = 0.45;
  controls.maxPolarAngle = 1.24;
  controls.update();

  const cityBlocks = generateCityGridBlocks(2026);
  const textures: ProceduralTextureBundle = createProceduralTextureBundle(
    renderer,
    HERO_LANDMARKS,
    cityBlocks,
  );
  const initialPreset = LIGHTING_PRESETS['overhead-studio'];
  const lightingRig: StudioLightingRig = createStudioLightingRig(
    scene,
    renderer,
    textures.softboxDiffuserMap,
    isSoftwareGl,
    initialPreset,
  );

  const seawallLights = generateSeawallLights(32);
  const heroLobbyZones = [
    ...HERO_LANDMARKS.map((t) => ({
      x: t.x,
      z: t.z,
      radius: Math.max(t.width, t.depth) * 0.62,
      strength: 0.92,
    })),
    { x: 7.4, z: 1.45, radius: 1.55, strength: 0.62 },
  ];

  // Shared monochrome architectural scale-model materials (zero wireframe edge lines)
  const heroGlassMat = createSdfGlassMaterial(
    textures.architecturalGrainMap,
    lightingRig.studioIblMap,
    textures.microImperfectionMap,
    textures.groundContactAoMap,
    BASIN_RADIUS,
    {
      mullionDensity: 13,
      floorDensity: 17,
      baseTint: 0x0a0c10,
      edgeLuminance: 0.92,
      isHero: true,
      allowSideChamfers: true,
    },
  );

  const urbanGridGlassMat = createSdfGlassMaterial(
    textures.architecturalGrainMap,
    lightingRig.studioIblMap,
    textures.microImperfectionMap,
    textures.groundContactAoMap,
    BASIN_RADIUS,
    {
      mullionDensity: 9,
      floorDensity: 12,
      baseTint: 0x080a0d,
      edgeLuminance: 0.68,
      isHero: false,
      allowSideChamfers: true,
    },
  );

  const heroLobbyGlassMat = createSdfGlassMaterial(
    textures.architecturalGrainMap,
    lightingRig.studioIblMap,
    textures.microImperfectionMap,
    textures.groundContactAoMap,
    BASIN_RADIUS,
    {
      mullionDensity: 15,
      floorDensity: 8,
      baseTint: 0x0a0c10,
      edgeLuminance: 0.78,
      isHero: true,
      allowSideChamfers: true,
      interiorMode: 'lobby',
    },
  );

  const terminalArchGlassMat = createSdfGlassMaterial(
    textures.architecturalGrainMap,
    lightingRig.studioIblMap,
    textures.microImperfectionMap,
    textures.groundContactAoMap,
    BASIN_RADIUS,
    {
      mullionDensity: 14,
      floorDensity: 6,
      baseTint: 0x090b0f,
      edgeLuminance: 0.56,
      isHero: true,
      allowSideChamfers: false,
      interiorMode: 'terminal-arch',
    },
  );

  const matteSlateMat = createMatteSlateMaterial(
    textures.architecturalGrainMap,
    textures.groundContactAoMap,
    textures.microImperfectionMap,
    lightingRig.studioIblMap,
    BASIN_RADIUS,
    { baseColor: 0x1a1c22, masonryWindows: false, seawallLampXZ: seawallLights },
  );

  const masonrySlateMat = createMatteSlateMaterial(
    textures.architecturalGrainMap,
    textures.groundContactAoMap,
    textures.microImperfectionMap,
    lightingRig.studioIblMap,
    BASIN_RADIUS,
    { baseColor: 0x20232b, masonryWindows: true },
  );

  const brushedMetalMat = createAnisotropicMetalMaterial(
    textures.brushedMetalMap,
    textures.architecturalGrainMap,
    lightingRig.studioIblMap,
    textures.microImperfectionMap,
    textures.groundContactAoMap,
    BASIN_RADIUS,
    {
      baseColor: 0x353841,
      radialAnisotropy: false,
      anisotropyStrength: 0.82,
      roughness: 0.35,
      allowSideChamfers: true,
    },
  );

  const plinthMetalMat = createAnisotropicMetalMaterial(
    textures.brushedMetalMap,
    textures.architecturalGrainMap,
    lightingRig.studioIblMap,
    textures.microImperfectionMap,
    textures.groundContactAoMap,
    BASIN_RADIUS,
    {
      baseColor: 0x17181d,
      radialAnisotropy: true,
      anisotropyStrength: 0.85,
      roughness: 0.36,
      allowSideChamfers: false,
    },
  );

  // Grounded studio cyclorama tabletop beneath the turntable plinth
  const cycloramaMat = createCycloramaFloorMaterial(
    textures.architecturalGrainMap,
    BASIN_RADIUS,
    textures.microImperfectionMap,
  );
  const cycloramaGeo = new THREE.CircleGeometry(BASIN_RADIUS + 10.5, 112);
  cycloramaGeo.rotateX(-Math.PI / 2);
  const cycloramaMesh = new THREE.Mesh(cycloramaGeo, cycloramaMat);
  cycloramaMesh.position.y = -0.441;
  cycloramaMesh.renderOrder = -1;
  scene.add(cycloramaMesh);

  // Physical diorama suspension root (depresses & tilts smoothly on RK4 springs)
  const dioramaRoot = new THREE.Group();
  scene.add(dioramaRoot);

  // 1. Precision-Machined Circular Anodized Diorama Plinth with 45° Chamfered Rim & Recessed Basin
  const plinthProfile = [
    new THREE.Vector2(0.0, -0.44),
    new THREE.Vector2(BASIN_RADIUS + 0.32, -0.44),
    new THREE.Vector2(BASIN_RADIUS + 0.34, -0.415),
    new THREE.Vector2(BASIN_RADIUS + 0.34, -0.39),
    new THREE.Vector2(BASIN_RADIUS + 0.25, -0.048),
    new THREE.Vector2(BASIN_RADIUS + 0.18, 0.024),
    new THREE.Vector2(BASIN_RADIUS + 0.145, 0.024),
    new THREE.Vector2(BASIN_RADIUS + 0.14, -0.025),
    new THREE.Vector2(0.0, -0.025),
  ];
  const plinthGeo = new THREE.LatheGeometry(plinthProfile, 128);
  plinthGeo.computeVertexNormals();
  const plinth = new THREE.Mesh(plinthGeo, plinthMetalMat);
  plinth.receiveShadow = true;
  dioramaRoot.add(plinth);

  const rimRingGeo = new THREE.TorusGeometry(BASIN_RADIUS + 0.155, 0.014, 12, 128);
  rimRingGeo.rotateX(Math.PI / 2);
  const rimRing = new THREE.Mesh(rimRingGeo, plinthMetalMat);
  rimRing.position.y = 0.020;
  dioramaRoot.add(rimRing);

  // 2. Urban Landmass with dead-matte slate/concrete, canyon AO, raymarched cast shadows & curved seawall coping
  const landGeo = buildLandmassGeometry();
  const landMat = createLandmassObsidianMaterial(
    textures.architecturalGrainMap,
    textures.groundContactAoMap,
    textures.microImperfectionMap,
    lightingRig.studioIblMap,
    BASIN_RADIUS,
    seawallLights,
    heroLobbyZones,
  );
  const landmass = new THREE.Mesh(landGeo, landMat);
  landmass.position.y = -0.04;
  dioramaRoot.add(landmass);

  const seawallCopingGeo = buildSeawallCopingGeometry();
  const seawallCoping = new THREE.Mesh(seawallCopingGeo, matteSlateMat);
  dioramaRoot.add(seawallCoping);

  // Inner curved concrete curb ring along the right/rear terrain perimeter inside the plinth rim
  const perimeterCurbGeo = buildPerimeterCurbGeometry();
  const perimeterCurb = new THREE.Mesh(perimeterCurbGeo, matteSlateMat);
  dioramaRoot.add(perimeterCurb);

  // 3. Harbor Water Plane with Real-Time Planar Reflections, Kelvin Wakes & Esplanade Pinpoints
  const WATER_Y = 0.015;
  const reflW = Math.max(256, Math.floor(width * initialDpr * 0.5));
  const reflH = Math.max(256, Math.floor(height * initialDpr * 0.5));
  const reflectionTarget = new THREE.WebGLRenderTarget(reflW, reflH, {
    type: postProcessor.isHdrFloatBuffer ? THREE.HalfFloatType : THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: true,
  });
  const reflectCamera = new THREE.PerspectiveCamera(
    computeFovForAspect(width / height),
    width / height,
    0.1,
    160,
  );

  const waterMat = createHarborWaterMaterial(
    textures.waterMicroRippleMap,
    lightingRig.studioIblMap,
    textures.groundContactAoMap,
    reflectionTarget.texture,
    BASIN_RADIUS,
    seawallLights,
  );
  waterMat.uniforms.uBoatScales.value = HARBOR_SAILBOATS.map((b) => b.scale);
  const waterGeo = new THREE.CircleGeometry(BASIN_RADIUS + 0.145, 96);
  waterGeo.rotateX(-Math.PI / 2);
  const waterMesh = new THREE.Mesh(waterGeo, waterMat);
  waterMesh.position.y = WATER_Y;
  dioramaRoot.add(waterMesh);

  // 3b. Real-Time Orthographic Key Shadow Map (1536 GPU / 768 SwiftShader, 16-bit RG depth + 3x3 PCF)
  const shadowRes = isSoftwareGl ? 768 : 1536;
  const shadowTarget = new THREE.WebGLRenderTarget(shadowRes, shadowRes, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    depthBuffer: true,
  });
  const shadowCamera = new THREE.OrthographicCamera(-14.5, 14.5, 14.5, -14.5, 2.0, 52.0);
  const shadowDepthMat = createShadowDepthMaterial();

  const shadowReceiverMats = [
    heroGlassMat,
    urbanGridGlassMat,
    heroLobbyGlassMat,
    terminalArchGlassMat,
    matteSlateMat,
    masonrySlateMat,
    brushedMetalMat,
    waterMat,
    landMat,
  ];
  for (const m of shadowReceiverMats) {
    m.uniforms.uShadowMap.value = shadowTarget.texture;
    m.uniforms.uShadowTexelSize.value.set(1 / shadowRes, 1 / shadowRes);
    m.uniforms.uHasShadowMap.value = 1.0;
  }

  // 4. Hero San Diego Landmarks
  const skylineGroup = new THREE.Group();
  dioramaRoot.add(skylineGroup);

  for (const landmark of HERO_LANDMARKS) {
    const towerGroup = buildCompositeHeroTower(
      landmark,
      heroGlassMat,
      brushedMetalMat,
      matteSlateMat,
      heroLobbyGlassMat,
    );
    skylineGroup.add(towerGroup);
  }

  // Convention Center & Sails Pavilion (satin brushed metal vaults + soft recessed arch bays)
  skylineGroup.add(
    buildConventionCenter(heroGlassMat, brushedMetalMat, matteSlateMat, terminalArchGlassMat),
  );

  // USS Midway & Navy Pier
  skylineGroup.add(buildMidwayPier(brushedMetalMat, matteSlateMat));

  // 5. Downtown Urban Grid: Architectural Differentiation (Glossy Curtain-Wall Glass vs Dead-Matte Masonry Slate,
  //    varied parapet lips, stepped mechanical penthouses, and mixed slate/metallic rooftop HVAC units)
  const unitBoxGeo = new THREE.BoxGeometry(1, 1, 1);
  const parapetGeo = createUnitParapetGeometry();

  const blockSpecs = cityBlocks.map((b, i) => ({
    block: b,
    arch: classifyMidRiseBlock(i, b),
  }));
  const glassCount = blockSpecs.filter((s) => s.arch.facadeMaterial === 'curtain-glass').length;
  const slateCount = blockSpecs.length - glassCount;
  const penthouseCount = blockSpecs.filter((s) => s.arch.hasMechanicalPenthouse).length;
  const slateHvacTotal = blockSpecs.reduce(
    (acc, s) => acc + (s.arch.hasMetallicHvac ? 1 : s.arch.hvacCount),
    0,
  );
  const metalHvacTotal = blockSpecs.reduce(
    (acc, s) => acc + (s.arch.hasMetallicHvac && s.arch.hvacCount === 2 ? 1 : s.arch.hasMetallicHvac ? 1 : 0),
    0,
  );

  const cityGlassInstanced = new THREE.InstancedMesh(
    unitBoxGeo,
    urbanGridGlassMat,
    Math.max(1, glassCount),
  );
  const citySlateInstanced = new THREE.InstancedMesh(
    unitBoxGeo,
    masonrySlateMat,
    Math.max(1, slateCount),
  );
  const parapetInstanced = new THREE.InstancedMesh(parapetGeo, matteSlateMat, cityBlocks.length);
  const penthouseInstanced = new THREE.InstancedMesh(
    unitBoxGeo,
    matteSlateMat,
    Math.max(1, penthouseCount),
  );
  const hvacSlateInstanced = new THREE.InstancedMesh(
    unitBoxGeo,
    matteSlateMat,
    Math.max(1, slateHvacTotal),
  );
  const hvacMetalInstanced = new THREE.InstancedMesh(
    unitBoxGeo,
    brushedMetalMat,
    Math.max(1, metalHvacTotal),
  );

  let glassIdx = 0;
  let slateIdx = 0;
  let pentIdx = 0;
  let hvacSlateIdx = 0;
  let hvacMetalIdx = 0;

  for (let i = 0; i < blockSpecs.length; i++) {
    const { block: b, arch } = blockSpecs[i];
    _scratchPos.set(b.x, 0.08 + b.height * 0.5, b.z);
    _scratchEuler.set(0, b.rotationY, 0);
    _scratchQuat.setFromEuler(_scratchEuler);
    _scratchScale.set(b.width, b.height, b.depth);
    _scratchMat4.compose(_scratchPos, _scratchQuat, _scratchScale);

    if (arch.facadeMaterial === 'curtain-glass') {
      cityGlassInstanced.setMatrixAt(glassIdx++, _scratchMat4);
    } else {
      citySlateInstanced.setMatrixAt(slateIdx++, _scratchMat4);
    }

    // Varied perimeter parapet lip around every mid-rise roof
    _scratchPos.set(b.x, 0.08 + b.height, b.z);
    _scratchScale.set(b.width, arch.parapetHeight, b.depth);
    _scratchMat4.compose(_scratchPos, _scratchQuat, _scratchScale);
    parapetInstanced.setMatrixAt(i, _scratchMat4);

    const cosR = Math.cos(b.rotationY);
    const sinR = Math.sin(b.rotationY);
    let roofBaseY = 0.08 + b.height;

    if (arch.hasMechanicalPenthouse) {
      _scratchPos.set(b.x, roofBaseY + arch.penthouseHeight * 0.5, b.z);
      _scratchScale.set(b.width * 0.52, arch.penthouseHeight, b.depth * 0.52);
      _scratchMat4.compose(_scratchPos, _scratchQuat, _scratchScale);
      penthouseInstanced.setMatrixAt(pentIdx++, _scratchMat4);
      roofBaseY += arch.penthouseHeight;
    }

    // Varied rooftop HVAC / cooling units (slate or satin metal enclosures)
    const offAx = b.width * (arch.hasMechanicalPenthouse ? 0.11 : 0.16);
    const offAz = b.depth * (arch.hasMechanicalPenthouse ? 0.10 : 0.14);
    const hvacAH = arch.parapetHeight * 1.42;
    _scratchPos.set(
      b.x + offAx * cosR + offAz * sinR,
      roofBaseY + hvacAH * 0.5,
      b.z - offAx * sinR + offAz * cosR,
    );
    _scratchScale.set(b.width * 0.24, hvacAH, b.depth * 0.24);
    _scratchMat4.compose(_scratchPos, _scratchQuat, _scratchScale);
    if (arch.hasMetallicHvac) {
      hvacMetalInstanced.setMatrixAt(hvacMetalIdx++, _scratchMat4);
    } else {
      hvacSlateInstanced.setMatrixAt(hvacSlateIdx++, _scratchMat4);
    }

    if (arch.hvacCount === 2) {
      const hvacBH = arch.parapetHeight * 0.92;
      _scratchPos.set(
        b.x - offAx * cosR - offAz * sinR,
        roofBaseY + hvacBH * 0.5,
        b.z + offAx * sinR - offAz * cosR,
      );
      _scratchScale.set(b.width * 0.17, hvacBH, b.depth * 0.19);
      _scratchMat4.compose(_scratchPos, _scratchQuat, _scratchScale);
      hvacSlateInstanced.setMatrixAt(hvacSlateIdx++, _scratchMat4);
    }
  }

  for (const inst of [
    cityGlassInstanced,
    citySlateInstanced,
    parapetInstanced,
    penthouseInstanced,
    hvacSlateInstanced,
    hvacMetalInstanced,
  ]) {
    inst.instanceMatrix.needsUpdate = true;
    inst.computeBoundingSphere();
    skylineGroup.add(inst);
  }

  // 6. Embarcadero Waterfront Promenade Streetlights (InstancedMesh)
  const poleGeo = new THREE.CylinderGeometry(0.018, 0.024, 0.28, 6);
  const poleMat = new THREE.MeshBasicMaterial({ color: 0x464a54 });
  const poleInstanced = new THREE.InstancedMesh(poleGeo, poleMat, seawallLights.length);

  const bulbGeo = new THREE.SphereGeometry(0.048, 8, 8);
  const bulbMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(2.21, 2.24, 2.31),
    toneMapped: false,
  });
  const bulbInstanced = new THREE.InstancedMesh(bulbGeo, bulbMat, seawallLights.length);

  for (let i = 0; i < seawallLights.length; i++) {
    const l = seawallLights[i];
    _scratchQuat.identity();
    _scratchScale.set(1, 1, 1);
    _scratchPos.set(l.x, 0.22, l.z);
    _scratchMat4.compose(_scratchPos, _scratchQuat, _scratchScale);
    poleInstanced.setMatrixAt(i, _scratchMat4);

    _scratchPos.set(l.x, 0.37, l.z);
    _scratchMat4.compose(_scratchPos, _scratchQuat, _scratchScale);
    bulbInstanced.setMatrixAt(i, _scratchMat4);
  }
  poleInstanced.instanceMatrix.needsUpdate = true;
  bulbInstanced.instanceMatrix.needsUpdate = true;
  poleInstanced.computeBoundingSphere();
  bulbInstanced.computeBoundingSphere();
  dioramaRoot.add(poleInstanced);
  dioramaRoot.add(bulbInstanced);

  // 7. 3D Miniature Harbor Sailboats (sculpted dark hull + metal mast/boom + 3D cambered sails)
  // Forward bow points along +Z so rotation.y = Math.atan2(dirX, dirZ) aligns bow with travel vector
  const boatHullShape = new THREE.Shape();
  boatHullShape.moveTo(0, -0.22); // pointed bow (becomes +Z after rotateX(-PI/2))
  boatHullShape.quadraticCurveTo(0.072, -0.02, 0.056, 0.16);
  boatHullShape.lineTo(-0.056, 0.16);
  boatHullShape.quadraticCurveTo(-0.072, -0.02, 0, -0.22);
  boatHullShape.closePath();

  const boatHullGeo = new THREE.ExtrudeGeometry(boatHullShape, {
    depth: 0.046,
    bevelEnabled: true,
    bevelSegments: 2,
    steps: 1,
    bevelSize: 0.008,
    bevelThickness: 0.010,
  });
  boatHullGeo.rotateX(-Math.PI / 2);

  const cabinGeo = new THREE.BoxGeometry(0.052, 0.026, 0.12);
  const mastGeo = new THREE.CylinderGeometry(0.005, 0.009, 0.56, 6);
  const boomGeo = new THREE.CylinderGeometry(0.004, 0.004, 0.22, 5);
  boomGeo.rotateX(Math.PI / 2);

  const mainSailGeo = createCamberedSailGeometry(0.50, 0.20, 0.024, false);
  const jibSailGeo = createCamberedSailGeometry(0.41, 0.135, 0.016, true);
  const sailMat = createSailFabricMaterial();

  const boatGroups: THREE.Group[] = HARBOR_SAILBOATS.map((spec) => {
    const boatGroup = new THREE.Group();
    const hull = new THREE.Mesh(boatHullGeo, matteSlateMat);
    hull.position.y = 0.0;

    const cabin = new THREE.Mesh(cabinGeo, brushedMetalMat);
    cabin.position.set(0, 0.050, -0.01);

    const mast = new THREE.Mesh(mastGeo, brushedMetalMat);
    mast.position.set(0, 0.304, 0.035);

    const boom = new THREE.Mesh(boomGeo, brushedMetalMat);
    boom.position.set(0, 0.090, -0.075);

    const mainSail = new THREE.Mesh(mainSailGeo, sailMat);
    mainSail.position.set(0, 0.095, 0.03);

    const jibSail = new THREE.Mesh(jibSailGeo, sailMat);
    jibSail.position.set(0, 0.070, 0.048);

    boatGroup.add(hull, cabin, mast, boom, mainSail, jibSail);
    boatGroup.scale.setScalar(spec.scale);
    boatGroup.position.set(spec.baseX, 0.013, spec.baseZ);
    boatGroup.rotation.y = spec.heading;
    dioramaRoot.add(boatGroup);
    return boatGroup;
  });

  // Pre-warm shaders on GPU devices
  if (!isSoftwareGl) {
    renderer.compile(scene, camera);
  }

  // Interactive softbox preset & camera transition state initialized from initialPreset
  const initElevAngle = initialPreset.softboxKeyElevation * (Math.PI * 0.42);
  const initHoriz = Math.cos(initElevAngle);
  const targetKeyDir = new THREE.Vector3(
    Math.cos(initialPreset.softboxKeyAzimuth) * initHoriz,
    Math.sin(initElevAngle),
    Math.sin(initialPreset.softboxKeyAzimuth) * initHoriz,
  ).normalize();
  const currentKeyDir = targetKeyDir.clone();
  let targetKeyInt = initialPreset.keyIntensity;
  let targetRimInt = initialPreset.rimIntensity;
  let targetFillInt = initialPreset.fillIntensity;
  let targetReflectivity = initialPreset.waterReflectivity;
  let targetEdgeGlow = initialPreset.glassEdgeGlow;

  let currentKeyInt = targetKeyInt;
  let currentRimInt = targetRimInt;
  let currentFillInt = targetFillInt;
  let currentReflectivity = targetReflectivity;
  let currentEdgeGlow = targetEdgeGlow;

  let pointerOffsetX = 0;
  let pointerOffsetY = 0;
  let canvasPointerDown = false;
  let cameraResetActive = false;

  // RK4 damped springs for tactile diorama suspension & specular parallax
  const pressSpring = createSpringState(0);
  const tiltXSpring = createSpringState(0);
  const tiltYSpring = createSpringState(0);
  const velXSpring = createSpringState(0);
  const velYSpring = createSpringState(0);

  const updatePointer = (e: PointerEvent) => {
    const rect = renderer.domElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = ((e.clientY - rect.top) / rect.height) * 2 - 1;
    const dx = nx - pointerOffsetX;
    const dy = ny - pointerOffsetY;
    pointerOffsetX = nx;
    pointerOffsetY = ny;
    if (!reducedMotion) {
      tiltXSpring.target = nx;
      tiltYSpring.target = ny;
      velXSpring.target = Math.max(-1, Math.min(1, dx * 22));
      velYSpring.target = Math.max(-1, Math.min(1, dy * 22));
    }
    if (reducedMotion) requestRender();
  };

  const resetPointer = () => {
    pointerOffsetX = 0;
    pointerOffsetY = 0;
    canvasPointerDown = false;
    tiltXSpring.target = 0;
    tiltYSpring.target = 0;
    velXSpring.target = 0;
    velYSpring.target = 0;
    if (reducedMotion) requestRender();
  };

  const onCanvasPointerDown = () => {
    canvasPointerDown = true;
    if (reducedMotion) requestRender();
  };

  const onCanvasPointerUp = () => {
    canvasPointerDown = false;
    if (reducedMotion) requestRender();
  };

  const onDoubleClick = () => {
    cameraResetActive = true;
    if (reducedMotion) {
      camera.position.copy(DEFAULT_CAMERA_POS);
      controls.target.copy(DEFAULT_CAMERA_TARGET);
      controls.update();
      requestRender();
    }
  };

  const onControlsStart = () => {
    cameraResetActive = false;
  };

  renderer.domElement.addEventListener('pointermove', updatePointer);
  renderer.domElement.addEventListener('pointerleave', resetPointer);
  renderer.domElement.addEventListener('pointerdown', onCanvasPointerDown);
  window.addEventListener('pointerup', onCanvasPointerUp);
  renderer.domElement.addEventListener('dblclick', onDoubleClick);
  controls.addEventListener('start', onControlsStart);

  // Resize observer: cleanly reallocate HDR, Planar Reflection & Dual-Kawase FBO attachments with clamped DPR
  const handleResize = () => {
    const w = Math.max(container.clientWidth, 320);
    const h = Math.max(container.clientHeight, 320);
    const dpr = isSoftwareGl ? 1 : Math.min(window.devicePixelRatio || 1, 2);
    camera.fov = computeFovForAspect(w / h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h);
    reflectionTarget.setSize(
      Math.max(256, Math.floor(w * dpr * 0.5)),
      Math.max(256, Math.floor(h * dpr * 0.5)),
    );
    postProcessor.setSize(w, h, renderer.getPixelRatio());
    renderFrame(0.016);
  };
  const resizeObserver = new ResizeObserver(handleResize);
  resizeObserver.observe(container);

  let frameId = 0;
  let disposed = false;
  let renderBusy = false;
  let renderDirty = false;
  let readyNotified = false;
  let lastTimestamp = performance.now();
  let elapsedSeconds = 0;

  function applyShaderUniforms(dt: number, elapsed: number) {
    const k = Math.min(1, 1 - Math.exp(-5.0 * dt));

    // Couple diorama suspension spring with canvas drag/press AND active UI switch depression
    const combinedTargetPress = Math.max(
      canvasPointerDown ? 1.0 : 0.0,
      globalHardwareTelemetry.activeUiPressDepth * 0.72,
    );
    pressSpring.target = combinedTargetPress;

    const pressCfg = reducedMotion
      ? SPRING_PRESETS.tactileButtonReduced
      : SPRING_PRESETS.chassisPanel;
    stepSpringRK4(pressSpring, dt, pressCfg);

    if (reducedMotion) {
      tiltXSpring.value = 0;
      tiltYSpring.value = 0;
      velXSpring.value = 0;
      velYSpring.value = 0;
    } else {
      tiltXSpring.target =
        pointerOffsetX * 0.65 +
        globalHardwareTelemetry.pointerNx * 0.35 +
        globalHardwareTelemetry.deviceTiltX * 0.4;
      tiltYSpring.target =
        pointerOffsetY * 0.65 +
        globalHardwareTelemetry.pointerNy * 0.35 +
        globalHardwareTelemetry.deviceTiltY * 0.4;
      stepSpringRK4(tiltXSpring, dt, SPRING_PRESETS.specularParallax);
      stepSpringRK4(tiltYSpring, dt, SPRING_PRESETS.specularParallax);

      velXSpring.target = Math.max(
        -1,
        Math.min(1, velXSpring.target * 0.85 + globalHardwareTelemetry.pointerVelX * 0.55),
      );
      velYSpring.target = Math.max(
        -1,
        Math.min(1, velYSpring.target * 0.85 + globalHardwareTelemetry.pointerVelY * 0.55),
      );
      stepSpringRK4(velXSpring, dt, SPRING_PRESETS.specularParallax);
      stepSpringRK4(velYSpring, dt, SPRING_PRESETS.specularParallax);
    }

    const activePress = Math.max(0, Math.min(1.15, pressSpring.value));
    dioramaRoot.position.y = -activePress * 0.14;
    if (!reducedMotion) {
      dioramaRoot.rotation.z = -tiltXSpring.value * 0.012;
      dioramaRoot.rotation.x = tiltYSpring.value * 0.010;
    }

    currentKeyDir.lerp(targetKeyDir, k);
    _scratchDir.copy(currentKeyDir);
    _scratchDir.x += tiltXSpring.value * 0.22;
    _scratchDir.y = Math.max(0.25, _scratchDir.y - tiltYSpring.value * 0.12 - activePress * 0.08);
    _scratchDir.normalize();

    currentKeyInt += (targetKeyInt - currentKeyInt) * k;
    currentRimInt += (targetRimInt - currentRimInt) * k;
    currentFillInt += (targetFillInt - currentFillInt) * k;
    currentReflectivity += (targetReflectivity - currentReflectivity) * k;
    currentEdgeGlow += (targetEdgeGlow - currentEdgeGlow) * k;

    const shaderMats = [
      heroGlassMat,
      urbanGridGlassMat,
      heroLobbyGlassMat,
      terminalArchGlassMat,
      matteSlateMat,
      masonrySlateMat,
      brushedMetalMat,
      plinthMetalMat,
      waterMat,
      landMat,
      cycloramaMat,
      sailMat,
    ];
    for (const m of shaderMats) {
      m.uniforms.uSoftboxKeyDir.value.copy(_scratchDir);
      m.uniforms.uKeyIntensity.value = currentKeyInt;
      m.uniforms.uRimIntensity.value = currentRimInt;
      m.uniforms.uFillIntensity.value = currentFillInt;
      m.uniforms.uPressDepth.value = activePress;
      m.uniforms.uTiltVec.value.set(tiltXSpring.value, tiltYSpring.value);
      m.uniforms.uVelocity.value.set(velXSpring.value, velYSpring.value);
    }

    heroGlassMat.uniforms.uEdgeLuminance.value = currentEdgeGlow;
    heroLobbyGlassMat.uniforms.uEdgeLuminance.value = currentEdgeGlow * 0.85;
    waterMat.uniforms.uReflectivity.value = currentReflectivity;
    waterMat.uniforms.uTime.value = elapsed;

    // Update 3D sailboat glides, headings & Kelvin wake uniform vectors
    const boatUniforms = waterMat.uniforms.uBoats.value as THREE.Vector4[];
    for (let i = 0; i < HARBOR_SAILBOATS.length; i++) {
      const spec = HARBOR_SAILBOATS[i];
      const angle = elapsed * spec.speed + spec.phase;
      const bx = spec.baseX + Math.cos(angle) * spec.orbitRadius;
      const bz = spec.baseZ + Math.sin(angle * 0.8) * (spec.orbitRadius * 0.55);
      const vx = -Math.sin(angle) * spec.orbitRadius * spec.speed;
      const vz = Math.cos(angle * 0.8) * (spec.orbitRadius * 0.55) * (spec.speed * 0.8);
      const vLen = Math.hypot(vx, vz) || 1;
      const dirX = vx / vLen;
      const dirZ = vz / vLen;

      boatGroups[i].position.set(
        bx,
        0.013 + Math.sin(elapsed * 1.8 + spec.phase) * 0.003,
        bz,
      );
      boatGroups[i].rotation.y = Math.atan2(dirX, dirZ);
      boatGroups[i].rotation.z = Math.sin(elapsed * 1.4 + spec.phase) * 0.035;
      boatUniforms[i].set(bx, bz, dirX, dirZ);
    }

    if (cameraResetActive) {
      camera.position.lerp(DEFAULT_CAMERA_POS, k * 1.4);
      controls.target.lerp(DEFAULT_CAMERA_TARGET, k * 1.4);
      if (camera.position.distanceTo(DEFAULT_CAMERA_POS) < 0.04) {
        camera.position.copy(DEFAULT_CAMERA_POS);
        controls.target.copy(DEFAULT_CAMERA_TARGET);
        cameraResetActive = false;
      }
    }

    controls.update();
  }

  function renderShadowPass() {
    shadowCamera.position.copy(_scratchDir).multiplyScalar(24.0);
    shadowCamera.position.y = Math.max(8.0, shadowCamera.position.y);
    shadowCamera.up.set(0, 1, 0);
    shadowCamera.lookAt(0, 1.5, 0);
    shadowCamera.updateMatrixWorld();

    _shadowMatrix.multiplyMatrices(shadowCamera.projectionMatrix, shadowCamera.matrixWorldInverse);
    for (const m of shadowReceiverMats) {
      m.uniforms.uShadowMatrix.value.copy(_shadowMatrix);
    }

    const prevOverride = scene.overrideMaterial;
    const prevBg = scene.background;
    const prevFog = scene.fog;
    const prevClearAlpha = renderer.getClearAlpha();

    waterMesh.visible = false;
    landmass.visible = false;
    seawallCoping.visible = false;
    perimeterCurb.visible = false;
    plinth.visible = false;
    rimRing.visible = false;
    cycloramaMesh.visible = false;

    scene.background = null;
    scene.fog = null;
    scene.overrideMaterial = shadowDepthMat;

    renderer.setRenderTarget(shadowTarget);
    renderer.setClearColor(_shadowClearColor, 1);
    renderer.clear();
    renderer.render(scene, shadowCamera);
    renderer.setRenderTarget(null);
    renderer.setClearColor(0x000000, prevClearAlpha);

    scene.overrideMaterial = prevOverride;
    scene.background = prevBg;
    scene.fog = prevFog;

    waterMesh.visible = true;
    landmass.visible = true;
    seawallCoping.visible = true;
    perimeterCurb.visible = true;
    plinth.visible = true;
    rimRing.visible = true;
    cycloramaMesh.visible = true;
  }

  function renderReflectionPass() {
    const worldWaterY = WATER_Y + dioramaRoot.position.y;
    reflectCamera.fov = camera.fov;
    reflectCamera.aspect = camera.aspect;
    reflectCamera.near = camera.near;
    reflectCamera.far = camera.far;
    reflectCamera.updateProjectionMatrix();
    reflectCamera.position.set(
      camera.position.x,
      2 * worldWaterY - camera.position.y,
      camera.position.z,
    );
    reflectCamera.up.set(0, 1, 0);
    reflectCamera.lookAt(
      controls.target.x,
      2 * worldWaterY - controls.target.y,
      controls.target.z,
    );
    reflectCamera.updateMatrixWorld();

    waterMat.uniforms.uReflectMatrix.value.multiplyMatrices(
      reflectCamera.projectionMatrix,
      reflectCamera.matrixWorldInverse,
    );

    heroGlassMat.uniforms.uIsReflectPass.value = 1.0;
    urbanGridGlassMat.uniforms.uIsReflectPass.value = 1.0;
    heroLobbyGlassMat.uniforms.uIsReflectPass.value = 1.0;
    terminalArchGlassMat.uniforms.uIsReflectPass.value = 1.0;
    matteSlateMat.uniforms.uIsReflectPass.value = 1.0;
    masonrySlateMat.uniforms.uIsReflectPass.value = 1.0;
    brushedMetalMat.uniforms.uIsReflectPass.value = 1.0;

    waterMesh.visible = false;
    landmass.visible = false;
    seawallCoping.visible = false;
    perimeterCurb.visible = false;
    plinth.visible = false;
    rimRing.visible = false;
    cycloramaMesh.visible = false;
    poleInstanced.visible = false;

    const prevBg = scene.background;
    const prevFog = scene.fog;
    scene.background = _blackBg;
    scene.fog = null;

    renderer.setRenderTarget(reflectionTarget);
    renderer.clear();
    renderer.render(scene, reflectCamera);
    renderer.setRenderTarget(null);

    scene.background = prevBg;
    scene.fog = prevFog;

    waterMesh.visible = true;
    landmass.visible = true;
    seawallCoping.visible = true;
    perimeterCurb.visible = true;
    plinth.visible = true;
    rimRing.visible = true;
    cycloramaMesh.visible = true;
    poleInstanced.visible = true;

    heroGlassMat.uniforms.uIsReflectPass.value = 0.0;
    urbanGridGlassMat.uniforms.uIsReflectPass.value = 0.0;
    heroLobbyGlassMat.uniforms.uIsReflectPass.value = 0.0;
    terminalArchGlassMat.uniforms.uIsReflectPass.value = 0.0;
    matteSlateMat.uniforms.uIsReflectPass.value = 0.0;
    masonrySlateMat.uniforms.uIsReflectPass.value = 0.0;
    brushedMetalMat.uniforms.uIsReflectPass.value = 0.0;
  }

  function renderFrame(dt: number) {
    _savedCamPos.copy(camera.position);
    _savedCamQuat.copy(camera.quaternion);

    // Subtle RK4 damped-spring cursor parallax camera orbit (±2 deg) without fighting OrbitControls
    if (!reducedMotion && !canvasPointerDown) {
      const yaw = -tiltXSpring.value * 0.036;
      const pitch = -tiltYSpring.value * 0.018;
      _parallaxOffset.copy(camera.position).sub(controls.target);
      _parallaxOffset.applyAxisAngle(_parallaxUpAxis, yaw);
      _parallaxRight.crossVectors(_parallaxUpAxis, _parallaxOffset).normalize();
      _parallaxOffset.applyAxisAngle(_parallaxRight, pitch);
      camera.position.copy(controls.target).add(_parallaxOffset);
      camera.lookAt(controls.target);
      camera.updateMatrixWorld();
    }

    try {
      renderShadowPass();
      renderReflectionPass();
      const focusDistance = camera.position.distanceTo(HERO_FOCAL_POINT);
      postProcessor.render(
        scene,
        camera,
        dt,
        focusDistance,
        [],
        reducedMotion ? 0 : elapsedSeconds,
      );
    } finally {
      camera.position.copy(_savedCamPos);
      camera.quaternion.copy(_savedCamQuat);
      camera.updateMatrixWorld();
    }
  }

  function afterDraw(cb: () => void) {
    if (disposed) return;
    const gl = renderer.getContext() as WebGL2RenderingContext;
    if (typeof gl.fenceSync === 'function') {
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (disposed) {
          if (sync) gl.deleteSync(sync);
          return;
        }
        const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    } else {
      frameId = window.setTimeout(cb, 80);
    }
  }

  function requestRender() {
    if (disposed) return;
    if (renderBusy) {
      renderDirty = true;
      return;
    }
    renderBusy = true;
    frameId = requestAnimationFrame((now) => {
      if (disposed) return;
      renderDirty = false;
      const dt = Math.min(Math.max((now - lastTimestamp) / 1000, 0.001), 0.05);
      lastTimestamp = now;
      elapsedSeconds += dt;
      applyShaderUniforms(dt, elapsedSeconds);
      renderFrame(dt);
      if (!readyNotified) {
        readyNotified = true;
        callbacks.onReady?.();
      }
      const done = () => {
        renderBusy = false;
        if (renderDirty) requestRender();
      };
      if (isSoftwareGl) afterDraw(done);
      else done();
    });
  }

  function tick() {
    if (disposed) return;
    const now = performance.now();
    const dt = Math.min(Math.max((now - lastTimestamp) / 1000, 0.001), 0.05);
    lastTimestamp = now;
    elapsedSeconds += dt;
    applyShaderUniforms(dt, elapsedSeconds);
    renderFrame(dt);
    if (!readyNotified) {
      readyNotified = true;
      callbacks.onReady?.();
    }
    if (isSoftwareGl) {
      afterDraw(tick);
    } else {
      frameId = requestAnimationFrame(tick);
    }
  }

  if (reducedMotion) {
    controls.addEventListener('change', requestRender);
    requestRender();
  } else {
    frameId = requestAnimationFrame(tick);
  }

  const setLightingPreset = (presetId: LightingPresetId) => {
    const preset = LIGHTING_PRESETS[presetId];
    if (!preset) return;
    lightingRig.applyPreset(preset);

    const elevAngle = preset.softboxKeyElevation * (Math.PI * 0.42);
    const y = Math.sin(elevAngle);
    const horiz = Math.cos(elevAngle);
    targetKeyDir.set(
      Math.cos(preset.softboxKeyAzimuth) * horiz,
      y,
      Math.sin(preset.softboxKeyAzimuth) * horiz,
    ).normalize();

    targetKeyInt = preset.keyIntensity;
    targetRimInt = preset.rimIntensity;
    targetFillInt = preset.fillIntensity;
    targetReflectivity = preset.waterReflectivity;
    targetEdgeGlow = preset.glassEdgeGlow;

    const bloomByPreset: Record<LightingPresetId, number> = {
      'overhead-studio': 0.19,
      'grazing-rim': 0.21,
      'harbor-dusk': 0.24,
    };
    postProcessor.setBloomStrength(bloomByPreset[presetId] ?? 0.19);

    if (reducedMotion) requestRender();
  };

  const resetCamera = () => {
    cameraResetActive = true;
    if (reducedMotion) {
      camera.position.copy(DEFAULT_CAMERA_POS);
      controls.target.copy(DEFAULT_CAMERA_TARGET);
      controls.update();
      requestRender();
    }
  };

  const dispose = () => {
    disposed = true;
    cancelAnimationFrame(frameId);
    window.clearTimeout(frameId);
    resizeObserver.disconnect();
    renderer.domElement.removeEventListener('pointermove', updatePointer);
    renderer.domElement.removeEventListener('pointerleave', resetPointer);
    renderer.domElement.removeEventListener('pointerdown', onCanvasPointerDown);
    window.removeEventListener('pointerup', onCanvasPointerUp);
    renderer.domElement.removeEventListener('dblclick', onDoubleClick);
    renderer.domElement.removeEventListener('webglcontextlost', onLost);
    renderer.domElement.removeEventListener('webglcontextrestored', onRestored);
    controls.removeEventListener('start', onControlsStart);
    controls.removeEventListener('change', requestRender);
    controls.dispose();

    scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.LineSegments) {
        obj.geometry.dispose();
        if (Array.isArray(obj.material)) {
          obj.material.forEach((mat) => mat.dispose());
        } else {
          obj.material.dispose();
        }
      }
    });

    shadowTarget.dispose();
    shadowDepthMat.dispose();
    reflectionTarget.dispose();
    textures.dispose();
    lightingRig.dispose();
    postProcessor.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode === container) {
      container.removeChild(renderer.domElement);
    }
  };

  return {
    setLightingPreset,
    resetCamera,
    dispose,
    isSoftwareRenderer: isSoftwareGl,
  };
}
