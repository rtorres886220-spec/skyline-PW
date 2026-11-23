import * as THREE from 'three';

export interface TowerReflector {
  x: number;
  z: number;
  width: number;
  height: number;
}

/**
 * Shared WebGL2 GLSL3 Procedural Studio IBL + Physical Optics Library:
 * - Samples the startup-baked mipmapped equirectangular studio IBL map (`textureLod`)
 *   according to surface roughness (`roughness * 6.5`)
 * - Computes screen-space MikkTSpace-style directional machined chamfer bevel normals (1-2px)
 * - Evaluates Schlick Fresnel, wrap diffuse, urban canyon AO, and real-time soft
 *   raymarched architectural heightfield cast shadows
 */
const GLSL3_OPTICS_COMMON = /* glsl */ `
const float PI = 3.14159265;
const float INV_TWO_PI = 0.159154943;

float fresnelSchlick(float cosTheta, float F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float wrapDiffuse(vec3 N, vec3 L, float wrap) {
  return clamp((dot(N, L) + wrap) / (1.0 + wrap), 0.0, 1.0);
}

vec2 dirToEquirectUV(vec3 dir, float azimuthShift) {
  vec3 d = normalize(dir);
  float u = fract(atan(-d.x, -d.z) * INV_TWO_PI + azimuthShift);
  float v = 1.0 - acos(clamp(d.y, -1.0, 1.0)) / PI;
  return vec2(u, clamp(v, 0.002, 0.998));
}

vec3 sampleStudioIBL(
  sampler2D iblMap,
  vec3 dir,
  float roughness,
  vec3 keyDir,
  float keyInt,
  float rimInt,
  float fillInt
) {
  float baseAz = atan(-(-0.44), -(0.48));
  float targetAz = atan(-keyDir.x, -keyDir.z);
  float azShift = (targetAz - baseAz) * 0.10;

  vec2 uv = dirToEquirectUV(dir, azShift);
  float lod = clamp(roughness * 6.5, 0.0, 6.5);
  vec3 enc = textureLod(iblMap, uv, lod).rgb;
  vec3 hdr = pow(max(enc, vec3(0.0001)), vec3(1.6)) * 2.8;

  // Preserve 3:1 key-to-fill studio contrast across key-facing vs shadow hemispheres
  float keyHemisphere = smoothstep(-0.25, 0.65, dot(normalize(dir), normalize(keyDir)));
  float rimWeight = smoothstep(0.45, 0.95, abs(dir.x)) * (1.0 - keyHemisphere * 0.4);
  float presetScale = mix(0.28 + 0.42 * fillInt, keyInt, keyHemisphere);
  presetScale *= mix(1.0, 0.75 + 0.35 * rimInt, rimWeight);
  return hdr * presetScale;
}

/**
 * Samples baked urban AO:
 * .x = tight crevice contact AO
 * .y = deep urban street canyon AO
 */
vec2 sampleUrbanSpatialAo(sampler2D spatialMap, vec2 worldXZ, float basinRadius) {
  vec2 uv = clamp(worldXZ / (basinRadius * 2.1) + 0.5, 0.001, 0.999);
  return texture(spatialMap, uv).rg;
}

/**
 * Soft raymarched directional cast shadows using the baked 512x512 skyline heightfield
 * stored in spatialMap.b (0..10 world units). Produces soft PCF-like penumbra shadows
 * from tall towers onto neighboring mid-rise roofs, setbacks, streets, and water.
 */
float sampleDirectionalHeightShadow(
  sampler2D spatialMap,
  vec3 worldPos,
  vec3 surfaceN,
  vec3 keyDir,
  float basinRadius
) {
  float extent = basinRadius * 1.05;
  vec3 dir = normalize(keyDir);
  float slopeY = max(dir.y, 0.24);
  vec2 stepHoriz = dir.xz / slopeY;

  // Offset origin slightly outside vertical walls along outward normal to eliminate self-shadow acne
  float isVertical = 1.0 - smoothstep(0.60, 0.85, abs(surfaceN.y));
  vec2 horizN = length(surfaceN.xz) > 0.001 ? normalize(surfaceN.xz) : vec2(0.0);
  vec2 originXZ = worldPos.xz + horizN * (0.18 * isVertical);

  float shadow = 1.0;
  float selfBias = 0.22;

  for (int i = 1; i <= 12; i++) {
    float fi = float(i);
    float dY = (fi * fi) * 0.055;
    vec2 sampleXZ = originXZ + stepHoriz * dY;
    vec2 uv = sampleXZ / (extent * 2.0) + 0.5;
    if (uv.x <= 0.002 || uv.x >= 0.998 || uv.y <= 0.002 || uv.y >= 0.998) {
      break;
    }
    float occH = textureLod(spatialMap, uv, 0.0).b * 10.0;
    float rayY = worldPos.y + dY + selfBias;
    if (occH > rayY) {
      float penumbra = clamp((occH - rayY) / (0.12 + dY * 0.11), 0.0, 1.0);
      shadow = min(shadow, 1.0 - penumbra * 0.92);
    }
  }
  return clamp(shadow, 0.08, 1.0);
}

/**
 * Computes subtle 1-2px machined chamfer bevel normal perturbation along UV face edges
 * using screen-space derivatives (fwidth) and a screen-space tangent frame.
 * Returns vec4(perturbedNormal.xyz, chamferEdgeMask).
 */
vec4 computeMachinedChamfer(
  vec3 N,
  vec3 worldPos,
  vec2 uv,
  float bevelStrength,
  float allowSideEdges
) {
  vec3 dp1 = dFdx(worldPos);
  vec3 dp2 = dFdy(worldPos);
  vec2 duv1 = dFdx(uv);
  vec2 duv2 = dFdy(uv);

  vec3 dp2perp = cross(dp2, N);
  vec3 dp1perp = cross(N, dp1);
  vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
  vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
  float maxLenSq = max(dot(T, T), dot(B, B));
  if (maxLenSq < 1e-8) {
    return vec4(N, 0.0);
  }
  float invMax = inversesqrt(maxLenSq);
  T *= invMax;
  B *= invMax;

  vec2 fwUv = clamp(fwidth(uv), vec2(0.0004), vec2(0.08));
  float edgeY = min(uv.y, 1.0 - uv.y);
  float maskY = 1.0 - smoothstep(0.0, max(fwUv.y * 1.9, 0.010), edgeY);
  float signY = uv.y > 0.5 ? 1.0 : -1.0;

  float edgeX = min(uv.x, 1.0 - uv.x);
  float uvWrapGuard = 1.0 - smoothstep(0.08, 0.25, fwUv.x);
  float maskX = (1.0 - smoothstep(0.0, max(fwUv.x * 1.9, 0.010), edgeX)) * allowSideEdges * uvWrapGuard;
  float signX = uv.x > 0.5 ? 1.0 : -1.0;

  vec3 tilt = T * (signX * maskX) + B * (signY * maskY);
  float chamferMask = clamp(max(maskX, maskY), 0.0, 1.0);
  vec3 chamferN = normalize(N + tilt * bevelStrength * chamferMask);
  return vec4(chamferN, chamferMask);
}

/**
 * Whisper-subtle dark atmospheric depth/height haze trapped strictly inside the lowest
 * urban street corridors between tall towers to separate foreground from background silhouettes.
 */
vec3 applyCanyonAtmosphere(vec3 color, vec3 worldPos, vec3 camPos, float canyonAo) {
  float lowCorridor = smoothstep(2.35, 0.10, worldPos.y);
  float deepCanyon = smoothstep(0.84, 0.30, canyonAo);
  float depthFactor = smoothstep(28.0, 52.0, length(camPos - worldPos));
  float mistAmount = lowCorridor * deepCanyon * depthFactor * 0.28;
  vec3 canyonMistColor = vec3(0.0085, 0.0092, 0.0114);
  return mix(color, canyonMistColor, clamp(mistAmount, 0.0, 0.28));
}

/**
 * Gently shifts eye/reflection vector using smoothed pointer parallax, velocity,
 * and tilt gimbal across brushed anisotropic metal and curved glass.
 */
vec3 applySpecularParallax(vec3 V, vec2 tiltVec, vec2 velVec, float strength) {
  vec3 worldUp = abs(V.y) > 0.94 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
  vec3 viewRight = normalize(cross(worldUp, V));
  vec3 viewUp = normalize(cross(V, viewRight));
  vec2 offset = (tiltVec * 0.14 + velVec * 0.085) * strength;
  return normalize(V + viewRight * offset.x + viewUp * offset.y);
}
`;

export function createHarborWaterMaterial(
  waterMicroRippleMap: THREE.Texture,
  studioIblMap: THREE.Texture,
  groundAoMap: THREE.Texture,
  planarReflectionMap: THREE.Texture,
  basinRadius: number,
  seawallLampXZ: Array<{ x: number; z: number }>,
): THREE.ShaderMaterial {
  // Vector4(x, z, dirX, dirZ) for instantaneous sailboat position & heading
  const boatStates = Array.from({ length: 6 }, () => new THREE.Vector4(0, 100, 1, 0));
  const paddedLamps: THREE.Vector2[] = [];
  for (let i = 0; i < 32; i++) {
    const l = seawallLampXZ[i];
    paddedLamps.push(l ? new THREE.Vector2(l.x, l.z) : new THREE.Vector2(999, 999));
  }

  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    transparent: true,
    depthWrite: true,
    uniforms: {
      uTime: { value: 0 },
      uWaterRippleMap: { value: waterMicroRippleMap },
      uStudioIblMap: { value: studioIblMap },
      uGroundAoMap: { value: groundAoMap },
      uReflectMap: { value: planarReflectionMap },
      uReflectMatrix: { value: new THREE.Matrix4() },
      uSoftboxKeyDir: { value: new THREE.Vector3(-0.44, 0.68, 0.58) },
      uKeyIntensity: { value: 1.26 },
      uRimIntensity: { value: 0.92 },
      uFillIntensity: { value: 0.40 },
      uReflectivity: { value: 0.92 },
      uBasinRadius: { value: basinRadius },
      uPressDepth: { value: 0.0 },
      uTiltVec: { value: new THREE.Vector2(0, 0) },
      uVelocity: { value: new THREE.Vector2(0, 0) },
      uBoats: { value: boatStates },
      uLamps: { value: paddedLamps },
      uLampCount: { value: Math.min(32, seawallLampXZ.length) },
    },
    vertexShader: /* glsl */ `
      uniform mat4 uReflectMatrix;

      out vec3 vWorldPos;
      out vec2 vUv;
      out vec4 vReflClipPos;

      void main() {
        vUv = uv;
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPos = worldPos.xyz;
        vReflClipPos = uReflectMatrix * worldPos;
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform sampler2D uWaterRippleMap;
      uniform sampler2D uStudioIblMap;
      uniform sampler2D uGroundAoMap;
      uniform sampler2D uReflectMap;
      uniform vec3 uSoftboxKeyDir;
      uniform float uKeyIntensity;
      uniform float uRimIntensity;
      uniform float uFillIntensity;
      uniform float uReflectivity;
      uniform float uBasinRadius;
      uniform float uPressDepth;
      uniform vec2 uTiltVec;
      uniform vec2 uVelocity;
      uniform vec4 uBoats[6];
      uniform vec2 uLamps[32];
      uniform int uLampCount;

      in vec3 vWorldPos;
      in vec2 vUv;
      in vec4 vReflClipPos;

      out vec4 fragColor;

      ${GLSL3_OPTICS_COMMON}

      float seawallZ(float x) {
        return 2.6 + 2.35 * cos(x * 0.24 + 0.35) - 0.14 * x;
      }

      void main() {
        vec2 p = vWorldPos.xz;
        float distFromCenter = length(p);
        if (distFromCenter > uBasinRadius - 0.04) {
          discard;
        }

        float swZ = seawallZ(p.x);
        float shoreDist = p.y - swZ;
        if (shoreDist < -0.18) {
          discard;
        }

        // Crisp high-frequency capillary wind ripple normals (zero cloudy smudges or blobs)
        vec2 uvA = p * 0.34 + vec2(uTime * 0.022, uTime * 0.014);
        vec2 uvB = p * 0.68 - vec2(uTime * 0.028, -uTime * 0.019);
        vec2 nA = texture(uWaterRippleMap, uvA).xy * 2.0 - 1.0;
        vec2 nB = texture(uWaterRippleMap, uvB).xy * 2.0 - 1.0;

        // Subtle cross-wind capillary wavelets
        float capX = cos(p.x * 3.8 + p.y * 1.4 + uTime * 1.3) * 0.008;
        float capZ = sin(p.y * 4.4 - p.x * 1.1 + uTime * 1.15) * 0.009;

        // Realistic V-shaped Kelvin wake wave normals & bow ripple etched cleanly behind each 3D sailboat
        vec2 wakeGrad = vec2(0.0);
        float wakeCrestMask = 0.0;
        for (int i = 0; i < 6; i++) {
          vec4 b = uBoats[i];
          vec2 dBoat = p - b.xy;
          if (dot(dBoat, dBoat) < 9.5) {
            vec2 fwd = normalize(b.zw);
            vec2 right = vec2(-fwd.y, fwd.x);
            float back = -dot(dBoat, fwd);
            float side = dot(dBoat, right);
            if (back > -0.10 && back < 2.95) {
              // Kelvin 19.47° half-angle V-wake arms (tan(19.47°) ≈ 0.354)
              float armOffset = 0.042 + max(0.0, back) * 0.354;
              float armDist = abs(abs(side) - armOffset);
              float env = exp(-max(0.0, back) * 0.72) * smoothstep(-0.08, 0.12, back);
              float vArm = exp(-pow(armDist / (0.032 + max(0.0, back) * 0.014), 2.0)) * env;
              // Crisp divergent feather crests along both V-arms
              float crestWave = cos(armDist * 58.0 - uTime * 3.4) * vArm;
              vec2 armNormalDir = normalize(right * (side >= 0.0 ? 1.0 : -1.0) - fwd * 0.354);
              wakeGrad += armNormalDir * crestWave * 0.28;
              wakeCrestMask += max(0.0, crestWave) * 0.85 + vArm * 0.45;
              // Subtle bow parting wave & transverse stern ripples inside the V-wedge
              float bowRipple = exp(-pow(back / 0.06, 2.0)) * exp(-pow(abs(side) / 0.075, 2.0));
              wakeGrad += (right * (side >= 0.0 ? 1.0 : -1.0) + fwd * 0.6) * bowRipple * 0.18;
              wakeCrestMask += bowRipple * 0.65;
              if (abs(side) < armOffset * 0.92 && back > 0.04) {
                float trans = sin(back * 26.0 - uTime * 3.2) * exp(-back * 1.15) * 0.07;
                wakeGrad += fwd * trans;
                wakeCrestMask += max(0.0, trans) * 1.4;
              }
            }
          }
        }

        vec3 N = normalize(vec3(
          nA.x * 0.024 + nB.x * 0.018 + capX + wakeGrad.x,
          1.0,
          nA.y * 0.024 + nB.y * 0.018 + capZ + wakeGrad.y
        ));

        vec3 V = normalize(cameraPosition - vWorldPos);
        vec3 V_spec = applySpecularParallax(V, uTiltVec, uVelocity, 0.85);
        vec3 keyL = normalize(uSoftboxKeyDir);
        float NdotV = clamp(dot(N, V_spec), 0.0, 1.0);

        // Schlick Fresnel for polished dark smoked resin/water (F0 = 0.032)
        float fresnel = fresnelSchlick(NdotV, 0.032);

        float waterShadow = sampleDirectionalHeightShadow(
          uGroundAoMap,
          vWorldPos,
          vec3(0.0, 1.0, 0.0),
          keyL,
          uBasinRadius
        );

        // Real-time Planar Reflection Pass (inverted skyscrapers, Convention Center & lamps)
        vec2 reflNdc = vReflClipPos.xy / max(0.0001, vReflClipPos.w);
        vec2 reflUv = reflNdc * 0.5 + 0.5;
        vec2 rippleDistort = N.xz * vec2(0.055, 0.115);
        vec2 sampleUv = clamp(reflUv + rippleDistort, vec2(0.002), vec2(0.998));
        vec3 planarRefl = texture(uReflectMap, sampleUv).rgb;

        // Studio IBL specular & sharp key softbox catch on Kelvin wakes + capillary crests
        vec3 R = reflect(-V_spec, N);
        float waterRoughness = 0.045;
        vec3 iblStudio = sampleStudioIBL(
          uStudioIblMap,
          R,
          waterRoughness,
          keyL,
          uKeyIntensity,
          uRimIntensity,
          uFillIntensity
        );
        vec3 H = normalize(keyL + V_spec);
        vec3 H_overhead = normalize(normalize(vec3(-0.20, 0.90, 0.38)) + V_spec);
        float wakeCrest = clamp(wakeCrestMask, 0.0, 1.0);
        float wakeKeySpec = pow(max(dot(N, H), 0.0), 150.0) * uKeyIntensity * waterShadow * 1.45
          + (pow(max(dot(N, H), 0.0), 28.0) * 0.22 + pow(max(dot(N, H_overhead), 0.0), 22.0) * 0.18)
            * wakeCrest * uKeyIntensity * mix(0.45, 1.0, waterShadow);

        // Deep dark Beer-Lambert smoked water volume
        float bayDepth = smoothstep(0.0, 2.2, max(0.0, shoreDist));
        vec3 beerAbsorption = exp(-vec3(1.20, 1.02, 0.86) * bayDepth);
        vec3 shallowSlate = vec3(0.0019, 0.0021, 0.0026) * beerAbsorption;
        vec3 deepObsidian = vec3(0.0007, 0.0008, 0.0011);
        vec3 waterVolume = mix(shallowSlate, deepObsidian, 0.72);

        // Crisp pinpoint specular reflections straight down into the rippled water below each esplanade lamp
        float lampGlints = 0.0;
        if (shoreDist > -0.10 && shoreDist < 1.65) {
          float pressBoost = 1.0 + uPressDepth * 0.35;
          for (int i = 0; i < 32; i++) {
            if (i >= uLampCount) break;
            vec2 lp = uLamps[i];
            float dx = (p.x + N.x * 0.085) - lp.x;
            float dz = p.y - lp.y;
            if (abs(dx) < 0.16 && dz > -0.02 && dz < 1.35) {
              // Vertically-rippled pinpoint specular column directly below the lamp
              float pin = exp(-pow(dx / 0.028, 2.0)) * exp(-max(0.0, dz) * 3.1);
              float rippleSparkle = 0.45 + 0.55 * smoothstep(-0.015, 0.028, N.z);
              lampGlints += pin * rippleSparkle * 0.48 * pressBoost;
            }
          }
        }

        float shoreCreviceAo = mix(0.24 - uPressDepth * 0.12, 1.0, pow(smoothstep(-0.04, 0.32, shoreDist), 1.0 + uPressDepth * 0.55));
        float rimDist = uBasinRadius - distFromCenter;
        float basinEdgeFade = smoothstep(0.05, 0.36, rimDist);

        // Combine deep absorption + Fresnel planar skyline reflection + crisp IBL & wake highlights
        float reflMask = (0.04 + 0.96 * fresnel) * uReflectivity * mix(0.42, 1.0, waterShadow);
        vec3 studioGloss = iblStudio * (0.02 + 0.98 * fresnel) * 0.0024 * uReflectivity * mix(0.35, 1.0, waterShadow);

        vec3 finalColor = waterVolume * shoreCreviceAo * mix(0.45, 1.0, waterShadow)
          + planarRefl * reflMask * 1.18
          + studioGloss
          + vec3(0.90, 0.92, 0.96) * (wakeKeySpec + lampGlints * uReflectivity);

        fragColor = vec4(finalColor, basinEdgeFade * 0.97);
      }
    `,
  });
}

/**
 * WebGL2 GLSL3 Dead-Matte Anodized Slate / Architectural Concrete Shader.
 * Used for:
 * - Urban landmass, street grids, plazas & Embarcadero esplanade
 * - Tower podium bases, stepped setback terraces, & Convention Center base colonnade
 * High roughness (~0.88), zero glossy glare, deep urban street canyon AO,
 * tight contact crevice AO, and real soft raymarched cast shadows from surrounding towers.
 */
export function createLandmassObsidianMaterial(
  grainTexture: THREE.Texture,
  groundAoMap: THREE.Texture,
  microImperfectionMap: THREE.Texture,
  studioIblMap: THREE.Texture,
  basinRadius: number,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      uGrainMap: { value: grainTexture },
      uGroundAoMap: { value: groundAoMap },
      uMicroMap: { value: microImperfectionMap },
      uStudioIblMap: { value: studioIblMap },
      uBasinRadius: { value: basinRadius },
      uSoftboxKeyDir: { value: new THREE.Vector3(-0.44, 0.68, 0.58) },
      uKeyIntensity: { value: 1.26 },
      uRimIntensity: { value: 0.92 },
      uFillIntensity: { value: 0.40 },
      uPressDepth: { value: 0.0 },
      uTiltVec: { value: new THREE.Vector2(0, 0) },
      uVelocity: { value: new THREE.Vector2(0, 0) },
    },
    vertexShader: /* glsl */ `
      out vec3 vWorldPos;
      out vec3 vWorldNormal;

      void main() {
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPos = worldPos.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uGrainMap;
      uniform sampler2D uGroundAoMap;
      uniform sampler2D uMicroMap;
      uniform sampler2D uStudioIblMap;
      uniform float uBasinRadius;
      uniform vec3 uSoftboxKeyDir;
      uniform float uKeyIntensity;
      uniform float uRimIntensity;
      uniform float uFillIntensity;
      uniform float uPressDepth;
      uniform vec2 uTiltVec;
      uniform vec2 uVelocity;

      in vec3 vWorldPos;
      in vec3 vWorldNormal;

      out vec4 fragColor;

      ${GLSL3_OPTICS_COMMON}

      float sdfLine(float coord, float freq, float thickness) {
        float scaled = coord * freq;
        float dist = abs(fract(scaled - 0.5) - 0.5);
        float fw = max(fwidth(scaled), 0.001);
        return 1.0 - smoothstep(thickness - fw, thickness + fw, dist);
      }

      void main() {
        vec3 baseN = normalize(vWorldNormal);
        vec3 keyL = normalize(uSoftboxKeyDir);
        vec3 fillL = normalize(vec3(-keyL.x * 0.9, 0.35, -keyL.z * 0.9));

        float grain = texture(uGrainMap, vWorldPos.xz * 0.42).r;
        vec3 micro = texture(uMicroMap, vWorldPos.xz * 0.28).rgb;

        // Dead-matte slate concrete micro-normal perturbation (roughness ~0.88)
        vec3 N = normalize(baseN + vec3((micro.g - 0.5) * 0.04, 0.0, (micro.b - 0.5) * 0.04));

        // Multi-layered Contact AO + deep urban street canyon AO
        vec2 aoSample = sampleUrbanSpatialAo(uGroundAoMap, vWorldPos.xz, uBasinRadius);
        float tightCreviceAo = mix(0.18 - uPressDepth * 0.12, 1.0, pow(aoSample.r, 1.0 + uPressDepth * 0.65));
        float canyonAo = mix(0.24 - uPressDepth * 0.08, 1.0, aoSample.g);
        float combinedAo = tightCreviceAo * canyonAo;

        // Real soft raymarched cast shadows from skyscrapers onto streets, plazas & esplanade
        float castShadow = sampleDirectionalHeightShadow(
          uGroundAoMap,
          vWorldPos,
          baseN,
          keyL,
          uBasinRadius
        );

        // Rotated San Diego downtown street grid
        float c = cos(0.22);
        float s = sin(0.22);
        vec2 streetUv = vec2(
          vWorldPos.x * c - vWorldPos.z * s,
          vWorldPos.x * s + vWorldPos.z * c
        );
        float streetGrid = max(
          sdfLine(streetUv.x, 0.606, 0.024),
          sdfLine(streetUv.y, 0.606, 0.024)
        ) * smoothstep(0.5, 0.95, baseN.y);

        // Strong 3:1 key-to-fill dead-matte diffuse (zero glossy glare, roughness ~0.88)
        float keyLambert = max(dot(N, keyL), 0.0) * castShadow * uKeyIntensity;
        float fillLambert = wrapDiffuse(N, fillL, 0.35) * uFillIntensity * 0.36;
        vec3 studioBounce = sampleStudioIBL(
          uStudioIblMap,
          N,
          0.88,
          keyL,
          uKeyIntensity,
          uRimIntensity,
          uFillIntensity
        );

        float litBalance = 0.16 * uFillIntensity + keyLambert * 0.88 + fillLambert + studioBounce.g * 0.08;
        vec3 slateBase =
          vec3(0.0085, 0.0088, 0.0098) *
          (0.90 + 0.20 * grain) *
          litBalance *
          combinedAo;

        // Seawall beveled concrete curb catches directional key light (no glowing white line)
        float edgeWall = 1.0 - smoothstep(0.22, 0.80, abs(baseN.y));
        float curbKey = max(dot(N, keyL), 0.0) * castShadow * uKeyIntensity;
        vec3 seawallConcrete =
          vec3(0.015, 0.016, 0.018) *
          edgeWall *
          (0.24 * uFillIntensity + curbKey * 1.15) *
          mix(0.45, 1.0, aoSample.g);

        // Subtle promenade streetlight bounce onto wet/matte Embarcadero concrete
        float seawallApproxDist = abs(vWorldPos.z - (2.6 + 2.35 * cos(vWorldPos.x * 0.24 + 0.35) - 0.14 * vWorldPos.x));
        float promenadeSpill = exp(-seawallApproxDist * 1.6) * (0.14 + uPressDepth * 0.65 + length(uVelocity) * 0.18);

        vec3 streetHighlight =
          vec3(0.016, 0.017, 0.020) *
          streetGrid *
          (0.28 + 0.72 * castShadow) *
          combinedAo;

        vec3 finalRgb = slateBase
          + streetHighlight
          + seawallConcrete
          + vec3(0.014, 0.015, 0.018) * promenadeSpill * mix(0.4, 1.0, aoSample.g);
        finalRgb = applyCanyonAtmosphere(finalRgb, vWorldPos, cameraPosition, aoSample.g);
        fragColor = vec4(finalRgb, 1.0);
      }
    `,
  });
}

/**
 * Dead-Matte Anodized Dark Slate / Concrete Material (roughness ~0.88, zero glossy glare).
 * Used for building podiums, stepped setback terraces, mechanical decks, and Convention colonnade.
 */
export function createMatteSlateMaterial(
  grainTexture: THREE.Texture,
  groundAoMap: THREE.Texture,
  microImperfectionMap: THREE.Texture,
  studioIblMap: THREE.Texture,
  basinRadius: number,
  baseColor = 0x1e2025,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      uGrainMap: { value: grainTexture },
      uGroundAoMap: { value: groundAoMap },
      uMicroMap: { value: microImperfectionMap },
      uStudioIblMap: { value: studioIblMap },
      uBaseColor: { value: new THREE.Color(baseColor) },
      uBasinRadius: { value: basinRadius },
      uSoftboxKeyDir: { value: new THREE.Vector3(-0.44, 0.68, 0.58) },
      uKeyIntensity: { value: 1.26 },
      uRimIntensity: { value: 0.92 },
      uFillIntensity: { value: 0.40 },
      uPressDepth: { value: 0.0 },
      uTiltVec: { value: new THREE.Vector2(0, 0) },
      uVelocity: { value: new THREE.Vector2(0, 0) },
      uIsReflectPass: { value: 0.0 },
    },
    vertexShader: /* glsl */ `
      out vec3 vWorldPos;
      out vec3 vWorldNormal;
      out vec2 vUv;

      void main() {
        vUv = uv;
        vec4 localPosition = vec4(position, 1.0);
        vec3 localNormal = normal;
        #ifdef USE_INSTANCING
          localPosition = instanceMatrix * localPosition;
          localNormal = mat3(instanceMatrix) * localNormal;
        #endif
        vec4 worldPos = modelMatrix * localPosition;
        vWorldPos = worldPos.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * localNormal);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uGrainMap;
      uniform sampler2D uGroundAoMap;
      uniform sampler2D uMicroMap;
      uniform sampler2D uStudioIblMap;
      uniform vec3 uBaseColor;
      uniform float uBasinRadius;
      uniform vec3 uSoftboxKeyDir;
      uniform float uKeyIntensity;
      uniform float uRimIntensity;
      uniform float uFillIntensity;
      uniform float uPressDepth;
      uniform vec2 uTiltVec;
      uniform vec2 uVelocity;
      uniform float uIsReflectPass;

      in vec3 vWorldPos;
      in vec3 vWorldNormal;
      in vec2 vUv;

      out vec4 fragColor;

      ${GLSL3_OPTICS_COMMON}

      void main() {
        vec3 baseN = normalize(vWorldNormal);
        if (uIsReflectPass > 0.5 && baseN.y < -0.5) {
          discard;
        }
        vec3 keyL = normalize(uSoftboxKeyDir);
        vec3 fillL = normalize(vec3(-keyL.x * 0.9, 0.35, -keyL.z * 0.9));

        float grain = texture(uGrainMap, vWorldPos.xz * 0.45 + vWorldPos.y * 0.35).r;
        vec3 micro = texture(uMicroMap, vWorldPos.xz * 0.32 + vec2(vWorldPos.y * 0.25, 0.0)).rgb;

        // Subtle directional bevel normal along corners (zero glow on shadow side)
        vec4 chamferRes = computeMachinedChamfer(baseN, vWorldPos, vUv, 0.42, 1.0);
        vec3 N = normalize(chamferRes.xyz + vec3((micro.g - 0.5) * 0.025));

        float castShadow = sampleDirectionalHeightShadow(
          uGroundAoMap,
          vWorldPos,
          baseN,
          keyL,
          uBasinRadius
        );
        vec2 spatialAo = sampleUrbanSpatialAo(uGroundAoMap, vWorldPos.xz, uBasinRadius);
        float baseAo = mix(0.42 - uPressDepth * 0.14, 1.0, smoothstep(0.08, 0.48, vWorldPos.y));
        float canyonAo = mix(0.36, 1.0, spatialAo.g);

        // 3:1 directional key vs cool dark fill (dead-matte roughness ~0.88, zero glossy glare)
        float keyDiffuse = max(dot(N, keyL), 0.0) * castShadow * uKeyIntensity;
        float fillDiffuse = wrapDiffuse(N, fillL, 0.35) * uFillIntensity * 0.35;
        vec3 studioDiff = sampleStudioIBL(
          uStudioIblMap,
          N,
          0.88,
          keyL,
          uKeyIntensity,
          uRimIntensity,
          uFillIntensity
        );

        float lit = 0.15 * uFillIntensity + keyDiffuse * 0.95 + fillDiffuse + studioDiff.g * 0.09;
        vec3 color = uBaseColor * (0.88 + 0.22 * grain) * lit * baseAo * canyonAo;

        // Subtle directional matte bevel highlight strictly on faces facing key light (zero shadow-side outline)
        float faceKeyVis = smoothstep(-0.08, 0.25, dot(baseN, keyL));
        float bevelKey = pow(max(dot(chamferRes.xyz, keyL), 0.0), 1.8) * castShadow * faceKeyVis * chamferRes.w;
        color += vec3(0.016, 0.017, 0.020) * bevelKey * uKeyIntensity;
        color = applyCanyonAtmosphere(color, vWorldPos, cameraPosition, spatialAo.g);

        fragColor = vec4(color, 1.0);
      }
    `,
  });
}

export interface SdfGlassOptions {
  mullionDensity?: number;
  floorDensity?: number;
  baseTint?: number;
  edgeLuminance?: number;
  isHero?: boolean;
  allowSideChamfers?: boolean;
}

/**
 * WebGL2 GLSL3 Architectural Curtain-Wall Glass Shader (Museum Scale-Model Realism):
 * - Smoked dark architectural glass with crisp horizontal spandrel bands & vertical mullions
 * - Per-pane deterministic normal micro-jitter (±0.03) so adjacent glass panes catch the
 *   studio overhead softbox at slightly different angles (zero salt-and-pepper noise!)
 * - Sparse, clean softly glowing horizontal interior floor bands
 * - Strong 3:1 key-to-fill studio shading + real raymarched cast shadows onto neighboring facades & roofs
 * - Dead-matte dark slate concrete rooftops (roughness ~0.88, zero glossy glare)
 */
export function createSdfGlassMaterial(
  grainTexture: THREE.Texture,
  studioIblMap: THREE.Texture,
  microImperfectionMap: THREE.Texture,
  groundAoMap: THREE.Texture,
  basinRadius: number,
  options: SdfGlassOptions = {},
): THREE.ShaderMaterial {
  const {
    mullionDensity = 12.0,
    floorDensity = 16.0,
    baseTint = 0x111317,
    edgeLuminance = 0.85,
    isHero = true,
    allowSideChamfers = true,
  } = options;

  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      uGrainMap: { value: grainTexture },
      uStudioIblMap: { value: studioIblMap },
      uMicroMap: { value: microImperfectionMap },
      uGroundAoMap: { value: groundAoMap },
      uBasinRadius: { value: basinRadius },
      uBaseColor: { value: new THREE.Color(baseTint) },
      uSoftboxKeyDir: { value: new THREE.Vector3(-0.44, 0.68, 0.58) },
      uKeyIntensity: { value: 1.26 },
      uRimIntensity: { value: 0.92 },
      uFillIntensity: { value: 0.40 },
      uMullionDensity: { value: mullionDensity },
      uFloorDensity: { value: floorDensity },
      uEdgeLuminance: { value: edgeLuminance },
      uIsHero: { value: isHero ? 1.0 : 0.48 },
      uAllowSideChamfers: { value: allowSideChamfers ? 1.0 : 0.0 },
      uHighlightBoost: { value: 0.0 },
      uPressDepth: { value: 0.0 },
      uTiltVec: { value: new THREE.Vector2(0, 0) },
      uVelocity: { value: new THREE.Vector2(0, 0) },
      uIsReflectPass: { value: 0.0 },
    },
    vertexShader: /* glsl */ `
      out vec3 vWorldPos;
      out vec3 vWorldNormal;
      out vec3 vLocalPos;
      out vec2 vUv;

      void main() {
        vUv = uv;
        vLocalPos = position;

        vec4 localPosition = vec4(position, 1.0);
        vec3 localNormal = normal;
        #ifdef USE_INSTANCING
          localPosition = instanceMatrix * localPosition;
          localNormal = mat3(instanceMatrix) * localNormal;
        #endif

        vec4 worldPosition = modelMatrix * localPosition;
        vWorldPos = worldPosition.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * localNormal);

        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uGrainMap;
      uniform sampler2D uStudioIblMap;
      uniform sampler2D uMicroMap;
      uniform sampler2D uGroundAoMap;
      uniform float uBasinRadius;
      uniform vec3 uBaseColor;
      uniform vec3 uSoftboxKeyDir;
      uniform float uKeyIntensity;
      uniform float uRimIntensity;
      uniform float uFillIntensity;
      uniform float uMullionDensity;
      uniform float uFloorDensity;
      uniform float uEdgeLuminance;
      uniform float uIsHero;
      uniform float uAllowSideChamfers;
      uniform float uHighlightBoost;
      uniform float uPressDepth;
      uniform vec2 uTiltVec;
      uniform vec2 uVelocity;
      uniform float uIsReflectPass;

      in vec3 vWorldPos;
      in vec3 vWorldNormal;
      in vec3 vLocalPos;
      in vec2 vUv;

      out vec4 fragColor;

      ${GLSL3_OPTICS_COMMON}

      float sdfGridLine(float coord, float freq, float thickness) {
        float scaled = coord * freq;
        float distToEdge = abs(fract(scaled - 0.5) - 0.5);
        float fw = max(fwidth(scaled), 0.001);
        return 1.0 - smoothstep(thickness - fw, thickness + fw, distToEdge);
      }

      float hash21(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
      }

      void main() {
        vec3 baseN = normalize(vWorldNormal);
        if (uIsReflectPass > 0.5 && baseN.y < -0.5) {
          discard;
        }
        vec3 V_raw = normalize(cameraPosition - vWorldPos);
        vec3 V = applySpecularParallax(V_raw, uTiltVec, uVelocity, 1.0);
        vec3 keyL = normalize(uSoftboxKeyDir);
        vec3 rimL = normalize(vec3(-0.82, 0.34, 0.44));
        vec3 fillL = normalize(vec3(-keyL.x * 0.9, 0.32, -keyL.z * 0.9));

        float grain = texture(uGrainMap, vWorldPos.xz * 0.42 + vWorldPos.y * 0.32).r;
        vec3 micro = texture(uMicroMap, vWorldPos.xz * 0.30 + vec2(vWorldPos.y * 0.25, 0.0)).rgb;

        // Directional machined chamfer bevel normal along corners & parapets
        vec4 chamferResult = computeMachinedChamfer(
          baseN,
          vWorldPos,
          vUv,
          0.48,
          uAllowSideChamfers
        );
        vec3 chamferShift = vec3(uTiltVec.x * 0.14, -uPressDepth * 0.10, uTiltVec.y * 0.14) * chamferResult.w;
        vec3 N = normalize(chamferResult.xyz + chamferShift);
        float chamferMask = chamferResult.w;

        // Distinguish dead-matte rooftop planes vs vertical glass curtain walls
        float isRoof = smoothstep(0.65, 0.90, abs(baseN.y));

        // Real cast height shadows & urban canyon AO
        float castShadow = sampleDirectionalHeightShadow(
          uGroundAoMap,
          vWorldPos,
          baseN,
          keyL,
          uBasinRadius
        );
        vec2 spatialAo = sampleUrbanSpatialAo(uGroundAoMap, vWorldPos.xz, uBasinRadius);
        float groundCreviceAo = mix(0.30 - uPressDepth * 0.14, 1.0, pow(smoothstep(0.08, 0.58, vWorldPos.y), 1.0 + uPressDepth * 0.55));
        float tierCreviceAo = mix(0.60 - uPressDepth * 0.10, 1.0, pow(smoothstep(0.0, 0.14, vUv.y), 1.0 + uPressDepth * 0.45));
        float canyonAo = mix(0.45, 1.0, mix(spatialAo.g, 1.0, clamp(vWorldPos.y / 3.8, 0.0, 1.0)));
        float contactAo = groundCreviceAo * tierCreviceAo * canyonAo;

        // Exact wall-aligned horizontal tangent so panes never stretch or skew on angled facades
        vec2 horizNormal = length(baseN.xz) > 0.01 ? normalize(baseN.xz) : vec2(0.0, 1.0);
        vec3 wallT = vec3(horizNormal.y, 0.0, -horizNormal.x);
        float wallX = dot(vWorldPos.xz, wallT.xz);

        float mullionFreq = uMullionDensity * 0.44;
        float floorFreq = uFloorDensity * 0.42;

        // Crisp horizontal spandrel bands (~26% of floor height) & vertical mullions
        float floorScaled = vWorldPos.y * floorFreq;
        float floorPhase = fract(floorScaled);
        float fwFloor = max(fwidth(floorScaled), 0.002);
        float spandrelMask = (1.0 - smoothstep(0.24 - fwFloor, 0.24 + fwFloor, floorPhase)) * (1.0 - isRoof);
        float vMullion = sdfGridLine(wallX, mullionFreq, 0.024) * (1.0 - isRoof);
        float frameMask = clamp(max(spandrelMask, vMullion), 0.0, 1.0);

        // Deterministic per-pane normal micro-jitter (±0.04) inside glass vision bays
        vec2 paneCell = floor(vec2(wallX * mullionFreq + 0.5, floorScaled));
        float jx = (hash21(paneCell) - 0.5) * 0.08;
        float jy = (hash21(paneCell + vec2(31.7, 17.3)) - 0.5) * 0.08;
        vec3 paneN = normalize(N + (wallT * jx + vec3(0.0, 1.0, 0.0) * jy) * (1.0 - frameMask) * (1.0 - isRoof));

        float NdotV = clamp(dot(paneN, V), 0.0, 1.0);
        float fresnel = fresnelSchlick(NdotV, 0.062);

        // Glass roughness ~0.075 with subtle 2% variation vs matte dark anodized spandrels (~0.66)
        float glassRoughness = clamp(
          mix(0.074 + (micro.r - 0.5) * 0.022, 0.66, frameMask),
          0.045,
          0.70
        );

        vec3 R_g = reflect(-V, paneN);
        vec3 refrR = refract(-V, paneN, 1.0 / 1.505);
        vec3 refrB = refract(-V, paneN, 1.0 / 1.535);
        vec3 dirR = normalize(mix(R_g, refrR, 0.035));
        vec3 dirB = normalize(mix(R_g, refrB, 0.035));

        float iblR = sampleStudioIBL(uStudioIblMap, dirR, glassRoughness, keyL, uKeyIntensity, uRimIntensity, uFillIntensity).r;
        float iblG = sampleStudioIBL(uStudioIblMap, R_g, glassRoughness, keyL, uKeyIntensity, uRimIntensity, uFillIntensity).g;
        float iblB = sampleStudioIBL(uStudioIblMap, dirB, glassRoughness, keyL, uKeyIntensity, uRimIntensity, uFillIntensity).b;
        vec3 studioIbl = vec3(iblR, iblG, iblB);

        // 3:1 key-to-fill lighting across vertical tower faces + real cast shadow darkening
        float keyFace = max(dot(paneN, keyL), 0.0) * castShadow;
        float fillFace = wrapDiffuse(paneN, fillL, 0.22) * uFillIntensity * 0.24;
        float keyDiffuseFactor = 0.12 * uFillIntensity + keyFace * uKeyIntensity * 0.96 + fillFace;

        // Beer-Lambert smoked glass depth absorption
        float opticalThickness = 0.55 / max(0.26, NdotV);
        vec3 beerLambert = exp(-vec3(0.54, 0.49, 0.43) * opticalThickness);

        // Sparse, clean softly glowing horizontal interior floor bands (zero static noise!)
        float floorRand = hash21(vec2(paneCell.y, 13.7));
        float wingBucket = floor(wallX * mullionFreq * 0.22);
        float wingRand = hash21(vec2(wingBucket, paneCell.y + 7.1));
        float occupiedBand = step(0.84, floorRand) * smoothstep(0.35, 0.65, wingRand);
        float visionVert = smoothstep(0.24, 0.34, floorPhase) * (1.0 - smoothstep(0.82, 0.98, floorPhase));
        float interiorLuminance = occupiedBand * visionVert * (1.0 - frameMask) * (0.072 + 0.052 * uIsHero);
        vec3 interiorGlow = vec3(0.64, 0.66, 0.72) * interiorLuminance * (1.0 + uPressDepth * 0.35) * beerLambert;

        // Smoked glass body vs matte dark anodized spandrel/mullion strips
        vec3 glassBody = uBaseColor * beerLambert * (0.55 + 0.88 * keyDiffuseFactor);
        vec3 spandrelBody = vec3(0.011, 0.012, 0.014) * (0.30 + 1.18 * keyDiffuseFactor);
        vec3 facadeBase = mix(glassBody, spandrelBody, frameMask);

        // Per-pane studio IBL specular reflection: crisp pane-to-pane variation on sunlit side, dark glass on shadow side
        float reflVisibility = mix(0.22, 1.0, keyFace * 0.85 + 0.15) * mix(0.35, 1.0, castShadow);
        vec3 glassReflection = studioIbl * (0.08 + 0.92 * fresnel) * 0.068 * (1.0 - frameMask * 0.82) * reflVisibility;

        // Subtle directional chamfer catchlight ONLY on sunlit-facing corners (zero wireframe outline on shadow edges!)
        float faceKeyVis = smoothstep(-0.08, 0.25, dot(baseN, keyL));
        float chamferKeyFacing = pow(max(dot(N, keyL), 0.0), 2.0) * castShadow * uKeyIntensity * faceKeyVis;
        float chamferRimFacing = pow(max(dot(N, rimL), 0.0), 2.4) * uRimIntensity * 0.28 * smoothstep(-0.1, 0.25, dot(baseN, rimL));
        float directionalEdge = (chamferKeyFacing + chamferRimFacing) * chamferMask;
        vec3 chamferCatch = vec3(0.24, 0.26, 0.30) * directionalEdge * uEdgeLuminance * (0.25 + 0.35 * uIsHero + uPressDepth * 0.12);

        vec3 glassLit = (facadeBase + interiorGlow) * contactAo
          + glassReflection * mix(0.55, 1.0, contactAo)
          + chamferCatch
          + vec3(0.18, 0.20, 0.24) * uHighlightBoost * 0.35;

        // Dead-matte architectural slate/concrete rooftop (roughness ~0.88, zero glossy glare, no bright perimeter border)
        float roofGrid = max(
          sdfGridLine(vWorldPos.x, 1.8, 0.020),
          sdfGridLine(vWorldPos.z, 1.8, 0.020)
        );
        float roofKey = max(dot(baseN, keyL), 0.0) * castShadow * uKeyIntensity;
        float roofFill = wrapDiffuse(baseN, fillL, 0.35) * uFillIntensity * 0.30;
        float roofLit = 0.14 * uFillIntensity + roofKey * 0.96 + roofFill;
        float roofBevel = pow(max(dot(N, keyL), 0.0), 2.0) * castShadow * chamferMask * smoothstep(0.0, 0.4, dot(N - baseN, keyL));

        vec3 roofColor =
          vec3(0.0085, 0.0089, 0.0099) * (0.88 + 0.22 * grain) * roofLit * canyonAo
          + vec3(0.010) * roofGrid * (0.30 + 0.70 * castShadow) * canyonAo
          + vec3(0.014, 0.015, 0.017) * roofBevel * uKeyIntensity;

        vec3 finalRgb = mix(glassLit, roofColor, isRoof);
        finalRgb = applyCanyonAtmosphere(finalRgb, vWorldPos, cameraPosition, spatialAo.g);
        fragColor = vec4(finalRgb, 1.0);
      }
    `,
  });
}

export interface AnisotropicMetalOptions {
  baseColor?: number;
  radialAnisotropy?: boolean;
  anisotropyStrength?: number;
  roughness?: number;
  allowSideChamfers?: boolean;
}

/**
 * WebGL2 GLSL3 Satin Brushed Architectural Metal Shader (roughness ~0.35).
 * Used for Brookfield/One America Plaza pyramid crown, Emerald caps, Harbor Club lanterns,
 * Convention Center curved Sails Pavilion vaults, USS Midway, and turntable plinth rim.
 * Features directional chamfer edge catchlights, smooth gradient highlights along curves/facets,
 * and real cast heightfield shadows.
 */
export function createAnisotropicMetalMaterial(
  brushedTexture: THREE.Texture,
  grainTexture: THREE.Texture,
  studioIblMap: THREE.Texture,
  microImperfectionMap: THREE.Texture,
  groundAoMap: THREE.Texture,
  basinRadius: number,
  options: AnisotropicMetalOptions = {},
): THREE.ShaderMaterial {
  const {
    baseColor = 0x2c2f37,
    radialAnisotropy = false,
    anisotropyStrength = 0.80,
    roughness = 0.35,
    allowSideChamfers = true,
  } = options;

  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      uBrushedMap: { value: brushedTexture },
      uGrainMap: { value: grainTexture },
      uStudioIblMap: { value: studioIblMap },
      uMicroMap: { value: microImperfectionMap },
      uGroundAoMap: { value: groundAoMap },
      uBasinRadius: { value: basinRadius },
      uBaseColor: { value: new THREE.Color(baseColor) },
      uSoftboxKeyDir: { value: new THREE.Vector3(-0.44, 0.68, 0.58) },
      uKeyIntensity: { value: 1.26 },
      uRimIntensity: { value: 0.92 },
      uFillIntensity: { value: 0.40 },
      uRadialAnisotropy: { value: radialAnisotropy ? 1.0 : 0.0 },
      uAnisotropy: { value: anisotropyStrength },
      uRoughness: { value: roughness },
      uAllowSideChamfers: { value: allowSideChamfers ? 1.0 : 0.0 },
      uPressDepth: { value: 0.0 },
      uTiltVec: { value: new THREE.Vector2(0, 0) },
      uVelocity: { value: new THREE.Vector2(0, 0) },
      uIsReflectPass: { value: 0.0 },
    },
    vertexShader: /* glsl */ `
      out vec3 vWorldPos;
      out vec3 vWorldNormal;
      out vec2 vUv;

      void main() {
        vUv = uv;
        vec4 localPosition = vec4(position, 1.0);
        vec3 localNormal = normal;
        #ifdef USE_INSTANCING
          localPosition = instanceMatrix * localPosition;
          localNormal = mat3(instanceMatrix) * localNormal;
        #endif
        vec4 worldPos = modelMatrix * localPosition;
        vWorldPos = worldPos.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * localNormal);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uBrushedMap;
      uniform sampler2D uGrainMap;
      uniform sampler2D uStudioIblMap;
      uniform sampler2D uMicroMap;
      uniform sampler2D uGroundAoMap;
      uniform float uBasinRadius;
      uniform vec3 uBaseColor;
      uniform vec3 uSoftboxKeyDir;
      uniform float uKeyIntensity;
      uniform float uRimIntensity;
      uniform float uFillIntensity;
      uniform float uRadialAnisotropy;
      uniform float uAnisotropy;
      uniform float uRoughness;
      uniform float uAllowSideChamfers;
      uniform float uPressDepth;
      uniform vec2 uTiltVec;
      uniform vec2 uVelocity;
      uniform float uIsReflectPass;

      in vec3 vWorldPos;
      in vec3 vWorldNormal;
      in vec2 vUv;

      out vec4 fragColor;

      ${GLSL3_OPTICS_COMMON}

      float wardAnisotropic(vec3 N, vec3 V, vec3 L, vec3 T, vec3 B, float ax, float ay) {
        vec3 H = normalize(L + V);
        float NdotL = max(dot(N, L), 0.001);
        float NdotV = max(dot(N, V), 0.001);
        float NdotH = max(dot(N, H), 0.001);
        float HdotT = dot(H, T);
        float HdotB = dot(H, B);
        float exponent = -2.0 * (pow(HdotT / ax, 2.0) + pow(HdotB / ay, 2.0)) / (1.0 + NdotH);
        float norm = 1.0 / (4.0 * 3.14159265 * ax * ay * sqrt(NdotL * NdotV));
        return clamp(norm * exp(exponent) * NdotL, 0.0, 2.5);
      }

      void main() {
        vec3 baseN = normalize(vWorldNormal);
        if (uIsReflectPass > 0.5 && baseN.y < -0.5) {
          discard;
        }
        vec3 V_raw = normalize(cameraPosition - vWorldPos);
        vec3 V = applySpecularParallax(V_raw, uTiltVec, uVelocity, 1.15);

        vec4 chamferResult = uRadialAnisotropy > 0.5
          ? vec4(baseN, 0.0)
          : computeMachinedChamfer(
              baseN,
              vWorldPos,
              vUv,
              0.38,
              uAllowSideChamfers
            );
        vec3 chamferShift = vec3(uTiltVec.x * 0.12, -uPressDepth * 0.10, uTiltVec.y * 0.12) * chamferResult.w;
        vec3 chamferN = normalize(chamferResult.xyz + chamferShift);
        float chamferMask = chamferResult.w;

        vec3 refAxis = uRadialAnisotropy > 0.5
          ? normalize(vec3(-vWorldPos.z, 0.001, vWorldPos.x))
          : vec3(0.0, 1.0, 0.0);
        if (abs(dot(chamferN, refAxis)) > 0.95) {
          refAxis = vec3(1.0, 0.0, 0.0);
        }
        vec3 T = normalize(cross(refAxis, chamferN));
        vec3 B = normalize(cross(chamferN, T));

        vec2 brushUv = uRadialAnisotropy > 0.5
          ? vec2(atan(vWorldPos.z, vWorldPos.x) * 2.5, length(vWorldPos.xz) * 0.6)
          : vec2(vWorldPos.x + vWorldPos.z, vWorldPos.y) * 0.75;
        float brush = texture(uBrushedMap, brushUv).r;
        float grain = texture(uGrainMap, vWorldPos.xz * 0.5).r;
        vec3 micro = texture(uMicroMap, brushUv * 1.4).rgb;

        vec3 N = normalize(chamferN + B * ((brush - 0.5) * 0.032 + (micro.b - 0.5) * 0.020));
        float NdotV = clamp(dot(N, V), 0.0, 1.0);

        float localRoughness = clamp(
          uRoughness + (brush - 0.5) * 0.04 + (micro.r - 0.5) * 0.028 - chamferMask * 0.035,
          0.18,
          0.52
        );

        float ax = max(0.10, localRoughness * (1.0 - uAnisotropy * 0.52));
        float ay = max(0.22, localRoughness * (1.0 + uAnisotropy * 0.82));

        vec3 keyL = normalize(uSoftboxKeyDir);
        vec3 rimL = normalize(vec3(-0.82, 0.34, 0.44));
        vec3 fillL = normalize(vec3(-keyL.x * 0.9, 0.32, -keyL.z * 0.9));

        float castShadow = uRadialAnisotropy > 0.5
          ? 1.0
          : sampleDirectionalHeightShadow(uGroundAoMap, vWorldPos, baseN, keyL, uBasinRadius);

        float anisoSpec =
          wardAnisotropic(N, V, keyL, T, B, ax, ay) * uKeyIntensity * castShadow * 0.72 +
          wardAnisotropic(N, V, rimL, T, B, ax, ay) * uRimIntensity * 0.42 +
          wardAnisotropic(N, V, fillL, T, B, ax * 1.35, ay * 1.35) * uFillIntensity * 0.16;

        vec3 bentN = normalize(N - B * clamp(dot(B, V), -0.32, 0.32) * uAnisotropy * 0.42);
        vec3 R = reflect(-V, bentN);
        vec3 iblSpec = sampleStudioIBL(
          uStudioIblMap,
          R,
          localRoughness * (0.85 + 0.22 * micro.g),
          keyL,
          uKeyIntensity,
          uRimIntensity,
          uFillIntensity
        );

        float baseAo = uRadialAnisotropy > 0.5
          ? 1.0
          : mix(0.42 - uPressDepth * 0.14, 1.0, pow(smoothstep(0.06, 0.42, vWorldPos.y), 1.0 + uPressDepth * 0.50));
        float specScale = uRadialAnisotropy > 0.5 ? 0.065 : 0.42;

        float fresnel = fresnelSchlick(NdotV, 0.58);
        // Strong 3:1 key-to-fill directional shading
        float keyDiff = max(dot(N, keyL), 0.0) * castShadow * uKeyIntensity;
        float rimDiff = max(dot(N, rimL), 0.0) * uRimIntensity * 0.25;
        float fillDiff = wrapDiffuse(N, fillL, 0.25) * uFillIntensity * 0.28;
        float diffuse = 0.14 * uFillIntensity + keyDiff * 0.72 + rimDiff + fillDiff;

        // Directional chamfer catchlight strictly facing key/rim softboxes
        float faceKeyVis = smoothstep(-0.08, 0.25, dot(baseN, keyL));
        float chamferKeyFacing = pow(max(dot(N, keyL), 0.0), 1.8) * castShadow * uKeyIntensity * faceKeyVis;
        float chamferRimFacing = pow(max(dot(N, rimL), 0.0), 2.2) * uRimIntensity * 0.32 * smoothstep(-0.1, 0.25, dot(baseN, rimL));
        float dirChamfer = (chamferKeyFacing + chamferRimFacing) * chamferMask;

        vec3 metalAlbedo = uBaseColor * (0.88 + 0.24 * brush) * (0.94 + 0.12 * grain);
        vec3 color = metalAlbedo * diffuse * baseAo
          + vec3(0.84, 0.87, 0.92) * (anisoSpec * 0.22 * baseAo * specScale)
          + iblSpec * (0.04 + 0.14 * fresnel) * baseAo * specScale * mix(0.45, 1.0, castShadow)
          + vec3(0.24, 0.26, 0.30) * dirChamfer * (1.0 + uPressDepth * 0.2) * specScale;

        if (uRadialAnisotropy < 0.5) {
          vec2 spatialAo = sampleUrbanSpatialAo(uGroundAoMap, vWorldPos.xz, uBasinRadius);
          color = applyCanyonAtmosphere(color, vWorldPos, cameraPosition, spatialAo.g);
        }

        fragColor = vec4(color, 1.0);
      }
    `,
  });
}

/**
 * WebGL2 GLSL3 Studio Cyclorama Floor & Turntable Ground Contact Shadow Shader.
 * Places the cylindrical plinth onto a continuous dark studio tabletop with a subtle
 * softbox key pool and a soft, asymmetrical oval/circular contact shadow directly beneath the plinth.
 */
export function createCycloramaFloorMaterial(
  grainTexture: THREE.Texture,
  basinRadius: number,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    transparent: true,
    depthWrite: false,
    uniforms: {
      uGrainMap: { value: grainTexture },
      uBasinRadius: { value: basinRadius },
      uSoftboxKeyDir: { value: new THREE.Vector3(-0.44, 0.68, 0.58) },
      uKeyIntensity: { value: 1.26 },
      uRimIntensity: { value: 0.92 },
      uFillIntensity: { value: 0.40 },
      uPressDepth: { value: 0.0 },
      uTiltVec: { value: new THREE.Vector2(0, 0) },
      uVelocity: { value: new THREE.Vector2(0, 0) },
    },
    vertexShader: /* glsl */ `
      out vec3 vWorldPos;

      void main() {
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPos = worldPos.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uGrainMap;
      uniform float uBasinRadius;
      uniform vec3 uSoftboxKeyDir;
      uniform float uKeyIntensity;
      uniform float uRimIntensity;
      uniform float uFillIntensity;
      uniform float uPressDepth;
      uniform vec2 uTiltVec;
      uniform vec2 uVelocity;

      in vec3 vWorldPos;
      out vec4 fragColor;

      void main() {
        vec2 p = vWorldPos.xz;
        float plinthR = uBasinRadius + 0.34;
        float dist = length(p);

        // Seamless radial cyclorama fade into exact studio background (#131315 linear ~0.0052)
        float outerFade = smoothstep(plinthR + 0.8, plinthR + 7.2, dist);
        if (outerFade >= 0.999) {
          discard;
        }

        // Subtle studio tabletop cyclorama sweep (slightly brighter under upper-left key pool)
        vec3 keyL = normalize(uSoftboxKeyDir);
        vec2 keyHoriz = normalize(keyL.xz);
        float keyPool = smoothstep(20.0, -8.0, dot(p, -keyHoriz));
        float centerPool = exp(-pow(dist / (plinthR * 1.38), 2.0));
        float fineGrain = texture(uGrainMap, p * 0.25).r;

        vec3 bgLinear = vec3(0.0046, 0.0046, 0.0054);
        vec3 tabletopLit = mix(
          bgLinear,
          vec3(0.0175, 0.0180, 0.0205),
          centerPool * (0.52 + 0.48 * keyPool) * (0.72 + 0.28 * uKeyIntensity)
        ) * (0.95 + 0.10 * fineGrain);

        // 1. Tight omnidirectional contact occlusion directly under the beveled plinth base
        float tightRimDist = max(0.0, dist - (plinthR - 0.12));
        float tightContactAo = 1.0 - exp(-tightRimDist * 2.1) * (0.88 + uPressDepth * 0.10);

        // 2. Soft asymmetrical oval key-cast shadow shifted opposite uSoftboxKeyDir.xz
        vec2 shadowShift = -keyHoriz * (1.35 + (1.0 - clamp(keyL.y, 0.25, 0.9)) * 1.45);
        vec2 rel = p - shadowShift;
        float along = dot(rel, keyHoriz) / (plinthR * 1.08);
        vec2 perpVec = vec2(-keyHoriz.y, keyHoriz.x);
        float across = dot(rel, perpVec) / (plinthR * 1.01);
        float ovalDist = sqrt(along * along + across * across);
        float ovalPenumbra = 1.0 - smoothstep(0.86, 1.24, ovalDist) * 0.82;
        float castShadowMask = mix(1.0, 0.14, 1.0 - ovalPenumbra);

        float totalShadow = min(tightContactAo, castShadowMask);
        vec3 groundedFloor = mix(tabletopLit * totalShadow, bgLinear, outerFade);

        fragColor = vec4(groundedFloor, 1.0 - outerFade);
      }
    `,
  });
}

/**
 * WebGL2 GLSL3 Matte Sailcloth / Composite Resin Material for 3D miniature sailboats.
 * Catches directional studio key light across cambered sail faces with realistic fold shadow.
 */
export function createSailFabricMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    side: THREE.DoubleSide,
    uniforms: {
      uSoftboxKeyDir: { value: new THREE.Vector3(-0.44, 0.68, 0.58) },
      uKeyIntensity: { value: 1.26 },
      uRimIntensity: { value: 0.92 },
      uFillIntensity: { value: 0.40 },
      uPressDepth: { value: 0.0 },
      uTiltVec: { value: new THREE.Vector2(0, 0) },
      uVelocity: { value: new THREE.Vector2(0, 0) },
    },
    vertexShader: /* glsl */ `
      out vec3 vWorldPos;
      out vec3 vWorldNormal;

      void main() {
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPos = worldPos.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uSoftboxKeyDir;
      uniform float uKeyIntensity;
      uniform float uRimIntensity;
      uniform float uFillIntensity;

      in vec3 vWorldPos;
      in vec3 vWorldNormal;

      out vec4 fragColor;

      void main() {
        vec3 N = normalize(vWorldNormal);
        if (!gl_FrontFacing) N = -N;
        vec3 keyL = normalize(uSoftboxKeyDir);
        vec3 V = normalize(cameraPosition - vWorldPos);

        float keyDiff = max(dot(N, keyL), 0.0) * uKeyIntensity;
        // Translucent sail backlight when studio softbox shines through the sailcloth
        float translucent = pow(clamp(dot(-V, keyL), 0.0, 1.0), 2.0) * 0.32 * uKeyIntensity;
        float fill = 0.24 * uFillIntensity;
        float rim = pow(1.0 - clamp(abs(dot(N, V)), 0.0, 1.0), 2.4) * 0.14 * uRimIntensity;

        vec3 sailBase = vec3(0.42, 0.44, 0.48);
        vec3 color = sailBase * (fill + keyDiff * 0.85 + translucent) + vec3(rim);
        fragColor = vec4(color, 1.0);
      }
    `,
  });
}
