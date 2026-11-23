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
const float PI = 3.14159265359;
const float INV_TWO_PI = 0.15915494309;
const vec3 STUDIO_KEY_WARM = vec3(0.996, 1.003, 1.012);
const vec3 STUDIO_FILL_COOL = vec3(0.935, 0.972, 1.052);
const vec3 STUDIO_INTERIOR_ALABASTER = vec3(0.816, 0.824, 0.840);
const vec3 STUDIO_INTERIOR_CHAMPAGNE = vec3(0.774, 0.788, 0.814);
const vec3 STUDIO_INTERIOR_DAYLIGHT = vec3(0.742, 0.794, 0.880);
const vec3 STUDIO_INTERIOR_LOBBY = vec3(0.706, 0.715, 0.732);
const vec3 STUDIO_WARM_POOL_3000K = vec3(0.696, 0.707, 0.728);
const vec3 STUDIO_PLATINUM_RIM = vec3(0.88, 0.92, 0.99);

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

float unpackDepth16(vec2 rg) {
  return (rg.r * 255.0 + rg.g) / 255.0;
}

/**
 * Real-time Orthographic Depth Shadow Map with 3x3 PCF soft penumbra filtering.
 * Tall towers cast crisp, grounded soft shadows across lower rooftops, street grids,
 * neighboring facades, and the harbor waterline.
 */
float sampleOrthographicShadowPCF(
  sampler2D shadowMap,
  mat4 shadowMatrix,
  vec2 shadowTexelSize,
  vec3 worldPos,
  vec3 surfaceN,
  vec3 keyDir,
  float hasShadowMap
) {
  if (hasShadowMap < 0.5) {
    return 1.0;
  }
  vec3 dir = normalize(keyDir);
  float slope = 1.0 - clamp(dot(surfaceN, dir), 0.0, 1.0);
  vec3 biasedWorld = worldPos + surfaceN * (0.028 + 0.040 * slope);
  vec4 shadowClip = shadowMatrix * vec4(biasedWorld, 1.0);
  vec3 shadowCoord = (shadowClip.xyz / max(0.0001, shadowClip.w)) * 0.5 + 0.5;

  if (
    shadowCoord.x <= 0.002 || shadowCoord.x >= 0.998 ||
    shadowCoord.y <= 0.002 || shadowCoord.y >= 0.998 ||
    shadowCoord.z <= 0.001 || shadowCoord.z >= 0.999
  ) {
    return 1.0;
  }

  float depthBias = 0.0018 + 0.0024 * slope;
  vec2 texel = shadowTexelSize * 1.25;
  float sum = 0.0;

  for (int ix = -1; ix <= 1; ix++) {
    for (int iy = -1; iy <= 1; iy++) {
      vec2 suv = clamp(shadowCoord.xy + vec2(float(ix), float(iy)) * texel, vec2(0.001), vec2(0.999));
      float occluderDepth = unpackDepth16(textureLod(shadowMap, suv, 0.0).rg);
      sum += (shadowCoord.z - depthBias > occluderDepth) ? 0.0 : 1.0;
    }
  }

  return clamp(mix(0.035, 1.0, sum / 9.0), 0.035, 1.0);
}

/**
 * Soft raymarched directional cast shadows using the baked 512x512 skyline heightfield
 * stored in spatialMap.b (0..10 world units). Used alongside real-time 3x3 PCF shadows.
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
  float inUnitU = step(-0.002, uv.x) * step(uv.x, 1.002);
  float inUnitV = step(-0.002, uv.y) * step(uv.y, 1.002);
  float inUnitUv = inUnitU * inUnitV;

  float edgeY = min(uv.y, 1.0 - uv.y);
  float maskY = (1.0 - smoothstep(0.0, max(fwUv.y * 1.45, 0.0045), edgeY)) * inUnitUv;
  float signY = uv.y > 0.5 ? 1.0 : -1.0;

  float edgeX = min(uv.x, 1.0 - uv.x);
  float uvWrapGuard = 1.0 - smoothstep(0.08, 0.25, fwUv.x);
  float maskX = (1.0 - smoothstep(0.0, max(fwUv.x * 1.45, 0.0045), edgeX)) * allowSideEdges * uvWrapGuard * inUnitUv;
  float signX = uv.x > 0.5 ? 1.0 : -1.0;

  vec3 tilt = T * (signX * maskX) + B * (signY * maskY);
  float chamferMask = clamp(max(maskX, maskY), 0.0, 1.0);
  vec3 chamferN = normalize(N + tilt * bevelStrength * chamferMask);
  return vec4(chamferN, chamferMask);
}

/**
 * Razor-thin platinum edge rim glint ONLY along sharp vertical corners,
 * stepped setbacks, and rooftop parapet edges — zero broad diffuse bleed across flat walls.
 */
vec3 computePlatinumEdgeGlint(
  vec3 baseN,
  vec3 chamferN,
  vec3 V,
  vec3 keyL,
  float chamferMask,
  float rimInt,
  float keyInt
) {
  float normalCrease = clamp(smoothstep(0.16, 0.72, length(fwidth(baseN))), 0.0, 1.0);
  float edgeSupport = clamp(max(chamferMask, normalCrease * 0.92), 0.0, 1.0);
  if (edgeSupport < 0.005) {
    return vec3(0.0);
  }

  vec3 edgeN = normalize(mix(baseN, chamferN, chamferMask));
  float ndv = clamp(dot(edgeN, V), 0.0, 1.0);
  float steepFresnel = pow(1.0 - ndv, 4.4);

  vec3 rightRimL = normalize(vec3(0.88, 0.44, -0.22));
  vec3 leftRimL = normalize(vec3(-0.82, 0.46, 0.34));
  vec3 zenithRimL = normalize(vec3(0.22, 0.94, -0.26));

  vec3 H_right = normalize(rightRimL + V);
  vec3 H_left = normalize(leftRimL + V);
  vec3 H_zen = normalize(zenithRimL + V);
  vec3 H_key = normalize(keyL + V);

  float specRight = pow(max(dot(edgeN, H_right), 0.0), 26.0);
  float specLeft = pow(max(dot(edgeN, H_left), 0.0), 28.0);
  float specZen = pow(max(dot(edgeN, H_zen), 0.0), 24.0);
  float specKey = pow(max(dot(edgeN, H_key), 0.0), 28.0);

  float rightSideWeight = smoothstep(-0.30, 0.60, edgeN.x) * 0.65 + 0.35;
  float roofEdgeBoost = smoothstep(0.35, 0.85, edgeN.y) * 0.45 + 0.55;

  float glint = edgeSupport * (
    steepFresnel * (0.42 + 0.78 * rightSideWeight) * rimInt +
    (specRight * 1.45 + specLeft * 0.85 + specZen * 0.95) * rimInt * roofEdgeBoost +
    specKey * 0.55 * keyInt
  );

  return STUDIO_PLATINUM_RIM * glint;
}

/**
 * Whisper-subtle cool obsidian-slate atmospheric haze inside lowest street corridors.
 */
vec3 applyCanyonAtmosphere(vec3 color, vec3 worldPos, vec3 camPos, float canyonAo) {
  float lowCorridor = smoothstep(2.35, 0.10, worldPos.y);
  float deepCanyon = smoothstep(0.84, 0.30, canyonAo);
  float depthFactor = smoothstep(28.0, 52.0, length(camPos - worldPos));
  float mistAmount = lowCorridor * deepCanyon * depthFactor * 0.28;
  vec3 canyonMistColor = vec3(0.0082, 0.0091, 0.0116);
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

export function createShadowDepthMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    side: THREE.FrontSide,
    vertexShader: /* glsl */ `
      out float vOrthoDepth;

      void main() {
        vec4 localPosition = vec4(position, 1.0);
        #ifdef USE_INSTANCING
          localPosition = instanceMatrix * localPosition;
        #endif
        vec4 clipPos = projectionMatrix * viewMatrix * modelMatrix * localPosition;
        vOrthoDepth = clipPos.z * 0.5 + 0.5;
        gl_Position = clipPos;
      }
    `,
    fragmentShader: /* glsl */ `
      in float vOrthoDepth;
      out vec4 fragColor;

      void main() {
        float c = clamp(vOrthoDepth, 0.0, 1.0) * 255.0;
        float hi = floor(c);
        float lo = fract(c);
        fragColor = vec4(hi / 255.0, lo, 0.0, 1.0);
      }
    `,
  });
}

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
  const boatScales = [1, 1, 1, 1, 1, 1];
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
      uShadowMap: { value: null },
      uShadowMatrix: { value: new THREE.Matrix4() },
      uShadowTexelSize: { value: new THREE.Vector2(1 / 1024, 1 / 1024) },
      uHasShadowMap: { value: 0.0 },
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
      uBoatScales: { value: boatScales },
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
      uniform sampler2D uShadowMap;
      uniform mat4 uShadowMatrix;
      uniform vec2 uShadowTexelSize;
      uniform float uHasShadowMap;
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
      uniform float uBoatScales[6];
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
        float rimRadius = uBasinRadius + 0.145;
        float distFromCenter = length(p);
        if (distFromCenter > rimRadius) {
          discard;
        }

        float swZ = seawallZ(p.x);
        float shoreDist = p.y - swZ;
        if (shoreDist < -0.18) {
          discard;
        }

        // 1. Fine high-frequency directional capillary wave normals (sheltered near seawall, richer out in the bay)
        float openBayFactor = mix(0.68, 1.38, smoothstep(0.0, 4.8, max(0.0, shoreDist)));
        vec2 uvA = p * 0.62 + vec2(uTime * 0.018, uTime * 0.012);
        vec2 uvB = p * 1.38 - vec2(uTime * 0.024, -uTime * 0.016);
        vec2 nA = texture(uWaterRippleMap, uvA).xy * 2.0 - 1.0;
        vec2 nB = texture(uWaterRippleMap, uvB).xy * 2.0 - 1.0;

        // Directional micro-capillary wind wavelets (finer horizontal bands across the harbor)
        float capX = cos(p.x * 6.4 + p.y * 2.3 + uTime * 1.25) * 0.011
                   + cos(p.x * 13.5 - p.y * 4.8 - uTime * 1.6) * 0.008;
        float capZ = sin(p.y * 11.2 + p.x * 1.9 + uTime * 1.15) * 0.018
                   + sin(p.y * 21.5 - p.x * 3.4 + uTime * 1.55) * 0.012;

        // 2. Miniature sailboat hull contact ripples & crisp V-shaped Kelvin wake etched purely into water normals
        vec2 wakeGrad = vec2(0.0);
        float boatContactAo = 1.0;
        for (int i = 0; i < 6; i++) {
          vec4 b = uBoats[i];
          vec2 dBoat = p - b.xy;
          if (dot(dBoat, dBoat) < 4.5) {
            float sc = max(0.65, uBoatScales[i]);
            vec2 fwd = normalize(b.zw);
            vec2 right = vec2(-fwd.y, fwd.x);
            float fwdCoord = dot(dBoat, fwd);   // + along bow
            float back = -fwdCoord;             // + behind stern
            float side = dot(dBoat, right);

            // Tight elliptical hull contact shadow & waterline meniscus hugging the hull
            float hullNx = side / (0.062 * sc);
            float hullNz = (fwdCoord + 0.02 * sc) / (0.185 * sc);
            float hullEllip = sqrt(hullNx * hullNx + hullNz * hullNz);
            boatContactAo = min(boatContactAo, mix(0.32, 1.0, smoothstep(0.75, 1.35, hullEllip)));

            if (hullEllip > 0.72 && hullEllip < 3.6) {
              float hullRing = max(0.0, hullEllip - 0.90);
              float contactWave = cos(hullRing * 62.0 - uTime * 2.6) * exp(-hullRing * 2.05);
              vec2 radialHull = normalize(right * hullNx + fwd * hullNz + vec2(1e-5));
              wakeGrad += radialHull * contactWave * 0.115;
            }

            // Crisp Kelvin 19.47° half-angle V-wake & divergent wavelets catching specular studio light
            float bowBack = back + 0.14 * sc;
            if (bowBack > 0.0 && bowBack < 1.95 * sc) {
              float armOffset = 0.022 * sc + bowBack * 0.354;
              float armDist = abs(side) - armOffset;
              float env = exp(-bowBack / (0.86 * sc)) * smoothstep(0.0, 0.12 * sc, bowBack);
              float armBand = exp(-pow(armDist / (0.015 * sc + bowBack * 0.010), 2.0)) * env;

              // Crisp divergent echelon wavelets angled along the Kelvin V-arms
              float echelonPhase = abs(side) * 74.0 - bowBack * 28.0 - uTime * 2.8;
              float crestWave = cos(echelonPhase) * armBand;
              vec2 armNormalDir = normalize(right * (side >= 0.0 ? 1.0 : -1.0) - fwd * 0.354);
              wakeGrad += armNormalDir * crestWave * 0.255;

              // Subtle secondary feather V-ridge inside main Kelvin cusp
              float innerDist = abs(side) - armOffset * 0.72;
              float innerBand = exp(-pow(innerDist / (0.013 * sc + bowBack * 0.009), 2.0)) * env * 0.55;
              wakeGrad += armNormalDir * cos(echelonPhase * 1.18 + 1.2) * innerBand * 0.12;

              // Crisp transverse stern wavelets strictly inside the V-wedge
              if (back > 0.08 * sc && abs(side) < armOffset * 0.88) {
                float transEnv = exp(-(back - 0.08 * sc) / (0.58 * sc))
                  * (1.0 - smoothstep(0.55, 0.90, abs(side) / max(0.01, armOffset)));
                float transSlope = cos(back * (50.0 / sc) - uTime * 2.9) * transEnv;
                wakeGrad += fwd * transSlope * 0.092;
              }
            }
          }
        }

        // Subtle circular basin meniscus curvature right along the inner rim wall (surface tension)
        float rimDist = max(0.0, rimRadius - distFromCenter);
        float meniscusMask = exp(-pow(rimDist / 0.14, 2.0));
        vec2 radialOut = distFromCenter > 0.001 ? p / distFromCenter : vec2(0.0, 1.0);
        vec2 meniscusGrad = -radialOut * meniscusMask * 0.12;

        vec2 rippleN = vec2(
          (nA.x * 0.036 + nB.x * 0.026 + capX) * openBayFactor,
          (nA.y * 0.044 + nB.y * 0.032 + capZ) * openBayFactor
        );

        vec3 N = normalize(vec3(
          rippleN.x + wakeGrad.x + meniscusGrad.x,
          1.0,
          rippleN.y + wakeGrad.y + meniscusGrad.y
        ));

        vec3 V = normalize(cameraPosition - vWorldPos);
        vec3 V_spec = applySpecularParallax(V, uTiltVec, uVelocity, 0.85);
        vec3 keyL = normalize(uSoftboxKeyDir);
        float NdotV = clamp(dot(N, V_spec), 0.0, 1.0);

        // Schlick Fresnel for dark harbor water (F0 = 0.028)
        float fresnel = fresnelSchlick(NdotV, 0.028);

        float waterShadow = min(
          sampleOrthographicShadowPCF(
            uShadowMap,
            uShadowMatrix,
            uShadowTexelSize,
            vWorldPos,
            vec3(0.0, 1.0, 0.0),
            keyL,
            uHasShadowMap
          ),
          mix(
            0.55,
            1.0,
            sampleDirectionalHeightShadow(uGroundAoMap, vWorldPos, vec3(0.0, 1.0, 0.0), keyL, uBasinRadius)
          )
        );

        // 3. Real-time Planar Reflection Pass with progressive capillary breakup & silhouette softening
        vec2 reflNdc = vReflClipPos.xy / max(0.0001, vReflClipPos.w);
        vec2 reflUv = reflNdc * 0.5 + 0.5;
        float reflDepthFactor = mix(0.45, 2.35, smoothstep(0.0, 5.5, max(0.0, shoreDist)));
        vec2 rippleDistort = (rippleN + wakeGrad * 0.45) * vec2(0.085, 0.185) * reflDepthFactor;
        vec2 sampleUv = clamp(reflUv + rippleDistort, vec2(0.002), vec2(0.998));

        // Subtle capillary-spread 3-tap reflection softening stronger further down the reflection
        float blurSpread = smoothstep(0.4, 5.5, max(0.0, shoreDist));
        vec2 jitterStep = vec2(0.0022, 0.0046) * (0.25 + 1.35 * blurSpread);
        vec3 reflCenter = texture(uReflectMap, sampleUv).rgb;
        vec3 reflTapA = texture(uReflectMap, clamp(sampleUv + vec2(jitterStep.x, jitterStep.y) + rippleDistort * 0.25, vec2(0.002), vec2(0.998))).rgb;
        vec3 reflTapB = texture(uReflectMap, clamp(sampleUv - vec2(jitterStep.x, jitterStep.y * 0.85) - rippleDistort * 0.20, vec2(0.002), vec2(0.998))).rgb;
        vec3 planarRefl = mix(reflCenter, reflCenter * 0.50 + (reflTapA + reflTapB) * 0.25, blurSpread * 0.88);

        // 4. Studio Cyclorama & Softbox Fresnel Reflections (#090B0E to #181E26) + Capillary & Kelvin Glints
        vec3 R = reflect(-V_spec, N);
        float waterRoughness = mix(0.048, 0.088, blurSpread);
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
        vec3 H_overhead = normalize(normalize(vec3(-0.22, 0.88, 0.42)) + V_spec);
        vec3 H_rim = normalize(normalize(vec3(-0.78, 0.42, 0.46)) + V_spec);

        // Ambient studio cyclorama dome reflection (sRGB #090B0E -> #181E26 linear equivalent)
        float grazingFactor = pow(1.0 - NdotV, 2.2);
        vec3 cycloramaDeep = vec3(0.0028, 0.0034, 0.0044);     // ~#090B0E sRGB
        vec3 cycloramaHorizon = vec3(0.0091, 0.0131, 0.0196);  // ~#181E26 sRGB
        vec3 cycloramaRefl = mix(cycloramaDeep, cycloramaHorizon, clamp(0.18 + 0.82 * grazingFactor + R.y * 0.25, 0.0, 1.0));

        // Broad overhead softbox sheen across the dark liquid surface (reveals capillary surface tension)
        float softboxSheen = (pow(max(dot(N, H_overhead), 0.0), 14.0) * 0.026
                            + pow(max(dot(N, H), 0.0), 28.0) * 0.020)
                           * (0.35 + 0.65 * fresnel) * uKeyIntensity * mix(0.45, 1.0, waterShadow);

        // Delicate micro-specular glints caught along capillary wave crests & steep Kelvin wakes
        float capillaryCrest = clamp(length(rippleN) * 14.0, 0.0, 1.0);
        float capillaryMicroGlint = pow(max(dot(N, H_overhead), 0.0), 72.0) * capillaryCrest * 0.042 * uKeyIntensity
                                  + pow(max(dot(N, H_rim), 0.0), 110.0) * capillaryCrest * 0.032 * uRimIntensity;
        float wakeStrength = clamp(length(wakeGrad) * 6.5, 0.0, 1.0);
        float sharpKeyGlint = pow(max(dot(N, H), 0.0), 150.0) * uKeyIntensity * waterShadow * (0.28 + 1.55 * wakeStrength);
        float wakeSoftGlint = pow(max(dot(N, H_overhead), 0.0), 42.0) * wakeStrength * 0.095 * uKeyIntensity * waterShadow;
        float meniscusGlint = pow(max(dot(N, H), 0.0), 48.0) * meniscusMask * 0.105 * uKeyIntensity;

        // 5. Deep dark smoked slate-harbor water volume (never collapses to dead #000000 black)
        float bayDepth = smoothstep(0.0, 2.4, max(0.0, shoreDist));
        vec3 beerAbsorption = exp(-vec3(0.92, 0.80, 0.68) * bayDepth);
        vec3 shallowSlate = vec3(0.0052, 0.0060, 0.0074) * beerAbsorption;
        vec3 deepHarbor = vec3(0.0029, 0.0035, 0.0046);
        vec3 waterVolume = mix(shallowSlate, deepHarbor, 0.58) * (0.72 + 0.28 * uFillIntensity);

        // 6. View-aligned vertical reflection shimmer trails directly below each esplanade lamppost
        float lampGlints = 0.0;
        if (shoreDist > -0.04 && shoreDist < 1.75) {
          float pressBoost = 1.0 + uPressDepth * 0.30;
          float waterMask = smoothstep(-0.02, 0.07, shoreDist);
          for (int i = 0; i < 32; i++) {
            if (i >= uLampCount) break;
            vec2 lp = uLamps[i];
            vec2 dLamp = p - lp;
            if (dot(dLamp, dLamp) < 2.25) {
              vec2 toCam = cameraPosition.xz - lp;
              float camLen = max(0.001, length(toCam));
              vec2 viewAxis = toCam / camLen;
              vec2 perpAxis = vec2(-viewAxis.y, viewAxis.x);

              // Project onto camera-vertical column directly below each lamp post
              float along = dot(dLamp, viewAxis);
              float across = dot(dLamp, perpAxis);

              if (along > 0.01 && along < 1.25 && abs(across) < 0.14) {
                float rippleHorizShift = dot(N.xz, perpAxis) * (0.045 + along * 0.035);
                float sigmaAcross = 0.018 + along * 0.012;
                float colMask = exp(-pow((across + rippleHorizShift) / sigmaAcross, 2.0));
                float vertEnvelope = smoothstep(0.0, 0.07, along) * exp(-along * 2.35);

                // Break up the vertical trail naturally into horizontal water-ripple sparkles
                float slopeAlong = dot(N.xz, viewAxis);
                float waveSpark = smoothstep(-0.014, 0.036, slopeAlong + capZ * 0.45);
                float horizBands = 0.25 + 0.75 * smoothstep(
                  -0.15,
                  0.65,
                  sin(along * 48.0 + nB.y * 4.8 + nA.x * 2.2 - uTime * 1.9)
                );
                lampGlints += colMask * vertEnvelope * waveSpark * horizBands * 0.32 * pressBoost * waterMask;
              }
            }
          }
        }

        float shoreCreviceAo = mix(0.24 - uPressDepth * 0.12, 1.0, pow(smoothstep(-0.04, 0.36, shoreDist), 1.0 + uPressDepth * 0.55));
        float basinEdgeFade = smoothstep(0.0, 0.045, rimDist);

        // Combine deep harbor volume + ambient cyclorama Fresnel + planar skyline reflection + crisp micro-glints
        float reflMask = (0.06 + 0.94 * fresnel) * uReflectivity * mix(0.44, 1.0, waterShadow) * boatContactAo;
        vec3 studioGloss = iblStudio * (0.08 + 0.92 * fresnel) * 0.030 * uReflectivity * mix(0.45, 1.0, waterShadow);
        vec3 ambientCyclorama = cycloramaRefl * (0.38 + 0.62 * fresnel) * shoreCreviceAo * mix(0.62, 1.0, waterShadow);

        vec3 finalColor = waterVolume * shoreCreviceAo * boatContactAo * mix(0.55, 1.0, waterShadow)
          + ambientCyclorama
          + planarRefl * reflMask * 1.24
          + studioGloss
          + vec3(0.88, 0.92, 0.97) * (softboxSheen + capillaryMicroGlint + wakeSoftGlint + sharpKeyGlint + meniscusGlint)
          + STUDIO_WARM_POOL_3000K * (lampGlints * uReflectivity * 1.15);

        fragColor = vec4(finalColor, basinEdgeFade * 0.985);
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
  seawallLampXZ: Array<{ x: number; z: number }> = [],
  heroLobbyZones: Array<{ x: number; z: number; radius: number; strength: number }> = [],
): THREE.ShaderMaterial {
  const paddedLamps: THREE.Vector2[] = [];
  for (let i = 0; i < 32; i++) {
    const l = seawallLampXZ[i];
    paddedLamps.push(l ? new THREE.Vector2(l.x, l.z) : new THREE.Vector2(999, 999));
  }
  const paddedLobbies: THREE.Vector4[] = [];
  for (let i = 0; i < 12; i++) {
    const h = heroLobbyZones[i];
    paddedLobbies.push(h ? new THREE.Vector4(h.x, h.z, h.radius, h.strength) : new THREE.Vector4(999, 999, 0.1, 0));
  }

  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      uGrainMap: { value: grainTexture },
      uGroundAoMap: { value: groundAoMap },
      uMicroMap: { value: microImperfectionMap },
      uStudioIblMap: { value: studioIblMap },
      uShadowMap: { value: null },
      uShadowMatrix: { value: new THREE.Matrix4() },
      uShadowTexelSize: { value: new THREE.Vector2(1 / 1024, 1 / 1024) },
      uHasShadowMap: { value: 0.0 },
      uBasinRadius: { value: basinRadius },
      uSoftboxKeyDir: { value: new THREE.Vector3(-0.44, 0.68, 0.58) },
      uKeyIntensity: { value: 1.26 },
      uRimIntensity: { value: 0.92 },
      uFillIntensity: { value: 0.40 },
      uPressDepth: { value: 0.0 },
      uTiltVec: { value: new THREE.Vector2(0, 0) },
      uVelocity: { value: new THREE.Vector2(0, 0) },
      uLamps: { value: paddedLamps },
      uLampCount: { value: Math.min(32, seawallLampXZ.length) },
      uHeroLobbies: { value: paddedLobbies },
      uHeroLobbyCount: { value: Math.min(12, heroLobbyZones.length) },
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
      uniform sampler2D uShadowMap;
      uniform mat4 uShadowMatrix;
      uniform vec2 uShadowTexelSize;
      uniform float uHasShadowMap;
      uniform float uBasinRadius;
      uniform vec3 uSoftboxKeyDir;
      uniform float uKeyIntensity;
      uniform float uRimIntensity;
      uniform float uFillIntensity;
      uniform float uPressDepth;
      uniform vec2 uTiltVec;
      uniform vec2 uVelocity;
      uniform vec2 uLamps[32];
      uniform int uLampCount;
      uniform vec4 uHeroLobbies[12];
      uniform int uHeroLobbyCount;

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

        // Real-time 3x3 PCF Orthographic Shadow Map + fallback heightfield shadow across streets & plazas
        float pcfShadow = sampleOrthographicShadowPCF(
          uShadowMap,
          uShadowMatrix,
          uShadowTexelSize,
          vWorldPos,
          baseN,
          keyL,
          uHasShadowMap
        );
        float heightShadow = sampleDirectionalHeightShadow(
          uGroundAoMap,
          vWorldPos,
          baseN,
          keyL,
          uBasinRadius
        );
        float castShadow = uHasShadowMap > 0.5 ? pcfShadow : heightShadow;

        // Rotated San Diego downtown street grid, cleanly masked before seawall coping & basin rim
        float c = cos(0.22);
        float s = sin(0.22);
        vec2 streetUv = vec2(
          vWorldPos.x * c - vWorldPos.z * s,
          vWorldPos.x * s + vWorldPos.z * c
        );
        float seawallCurveZ = 2.6 + 2.35 * cos(vWorldPos.x * 0.24 + 0.35) - 0.14 * vWorldPos.x;
        float seawallMarginMask = smoothstep(0.22, 0.56, seawallCurveZ - vWorldPos.z);
        float perimeterMask = smoothstep(uBasinRadius - 0.28, uBasinRadius - 0.62, length(vWorldPos.xz));
        float streetGrid = max(
          sdfLine(streetUv.x, 0.606, 0.024),
          sdfLine(streetUv.y, 0.606, 0.024)
        ) * smoothstep(0.5, 0.95, baseN.y) * seawallMarginMask * perimeterMask;

        // 3:1 key-to-fill dead-matte diffuse with subtle warm key / cool obsidian fill temperature
        vec3 keyTerm = STUDIO_KEY_WARM * max(dot(N, keyL), 0.0) * castShadow * uKeyIntensity * 1.04;
        vec3 fillTerm = STUDIO_FILL_COOL * (0.14 * uFillIntensity + wrapDiffuse(N, fillL, 0.35) * uFillIntensity * 0.34);
        vec3 studioBounce = sampleStudioIBL(
          uStudioIblMap,
          N,
          0.88,
          keyL,
          uKeyIntensity,
          uRimIntensity,
          uFillIntensity
        );

        vec3 litBalance = keyTerm + fillTerm + studioBounce * 0.075;
        vec3 slateBase =
          vec3(0.0086, 0.0089, 0.0101) *
          (0.90 + 0.20 * grain) *
          litBalance *
          combinedAo;

        // Seawall beveled concrete curb catches neutral-warm key light
        float edgeWall = 1.0 - smoothstep(0.22, 0.80, abs(baseN.y));
        float curbKey = max(dot(N, keyL), 0.0) * castShadow * uKeyIntensity;
        vec3 seawallConcrete =
          vec3(0.015, 0.016, 0.018) *
          edgeWall *
          (STUDIO_FILL_COOL * 0.24 * uFillIntensity + STUDIO_KEY_WARM * curbKey * 1.15) *
          mix(0.45, 1.0, aoSample.g);

        // Subtle localized warm champagne-amber pools (2900K-3000K) beneath curved waterfront esplanade lamps
        float lampPoolAccum = 0.0;
        for (int i = 0; i < 32; i++) {
          if (i >= uLampCount) break;
          vec2 dLamp = vWorldPos.xz - uLamps[i];
          float d2 = dot(dLamp, dLamp);
          if (d2 < 0.36) {
            lampPoolAccum += exp(-d2 / (0.18 * 0.18));
          }
        }
        float lampPoolMask = smoothstep(-0.18, 0.08, seawallCurveZ - vWorldPos.z) * smoothstep(0.35, 0.90, baseN.y);
        float esplanadeWarmPool = min(1.15, lampPoolAccum) * lampPoolMask * (0.48 + uPressDepth * 0.20);

        // Soft warm champagne glow spilling gently from ground-floor hero tower lobbies & colonnades
        float lobbySpillAccum = 0.0;
        for (int i = 0; i < 12; i++) {
          if (i >= uHeroLobbyCount) break;
          vec4 hb = uHeroLobbies[i];
          vec2 dLobby = vWorldPos.xz - hb.xy;
          if (dot(dLobby, dLobby) < 4.5) {
            float distEdge = max(0.0, length(dLobby) - hb.z);
            lobbySpillAccum += exp(-distEdge * 3.1) * smoothstep(1.20, 0.0, distEdge) * hb.w;
          }
        }
        float lobbyWarmSpill = min(1.0, lobbySpillAccum) * smoothstep(0.40, 0.95, baseN.y) * mix(0.28, 1.0, tightCreviceAo);

        vec3 streetHighlight =
          vec3(0.016, 0.017, 0.020) *
          streetGrid *
          mix(STUDIO_FILL_COOL * 0.25, STUDIO_KEY_WARM, castShadow) *
          combinedAo;

        vec3 finalRgb = slateBase
          + streetHighlight
          + seawallConcrete
          + STUDIO_WARM_POOL_3000K * (esplanadeWarmPool * 0.095 + lobbyWarmSpill * 0.088) * mix(0.45, 1.0, aoSample.g);
        finalRgb = applyCanyonAtmosphere(finalRgb, vWorldPos, cameraPosition, aoSample.g);
        fragColor = vec4(finalRgb, 1.0);
      }
    `,
  });
}

export interface MatteSlateOptions {
  baseColor?: number;
  masonryWindows?: boolean;
  seawallLampXZ?: Array<{ x: number; z: number }>;
}

/**
 * Dead-Matte Anodized Dark Slate / Architectural Limestone Material (roughness ~0.88).
 * Used for building podiums, parapets, mechanical penthouses, and (with masonryWindows: true)
 * punched architectural masonry/limestone mid-rise blocks that contrast against glossy glass towers.
 */
export function createMatteSlateMaterial(
  grainTexture: THREE.Texture,
  groundAoMap: THREE.Texture,
  microImperfectionMap: THREE.Texture,
  studioIblMap: THREE.Texture,
  basinRadius: number,
  optionsOrColor: number | MatteSlateOptions = 0x1e2025,
): THREE.ShaderMaterial {
  const opts: MatteSlateOptions =
    typeof optionsOrColor === 'number' ? { baseColor: optionsOrColor } : optionsOrColor;
  const baseColor = opts.baseColor ?? 0x1e2025;
  const masonryWindows = opts.masonryWindows ?? false;
  const seawallLampXZ = opts.seawallLampXZ ?? [];
  const paddedLamps: THREE.Vector2[] = [];
  for (let i = 0; i < 32; i++) {
    const l = seawallLampXZ[i];
    paddedLamps.push(l ? new THREE.Vector2(l.x, l.z) : new THREE.Vector2(999, 999));
  }

  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      uGrainMap: { value: grainTexture },
      uGroundAoMap: { value: groundAoMap },
      uMicroMap: { value: microImperfectionMap },
      uStudioIblMap: { value: studioIblMap },
      uShadowMap: { value: null },
      uShadowMatrix: { value: new THREE.Matrix4() },
      uShadowTexelSize: { value: new THREE.Vector2(1 / 1024, 1 / 1024) },
      uHasShadowMap: { value: 0.0 },
      uBaseColor: { value: new THREE.Color(baseColor) },
      uMasonryWindows: { value: masonryWindows ? 1.0 : 0.0 },
      uBasinRadius: { value: basinRadius },
      uSoftboxKeyDir: { value: new THREE.Vector3(-0.44, 0.68, 0.58) },
      uKeyIntensity: { value: 1.26 },
      uRimIntensity: { value: 0.92 },
      uFillIntensity: { value: 0.40 },
      uPressDepth: { value: 0.0 },
      uTiltVec: { value: new THREE.Vector2(0, 0) },
      uVelocity: { value: new THREE.Vector2(0, 0) },
      uIsReflectPass: { value: 0.0 },
      uLamps: { value: paddedLamps },
      uLampCount: { value: Math.min(32, seawallLampXZ.length) },
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
      uniform sampler2D uShadowMap;
      uniform mat4 uShadowMatrix;
      uniform vec2 uShadowTexelSize;
      uniform float uHasShadowMap;
      uniform vec3 uBaseColor;
      uniform float uMasonryWindows;
      uniform float uBasinRadius;
      uniform vec3 uSoftboxKeyDir;
      uniform float uKeyIntensity;
      uniform float uRimIntensity;
      uniform float uFillIntensity;
      uniform float uPressDepth;
      uniform vec2 uTiltVec;
      uniform vec2 uVelocity;
      uniform float uIsReflectPass;
      uniform vec2 uLamps[32];
      uniform int uLampCount;

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

        float pcfShadow = sampleOrthographicShadowPCF(
          uShadowMap,
          uShadowMatrix,
          uShadowTexelSize,
          vWorldPos,
          baseN,
          keyL,
          uHasShadowMap
        );
        float heightShadow = sampleDirectionalHeightShadow(
          uGroundAoMap,
          vWorldPos,
          baseN,
          keyL,
          uBasinRadius
        );
        float castShadow = uHasShadowMap > 0.5 ? pcfShadow : heightShadow;

        vec2 spatialAo = sampleUrbanSpatialAo(uGroundAoMap, vWorldPos.xz, uBasinRadius);
        float baseAo = mix(0.42 - uPressDepth * 0.14, 1.0, smoothstep(0.08, 0.48, vWorldPos.y));
        float canyonAo = mix(0.36, 1.0, spatialAo.g);

        // Optional punched architectural limestone/slate masonry bays on mid-rise blocks
        float isRoof = smoothstep(0.65, 0.90, abs(baseN.y));
        float masonryRecess = 0.0;
        float courseLine = 0.0;
        if (uMasonryWindows > 0.5 && isRoof < 0.5) {
          vec2 horizNormal = length(baseN.xz) > 0.01 ? normalize(baseN.xz) : vec2(0.0, 1.0);
          vec3 wallT = vec3(horizNormal.y, 0.0, -horizNormal.x);
          float wallX = dot(vWorldPos.xz, wallT.xz);
          float bayU = fract(wallX * 3.1);
          float bayV = fract(vWorldPos.y * 3.2);
          float winX = smoothstep(0.16, 0.24, bayU) * (1.0 - smoothstep(0.76, 0.84, bayU));
          float winY = smoothstep(0.24, 0.32, bayV) * (1.0 - smoothstep(0.80, 0.88, bayV));
          masonryRecess = winX * winY;
          float lintelAO = smoothstep(0.72, 0.88, bayV) * winX;
          masonryRecess = max(masonryRecess * 0.85, lintelAO);
          // Subtle horizontal ashlar stone course seam
          float cDist = abs(bayV - 0.04);
          courseLine = 1.0 - smoothstep(0.0, 0.045, cDist);
        }

        // 3:1 directional key vs cool obsidian fill (dead-matte roughness ~0.88)
        vec3 keyTerm = STUDIO_KEY_WARM * max(dot(N, keyL), 0.0) * castShadow * uKeyIntensity * 0.96;
        vec3 fillTerm = STUDIO_FILL_COOL * (0.15 * uFillIntensity + wrapDiffuse(N, fillL, 0.35) * uFillIntensity * 0.36);
        vec3 studioDiff = sampleStudioIBL(
          uStudioIblMap,
          N,
          0.88,
          keyL,
          uKeyIntensity,
          uRimIntensity,
          uFillIntensity
        );

        vec3 lit = keyTerm + fillTerm + studioDiff * 0.08;
        vec3 albedo = uBaseColor * (0.86 + 0.26 * grain);
        if (uMasonryWindows > 0.5) {
          albedo = mix(albedo, albedo * 0.26, masonryRecess);
          albedo *= 1.0 - courseLine * 0.24;
        }
        vec3 color = albedo * lit * baseAo * canyonAo;

        // Subtle warm-neutral matte bevel catchlight strictly on faces facing key light
        float faceKeyVis = smoothstep(-0.08, 0.25, dot(baseN, keyL));
        float bevelKey = pow(max(dot(chamferRes.xyz, keyL), 0.0), 1.8) * castShadow * faceKeyVis * chamferRes.w;
        color += STUDIO_KEY_WARM * vec3(0.016, 0.017, 0.019) * bevelKey * uKeyIntensity;

        // Razor-thin platinum edge rim glint along sharp vertical corners, setbacks & parapet lips
        vec3 V_slate = normalize(cameraPosition - vWorldPos);
        vec3 platinumRim = computePlatinumEdgeGlint(
          baseN,
          chamferRes.xyz,
          V_slate,
          keyL,
          chamferRes.w,
          uRimIntensity,
          uKeyIntensity
        );
        color += platinumRim * 0.26 * baseAo;

        // Faint 2950K warm champagne pools cast directly onto seawall coping promenade beneath lamps
        if (vWorldPos.y < 0.24 && uLampCount > 0) {
          float copingPool = 0.0;
          for (int i = 0; i < 32; i++) {
            if (i >= uLampCount) break;
            vec2 dLamp = vWorldPos.xz - uLamps[i];
            float d2 = dot(dLamp, dLamp);
            if (d2 < 0.32) {
              copingPool += exp(-d2 / (0.16 * 0.16));
            }
          }
          color += STUDIO_WARM_POOL_3000K * min(1.0, copingPool) * 0.052 * smoothstep(0.10, 0.90, baseN.y);
        }

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
  interiorMode?: 'tower' | 'lobby' | 'terminal-arch';
}

/**
 * WebGL2 GLSL3 Architectural Curtain-Wall Glass Shader (Museum Scale-Model Realism):
 * - Smoked dark high-IOR architectural glass with crisp horizontal spandrel bands & vertical mullions
 * - Clustered contiguous office suites (3–8 adjacent bays), occasional wrapped corner offices,
 *   warm ground-floor hero lobbies, and soft vaulted terminal bays
 * - True parallax depth behind dark facade mullions/spandrels with ceiling-to-floor luminance cascade
 *   and Fresnel glazing attenuation at oblique viewing angles
 * - Restrained photographic alabaster / champagne-ivory / sparse cool daylight interior palette
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
    interiorMode = 'tower',
  } = options;
  const interiorModeCode = interiorMode === 'lobby' ? 1.0 : interiorMode === 'terminal-arch' ? 2.0 : 0.0;

  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      uGrainMap: { value: grainTexture },
      uStudioIblMap: { value: studioIblMap },
      uMicroMap: { value: microImperfectionMap },
      uGroundAoMap: { value: groundAoMap },
      uShadowMap: { value: null },
      uShadowMatrix: { value: new THREE.Matrix4() },
      uShadowTexelSize: { value: new THREE.Vector2(1 / 1024, 1 / 1024) },
      uHasShadowMap: { value: 0.0 },
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
      uInteriorMode: { value: interiorModeCode },
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
      out vec3 vBuildingOrigin;
      out vec2 vUv;

      void main() {
        vUv = uv;
        vLocalPos = position;

        vec4 localPosition = vec4(position, 1.0);
        vec4 originLocal = vec4(0.0, 0.0, 0.0, 1.0);
        vec3 localNormal = normal;
        #ifdef USE_INSTANCING
          localPosition = instanceMatrix * localPosition;
          originLocal = instanceMatrix * originLocal;
          localNormal = mat3(instanceMatrix) * localNormal;
        #endif

        vec4 worldPosition = modelMatrix * localPosition;
        vWorldPos = worldPosition.xyz;
        vBuildingOrigin = (modelMatrix * originLocal).xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * localNormal);

        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uGrainMap;
      uniform sampler2D uStudioIblMap;
      uniform sampler2D uMicroMap;
      uniform sampler2D uGroundAoMap;
      uniform sampler2D uShadowMap;
      uniform mat4 uShadowMatrix;
      uniform vec2 uShadowTexelSize;
      uniform float uHasShadowMap;
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
      uniform float uInteriorMode;
      uniform float uHighlightBoost;
      uniform float uPressDepth;
      uniform vec2 uTiltVec;
      uniform vec2 uVelocity;
      uniform float uIsReflectPass;

      in vec3 vWorldPos;
      in vec3 vWorldNormal;
      in vec3 vLocalPos;
      in vec3 vBuildingOrigin;
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

        // Real-time 3x3 PCF Orthographic Shadow Map + Urban Canyon AO
        float pcfShadow = sampleOrthographicShadowPCF(
          uShadowMap,
          uShadowMatrix,
          uShadowTexelSize,
          vWorldPos,
          baseN,
          keyL,
          uHasShadowMap
        );
        float heightShadow = sampleDirectionalHeightShadow(
          uGroundAoMap,
          vWorldPos,
          baseN,
          keyL,
          uBasinRadius
        );
        float castShadow = uHasShadowMap > 0.5 ? pcfShadow : heightShadow;

        vec2 spatialAo = sampleUrbanSpatialAo(uGroundAoMap, vWorldPos.xz, uBasinRadius);
        float groundCreviceAo = mix(0.30 - uPressDepth * 0.14, 1.0, pow(smoothstep(0.08, 0.58, vWorldPos.y), 1.0 + uPressDepth * 0.55));
        float tierCreviceAo = mix(0.60 - uPressDepth * 0.10, 1.0, pow(smoothstep(0.0, 0.14, vUv.y), 1.0 + uPressDepth * 0.45));
        float canyonAo = mix(0.45, 1.0, mix(spatialAo.g, 1.0, clamp(vWorldPos.y / 3.8, 0.0, 1.0)));
        float contactAo = groundCreviceAo * tierCreviceAo * canyonAo;

        // Exact wall-aligned horizontal tangent so panes & interior parallax never skew on angled facades
        vec2 horizNormal = length(baseN.xz) > 0.01 ? normalize(baseN.xz) : vec2(0.0, 1.0);
        vec3 wallT = vec3(horizNormal.y, 0.0, -horizNormal.x);
        float wallX = dot(vWorldPos.xz, wallT.xz);

        float mullionFreq = uMullionDensity * 0.44;
        float floorFreq = uFloorDensity * 0.42;

        // Crisp horizontal spandrel bands & prominent dark vertical mullions IN FRONT of lit interiors
        float floorScaled = vWorldPos.y * floorFreq;
        float floorPhase = fract(floorScaled);
        float fwFloor = max(fwidth(floorScaled), 0.002);
        float spandrelRatio = uInteriorMode > 1.5 ? 0.05 : (uInteriorMode > 0.5 ? 0.12 : 0.25);
        float mullionThick = uInteriorMode > 1.5 ? 0.036 : (uInteriorMode > 0.5 ? 0.050 : 0.048);
        float spandrelMask = (1.0 - smoothstep(spandrelRatio - fwFloor, spandrelRatio + fwFloor, floorPhase)) * (1.0 - isRoof);
        if (uInteriorMode > 1.5) {
          spandrelMask = 0.0;
        }
        float vMullion = sdfGridLine(wallX, mullionFreq, mullionThick) * (1.0 - isRoof);
        float frameMask = clamp(max(spandrelMask, vMullion), 0.0, 1.0);

        // Subtle per-pane normal micro-tilt (±0.022) inside glass vision bays
        vec2 paneCell = floor(vec2(wallX * mullionFreq + 0.5, floorScaled));
        float jx = (hash21(paneCell) - 0.5) * 0.042;
        float jy = (hash21(paneCell + vec2(31.7, 17.3)) - 0.5) * 0.042;
        vec3 paneN = normalize(N + (wallT * jx + vec3(0.0, 1.0, 0.0) * jy) * (1.0 - frameMask) * (1.0 - isRoof));

        float NdotV = clamp(dot(paneN, V), 0.0, 1.0);
        // High-IOR glossy architectural curtain-wall glass (F0 = 0.095)
        float fresnel = fresnelSchlick(NdotV, 0.095);

        // Crisp high-gloss outer glass (~0.036) vs matte dark anodized mullions/spandrels (~0.68)
        float glassRoughness = clamp(
          mix(0.036 + (micro.r - 0.5) * 0.016, 0.68, frameMask),
          0.026,
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

        // Raking key-to-fill contrast across vertical tower faces (stronger cast shadow separation)
        float keyFace = max(dot(paneN, keyL), 0.0) * castShadow;
        float fillFace = wrapDiffuse(paneN, fillL, 0.22) * uFillIntensity * 0.20;
        vec3 keyDiffuseRgb = STUDIO_FILL_COOL * (0.10 * uFillIntensity + fillFace)
          + STUDIO_KEY_WARM * (keyFace * uKeyIntensity * 1.12);

        // Beer-Lambert smoked glass depth absorption
        float opticalThickness = 0.56 / max(0.24, NdotV);
        vec3 beerLambert = exp(-vec3(0.54, 0.49, 0.42) * opticalThickness);

        // Inner Depth Layer: GLSL Tangent-Space Interior Parallax (recessed suites behind crisp dark mullions)
        float bayU = fract(wallX * mullionFreq + 0.5);
        float bayV = clamp((floorPhase - spandrelRatio) / max(0.2, 1.0 - spandrelRatio), 0.001, 0.999);
        float vDotN = max(0.22, abs(dot(V_raw, baseN)));
        vec2 rayT = vec2(-dot(V_raw, wallT), -V_raw.y) / vDotN;

        float tSlab = (rayT.y > 0.0001)
          ? (1.0 - bayV) / rayT.y
          : ((rayT.y < -0.0001) ? bayV / (-rayT.y) : 99.0);
        float tCol = (rayT.x > 0.0001)
          ? (1.0 - bayU) / rayT.x
          : ((rayT.x < -0.0001) ? bayU / (-rayT.x) : 99.0);
        float tBack = 0.54;
        float tHit = min(tBack, min(tSlab, tCol));
        vec2 hitUv = clamp(vec2(bayU, bayV) + rayT * tHit, vec2(0.0), vec2(1.0));
        float hitSlab = step(tSlab, min(tCol, tBack));
        float hitCol = step(tCol, min(tSlab, tBack)) * (1.0 - hitSlab);

        // Deeper recessed shadow inside mullion reveals & upper/lower slab soffits
        float interiorBoxAo = (0.34 + 0.66 * smoothstep(0.0, 0.16, hitUv.y) * smoothstep(1.0, 0.78, hitUv.y))
          * (0.38 + 0.62 * smoothstep(0.0, 0.14, hitUv.x) * smoothstep(1.0, 0.86, hitUv.x));

        // 1. Clustered Architectural Suite Occupancy (no noisy checkerboard pixels)
        vec2 bKey = floor(vBuildingOrigin.xz * 2.5 + vec2(0.5));
        float bSeed = hash21(bKey);
        float floorIdx = paneCell.y;
        float floorRand = hash21(vec2(floorIdx, bSeed * 97.3 + 13.7));

        // Group bays into macro clusters of 10 bays and ignite contiguous suites of 3–8 adjacent bays
        float macroIdx = floor(paneCell.x / 10.0);
        float localBay = paneCell.x - macroIdx * 10.0;
        float suiteSeed = hash21(vec2(macroIdx * 11.3 + bSeed * 41.0, floorIdx + 7.1));
        float suiteStart = floor(hash21(vec2(suiteSeed, floorIdx * 3.1)) * 4.0);       // 0..3
        float suiteLen = 3.0 + floor(hash21(vec2(floorIdx, suiteSeed * 19.7)) * 5.6); // 3..8 contiguous bays

        float floorActive = step(0.56, floorRand);
        float suiteActive = step(0.58, suiteSeed);
        float inSuiteRun = step(suiteStart, localBay) * step(localBay, suiteStart + suiteLen - 0.5);
        // Subtle 1-bay internal suite partition / blind variation (only ~7% dim break inside long runs)
        float partitionDim = mix(0.84, 1.0, hash21(vec2(paneCell.x, floorIdx * 5.3 + suiteSeed)));
        float blindMask = step(0.07, hash21(paneCell + vec2(17.9, 61.3)));
        float suiteMask = floorActive * suiteActive * inSuiteRun * partitionDim * blindMask;

        // Occasional lit corner office wrapping cleanly across adjacent perpendicular walls
        vec2 relXZ = vWorldPos.xz - vBuildingOrigin.xz;
        float relLen = length(relXZ);
        vec2 relDir = relLen > 0.02 ? (relXZ / relLen) : horizNormal;
        float cornerPick = floor(hash21(vec2(floorIdx * 7.3, bSeed * 53.0)) * 4.0);
        float cornerAng = (cornerPick * 0.5 + 0.25) * PI + 0.22;
        vec2 cornerDir = vec2(cos(cornerAng), sin(cornerAng));
        float cornerWindow = smoothstep(0.82, 0.90, dot(relDir, cornerDir)) * step(0.84, hash21(vec2(floorIdx + 3.7, bSeed * 29.0)));

        // Ground-floor warm lobby zone on hero towers & inset podium colonnade
        float heroBaseZone = smoothstep(0.95, 0.58, vWorldPos.y) * smoothstep(0.45, 0.88, uIsHero);
        float lobbyRun = step(0.28, hash21(vec2(floor(paneCell.x / 4.0), bSeed * 19.0)));

        float modeTower = clamp(max(suiteMask, cornerWindow), 0.0, 1.0);
        if (heroBaseZone > 0.01) {
          modeTower = max(modeTower, heroBaseZone * lobbyRun * 0.92);
        }

        // Dedicated inset Lobby Mode (hero podium colonnade) & Vaulted Terminal Arch Mode
        float modeLobby = (0.76 + 0.22 * hash21(vec2(floor(paneCell.x / 3.0), bSeed * 31.0)))
          * smoothstep(0.06, 0.36, vWorldPos.y);
        float archH = clamp((vWorldPos.y - 0.96) / 0.76, 0.0, 1.0);
        float archVaultGrad = smoothstep(0.0, 0.24, archH) * smoothstep(1.02, 0.28, archH);
        float modeTerminal = (0.74 + 0.24 * hash21(vec2(floor(vWorldPos.x * 1.1), floorIdx))) * archVaultGrad;

        float occupancy = uInteriorMode > 1.5
          ? modeTerminal
          : (uInteriorMode > 0.5 ? modeLobby : modeTower);

        // 2. True Recessed Parallax + Ceiling-to-Floor Brightness Cascade
        float backWallMask = (1.0 - hitSlab) * (1.0 - hitCol * 0.86);
        float interiorDepthShade = mix(0.24, 1.0, backWallMask) * interiorBoxAo;

        // Brighter near the interior ceiling (hitUv.y -> 0.88), softer downward toward floor/desk
        float ceilingCascade = mix(0.28, 1.0, smoothstep(0.06, 0.86, hitUv.y));
        float trofferHighlight = 1.0 + 0.32 * exp(-pow((hitUv.y - 0.82) / 0.14, 2.0));
        float sillSoffitClip = smoothstep(0.03, 0.14, hitUv.y) * smoothstep(0.98, 0.89, hitUv.y)
          * smoothstep(0.04, 0.16, hitUv.x) * smoothstep(0.96, 0.84, hitUv.x);
        float bayGlowProfile = uInteriorMode > 1.5
          ? mix(0.70, 1.0, ceilingCascade) * smoothstep(0.02, 0.12, hitUv.x) * smoothstep(0.98, 0.88, hitUv.x)
          : ceilingCascade * trofferHighlight * sillSoffitClip;

        // 3. Restrained Luxury Palette (warm champagne-ivory + sparse daylight + 2950K lobbies & vaulted bays)
        float tempRoll = hash21(vec2(suiteSeed * 31.7, floorIdx + 2.3));
        vec3 baseSuiteTint = mix(STUDIO_INTERIOR_ALABASTER, STUDIO_INTERIOR_CHAMPAGNE, smoothstep(0.10, 0.68, tempRoll));
        float isCoolDaylight = step(0.86, tempRoll) * (1.0 - heroBaseZone);
        vec3 roomTint = mix(baseSuiteTint, STUDIO_INTERIOR_DAYLIGHT, isCoolDaylight);
        if (uInteriorMode > 0.5 || heroBaseZone > 0.2) {
          roomTint = mix(roomTint, STUDIO_INTERIOR_LOBBY, max(heroBaseZone, step(0.5, uInteriorMode)) * 0.94);
        }

        // Fresnel glazing attenuation: oblique faces dim cleanly under dark glossy outer glass
        float glassTransmission = (1.0 - fresnel) * smoothstep(0.06, 0.38, NdotV);
        float suiteLuminance = 0.58 + 0.50 * hash21(vec2(suiteSeed * 13.0, floorIdx * 7.0));
        float heroScale = uInteriorMode > 1.5 ? 1.28 : (uInteriorMode > 0.5 ? 1.18 : (0.68 + 0.58 * uIsHero));
        float reflectShimmer = mix(1.0, 1.52, uIsReflectPass);

        float interiorLum = occupancy * bayGlowProfile
          * (1.0 - frameMask) * (1.0 - isRoof)
          * suiteLuminance * heroScale * reflectShimmer;

        vec3 slabInteriorBase = vec3(0.0038, 0.0042, 0.0050) * interiorDepthShade * (0.24 + 0.76 * keyFace);
        vec3 interiorGlow = roomTint * interiorLum * backWallMask * interiorBoxAo * glassTransmission * (1.0 + uPressDepth * 0.25);
        // Preserve crisp neutral gallery white / soft platinum luminance inside recessed lobbies & arches
        vec3 tintedAbsorption = mix(beerLambert, vec3(0.858, 0.866, 0.880), step(0.5, uInteriorMode) * 0.65);
        vec3 innerParallaxLayer = slabInteriorBase * beerLambert + interiorGlow * tintedAbsorption;

        // Pitch-dark smoked outer glass body vs matte dark anodized spandrel/mullion strips IN FRONT
        vec3 glassBody = uBaseColor * beerLambert * (vec3(0.08) + 0.96 * keyDiffuseRgb) + innerParallaxLayer;
        vec3 spandrelBody = vec3(0.0068, 0.0074, 0.0090) * (vec3(0.14) + 1.18 * keyDiffuseRgb);
        vec3 facadeBase = mix(glassBody, spandrelBody, frameMask);

        // Outer Glass Layer: Dark Glossy Smoked Glass with Narrow, Sleek Vertical Softbox Streaks Along Edges
        float heightSheen = smoothstep(0.50, 7.6, vWorldPos.y);
        vec3 stripDirA = normalize(vec3(keyL.x * 0.78, 0.32, keyL.z * 0.78));
        vec3 stripDirB = normalize(vec3(0.76, 0.34, -0.32));
        vec3 R_vert = normalize(vec3(R_g.x, R_g.y * 0.22, R_g.z));
        float narrowStreakA = pow(max(0.0, dot(R_vert, stripDirA)), 26.0);
        float narrowStreakB = pow(max(0.0, dot(R_vert, stripDirB)), 34.0) * 0.65;

        float inUnitU_g = step(-0.002, vUv.x) * step(vUv.x, 1.002);
        float facadeEdgeU = inUnitU_g * (1.0 - smoothstep(0.0, 0.14, min(vUv.x, 1.0 - vUv.x))) * uAllowSideChamfers;
        float grazingGlass = pow(1.0 - NdotV, 5.2);

        vec3 narrowSoftboxStreaks = STUDIO_KEY_WARM
          * (0.14 + 0.36 * heightSheen)
          * (narrowStreakA * uKeyIntensity + narrowStreakB * uRimIntensity * 0.75)
          * (0.28 + 0.72 * fresnel + 0.45 * facadeEdgeU)
          * mix(0.16, 1.0, castShadow)
          * (1.0 - frameMask * 0.90)
          * (1.0 - isRoof);

        vec3 edgeGrazingSheen = STUDIO_PLATINUM_RIM
          * grazingGlass
          * (0.052 + 0.125 * facadeEdgeU)
          * (1.0 - frameMask * 0.88)
          * (1.0 - isRoof)
          * (0.5 + 0.5 * uRimIntensity);

        float reflVisibility = mix(0.14, 1.0, keyFace * 0.88 + 0.12) * mix(0.24, 1.0, castShadow);
        vec3 glassReflection = studioIbl * pow(1.0 - NdotV, 3.2) * 0.042 * (1.0 - frameMask * 0.88) * reflVisibility
          + narrowSoftboxStreaks
          + edgeGrazingSheen;

        // Razor-thin platinum edge rim glint ONLY along sharp vertical corners, setbacks & parapets
        float faceKeyVis = smoothstep(-0.08, 0.25, dot(baseN, keyL));
        float chamferKeyFacing = pow(max(dot(N, keyL), 0.0), 2.0) * castShadow * uKeyIntensity * faceKeyVis;
        vec3 chamferCatch = STUDIO_KEY_WARM * chamferKeyFacing
          * chamferMask * vec3(0.12, 0.13, 0.15) * uEdgeLuminance * (0.25 + 0.35 * uIsHero);
        vec3 platinumRim = computePlatinumEdgeGlint(
          baseN,
          N,
          V,
          keyL,
          chamferMask,
          uRimIntensity,
          uKeyIntensity
        ) * (0.24 + 0.12 * uIsHero) * uEdgeLuminance * (1.0 + uPressDepth * 0.12);

        vec3 glassLit = facadeBase * contactAo
          + glassReflection * mix(0.55, 1.0, contactAo)
          + chamferCatch
          + platinumRim
          + vec3(0.18, 0.19, 0.22) * uHighlightBoost * 0.35;

        // Dead-matte architectural slate/concrete rooftop (roughness ~0.88, deep obsidian, crisp parapet edge rim)
        float roofGrid = max(
          sdfGridLine(vWorldPos.x, 1.8, 0.020),
          sdfGridLine(vWorldPos.z, 1.8, 0.020)
        );
        float roofKey = max(dot(baseN, keyL), 0.0) * castShadow * uKeyIntensity;
        float roofFill = wrapDiffuse(baseN, fillL, 0.35) * uFillIntensity * 0.30;
        vec3 roofLit = STUDIO_FILL_COOL * (0.14 * uFillIntensity + roofFill) + STUDIO_KEY_WARM * (roofKey * 0.98);
        float roofBevel = pow(max(dot(N, keyL), 0.0), 2.0) * castShadow * chamferMask * smoothstep(0.0, 0.4, dot(N - baseN, keyL));

        vec3 roofColor =
          vec3(0.0082, 0.0086, 0.0098) * (0.88 + 0.22 * grain) * roofLit * canyonAo
          + vec3(0.010) * roofGrid * mix(STUDIO_FILL_COOL * 0.28, STUDIO_KEY_WARM, castShadow) * canyonAo
          + STUDIO_KEY_WARM * vec3(0.015, 0.015, 0.016) * roofBevel * uKeyIntensity
          + platinumRim * 0.85;

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
      uShadowMap: { value: null },
      uShadowMatrix: { value: new THREE.Matrix4() },
      uShadowTexelSize: { value: new THREE.Vector2(1 / 1024, 1 / 1024) },
      uHasShadowMap: { value: 0.0 },
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
      uniform sampler2D uShadowMap;
      uniform mat4 uShadowMatrix;
      uniform vec2 uShadowTexelSize;
      uniform float uHasShadowMap;
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

        float plinthRimBand = 0.0;
        vec4 chamferResult;
        if (uRadialAnisotropy > 0.5) {
          // Machined circular plinth bevel: crisp specular catchlight along the upper chamfer & rim lip
          float radDist = length(vWorldPos.xz);
          float upperBevelZone = smoothstep(-0.055, -0.008, vWorldPos.y)
            * smoothstep(uBasinRadius + 0.12, uBasinRadius + 0.185, radDist);
          float topLipZone = smoothstep(0.008, 0.026, vWorldPos.y)
            * smoothstep(uBasinRadius + 0.11, uBasinRadius + 0.16, radDist);
          plinthRimBand = clamp(max(upperBevelZone, topLipZone), 0.0, 1.0);
          chamferResult = vec4(baseN, plinthRimBand);
        } else {
          chamferResult = computeMachinedChamfer(
            baseN,
            vWorldPos,
            vUv,
            0.38,
            uAllowSideChamfers
          );
        }
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
          uRoughness + (brush - 0.5) * 0.04 + (micro.r - 0.5) * 0.028 - chamferMask * 0.045,
          0.16,
          0.52
        );

        float ax = max(0.10, localRoughness * (1.0 - uAnisotropy * 0.52));
        float ay = max(0.22, localRoughness * (1.0 + uAnisotropy * 0.82));

        vec3 keyL = normalize(uSoftboxKeyDir);
        vec3 rimL = normalize(vec3(-0.82, 0.34, 0.44));
        vec3 fillL = normalize(vec3(-keyL.x * 0.9, 0.32, -keyL.z * 0.9));

        float castShadow = uRadialAnisotropy > 0.5
          ? 1.0
          : (uHasShadowMap > 0.5
              ? sampleOrthographicShadowPCF(uShadowMap, uShadowMatrix, uShadowTexelSize, vWorldPos, baseN, keyL, uHasShadowMap)
              : sampleDirectionalHeightShadow(uGroundAoMap, vWorldPos, baseN, keyL, uBasinRadius));

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
          ? mix(0.66 - uPressDepth * 0.10, 1.0, smoothstep(-0.44, -0.28, vWorldPos.y))
          : mix(0.42 - uPressDepth * 0.14, 1.0, pow(smoothstep(0.06, 0.42, vWorldPos.y), 1.0 + uPressDepth * 0.50));
        float specScale = uRadialAnisotropy > 0.5 ? 0.14 : 0.42;

        float fresnel = fresnelSchlick(NdotV, 0.58);
        // Strong 3:1 key-to-fill directional shading with subtle warm key / cool shadow tint
        float keyDiff = max(dot(N, keyL), 0.0) * castShadow * uKeyIntensity;
        float rimDiff = max(dot(N, rimL), 0.0) * uRimIntensity * 0.25;
        float fillDiff = wrapDiffuse(N, fillL, 0.25) * uFillIntensity * 0.28;
        vec3 diffuseRgb = STUDIO_FILL_COOL * (0.14 * uFillIntensity + fillDiff)
          + STUDIO_KEY_WARM * (keyDiff * 0.74)
          + vec3(0.98, 0.99, 1.02) * rimDiff;

        // Directional chamfer catchlight strictly facing key/rim softboxes
        float faceKeyVis = smoothstep(-0.08, 0.25, dot(baseN, keyL));
        float chamferKeyFacing = pow(max(dot(N, keyL), 0.0), 2.0) * castShadow * uKeyIntensity * faceKeyVis;
        float chamferRimFacing = pow(max(dot(N, rimL), 0.0), 2.2) * uRimIntensity * 0.32 * smoothstep(-0.1, 0.25, dot(baseN, rimL));

        // Additional crisp machined edge sheen on the anodized plinth bevel facing the key light
        float plinthRadialKey = 0.0;
        if (uRadialAnisotropy > 0.5 && plinthRimBand > 0.01) {
          vec2 radDir = normalize(vWorldPos.xz + vec2(1e-5));
          float keyArc = smoothstep(-0.15, 0.65, dot(radDir, normalize(keyL.xz)));
          vec3 H_plinth = normalize(keyL + V);
          float bevelSpec = pow(max(dot(N, H_plinth), 0.0), 38.0);
          plinthRadialKey = plinthRimBand * (0.28 * keyArc + 0.72 * bevelSpec) * uKeyIntensity;
        }

        vec3 metalAlbedo = uBaseColor * (0.88 + 0.24 * brush) * (0.94 + 0.12 * grain);
        vec3 color = metalAlbedo * diffuseRgb * baseAo
          + STUDIO_KEY_WARM * vec3(0.84, 0.86, 0.89) * (anisoSpec * 0.22 * baseAo * specScale)
          + iblSpec * (0.04 + 0.14 * fresnel) * baseAo * specScale * mix(0.45, 1.0, castShadow)
          + (STUDIO_KEY_WARM * chamferKeyFacing + vec3(0.98, 0.99, 1.02) * chamferRimFacing) * chamferMask * vec3(0.24, 0.25, 0.28) * (1.0 + uPressDepth * 0.2) * specScale
          + STUDIO_KEY_WARM * vec3(0.105, 0.108, 0.116) * plinthRadialKey;

        if (uRadialAnisotropy < 0.5) {
          vec3 platinumRim = computePlatinumEdgeGlint(
            baseN,
            chamferN,
            V,
            keyL,
            chamferMask,
            uRimIntensity,
            uKeyIntensity
          );
          color += platinumRim * 0.22 * baseAo;

          // Subtle warm champagne-amber under-glow on recessed underside of Battery Maritime / Sails arches
          float inArchZone = step(4.6, vWorldPos.x) * step(vWorldPos.x, 10.2)
            * step(0.95, vWorldPos.y) * step(vWorldPos.y, 1.82)
            * step(-1.1, vWorldPos.z) * step(vWorldPos.z, 1.9);
          float undersideMask = smoothstep(0.18, -0.45, baseN.y) * inArchZone;
          color += STUDIO_WARM_POOL_3000K * undersideMask * 0.052;

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
  microImperfectionMap: THREE.Texture,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    transparent: true,
    depthWrite: false,
    uniforms: {
      uGrainMap: { value: grainTexture },
      uMicroMap: { value: microImperfectionMap },
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
      uniform sampler2D uMicroMap;
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

        // Seamless radial cyclorama fade into transparent page studio backdrop (zero rectangular seam)
        float outerFade = smoothstep(plinthR + 0.20, plinthR + 4.8, dist);
        if (outerFade >= 0.999) {
          discard;
        }

        // Subtle bead-blasted dark slate tabletop micro-grain & key-light softbox pool
        vec3 keyL = normalize(uSoftboxKeyDir);
        vec2 keyHoriz = normalize(keyL.xz);
        float keyPool = smoothstep(18.0, -8.0, dot(p, -keyHoriz));
        float centerPool = exp(-pow(dist / (plinthR * 1.22), 2.0));
        float fineGrain = texture(uGrainMap, p * 0.48).r;
        vec3 beadGrain = texture(uMicroMap, p * 1.65).rgb;
        float slateMicro = (beadGrain.r - 0.5) * 0.14 + (fineGrain - 0.5) * 0.12;

        vec3 studioBase = vec3(0.0046, 0.0048, 0.0055);
        vec3 tabletopLit = mix(
          studioBase,
          vec3(0.0158, 0.0164, 0.0190),
          centerPool * (0.42 + 0.58 * keyPool) * (0.72 + 0.28 * uKeyIntensity)
        ) * (1.0 + slateMicro);

        // 1. Tight ~1px dark ambient occlusion crevice directly under the metal plinth rim
        float creviceDist = max(0.0, dist - (plinthR - 0.02));
        float tightRimCrevice = 1.0 - exp(-pow(creviceDist / 0.045, 1.15)) * (0.96 + uPressDepth * 0.03);
        float nearFootAo = 1.0 - exp(-creviceDist * 3.2) * 0.62;
        float tightContactAo = min(tightRimCrevice, nearFootAo);

        // 2. Soft directional tabletop shadow cast toward the lower-right (opposite upper-left key light)
        vec2 shadowShift = -keyHoriz * (1.28 + (1.0 - clamp(keyL.y, 0.25, 0.85)) * 1.38);
        vec2 rel = p - shadowShift;
        float along = dot(rel, keyHoriz) / (plinthR * 1.08);
        vec2 perpVec = vec2(-keyHoriz.y, keyHoriz.x);
        float across = dot(rel, perpVec) / (plinthR * 1.01);
        float ovalDist = sqrt(along * along + across * across);
        float ovalPenumbra = 1.0 - smoothstep(0.82, 1.22, ovalDist) * 0.88;
        float castShadowMask = mix(1.0, 0.09, 1.0 - ovalPenumbra);

        // 3. Faint specular bounce of the brushed silver plinth edge onto the tabletop right outside its rim
        vec2 radDir = dist > 0.001 ? p / dist : vec2(0.0, 1.0);
        float rimBounceArc = smoothstep(-0.35, 0.72, dot(radDir, keyHoriz));
        float bounceBand = exp(-pow(max(0.0, dist - (plinthR + 0.045)) / 0.16, 2.0)) * step(plinthR - 0.01, dist);
        float microGlint = 0.78 + 0.44 * beadGrain.g;
        vec3 plinthSilverBounce = vec3(0.018, 0.019, 0.021)
          * bounceBand
          * (0.28 + 0.72 * rimBounceArc)
          * microGlint
          * uKeyIntensity
          * (1.0 + uPressDepth * 0.25);

        float totalShadow = min(tightContactAo, castShadowMask);
        vec3 groundedFloor = mix(tabletopLit * totalShadow + plinthSilverBounce, studioBase, outerFade * 0.72);
        float alpha = (1.0 - outerFade) * 0.94;

        fragColor = vec4(groundedFloor, alpha);
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
