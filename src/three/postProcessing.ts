import * as THREE from 'three';

export interface HdrPostProcessor {
  render: (
    scene: THREE.Scene,
    camera: THREE.Camera,
    dtSeconds: number,
    focusDistance?: number,
    depthExcludeObjects?: THREE.Object3D[],
    timeSeconds?: number,
  ) => void;
  setSize: (width: number, height: number, pixelRatio: number) => void;
  setBloomStrength: (strength: number) => void;
  dispose: () => void;
  readonly isHdrFloatBuffer: boolean;
  readonly activeBloomLevels: number;
}

const FULLSCREEN_VERTEX_SHADER = /* glsl */ `
  out vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

/**
 * Lightweight half-resolution linear view-depth shader supporting InstancedMesh.
 * Packs normalized distance [0..80m] into RG channels (16-bit precision) for
 * artifact-free macro tilt-shift DoF without MSAA DepthTexture incompatibilities.
 */
const DOF_DEPTH_VERTEX_SHADER = /* glsl */ `
  out float vViewDist;

  void main() {
    vec4 localPos = vec4(position, 1.0);
    #ifdef USE_INSTANCING
      localPos = instanceMatrix * localPos;
    #endif
    vec4 mvPos = viewMatrix * modelMatrix * localPos;
    vViewDist = -mvPos.z;
    gl_Position = projectionMatrix * mvPos;
  }
`;

const DOF_DEPTH_FRAGMENT_SHADER = /* glsl */ `
  in float vViewDist;
  out vec4 fragColor;

  void main() {
    float norm = clamp(vViewDist / 85.0, 0.0, 1.0);
    float scaled = norm * 255.0;
    float hi = floor(scaled) / 255.0;
    float lo = fract(scaled);
    fragColor = vec4(hi, lo, 0.0, 1.0);
  }
`;

/**
 * Pass 1: HDR Peak Extraction (Prefilter) with soft knee.
 */
const PREFILTER_FRAGMENT_SHADER = /* glsl */ `
  uniform sampler2D uSourceTex;
  uniform vec2 uTexelSize;
  uniform float uThreshold;
  uniform float uSoftKnee;

  in vec2 vUv;
  out vec4 fragColor;

  vec3 sampleBox(vec2 uv) {
    vec2 o = uTexelSize * 0.5;
    vec3 a = texture(uSourceTex, uv + vec2(-o.x, -o.y)).rgb;
    vec3 b = texture(uSourceTex, uv + vec2( o.x, -o.y)).rgb;
    vec3 c = texture(uSourceTex, uv + vec2(-o.x,  o.y)).rgb;
    vec3 d = texture(uSourceTex, uv + vec2( o.x,  o.y)).rgb;
    return (a + b + c + d) * 0.25;
  }

  void main() {
    vec3 col = max(sampleBox(vUv), vec3(0.0));
    float br = max(col.r, max(col.g, col.b));
    float knee = uThreshold * uSoftKnee + 1e-5;
    float soft = clamp(br - uThreshold + knee, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee + 1e-5);
    float contrib = max(soft, br - uThreshold) / max(br, 1e-5);
    vec3 extracted = min(col * max(contrib, 0.0), vec3(8.0));
    fragColor = vec4(extracted, 1.0);
  }
`;

const KAWASE_DOWNSAMPLE_SHADER = /* glsl */ `
  uniform sampler2D uSourceTex;
  uniform vec2 uHalfPixel;

  in vec2 vUv;
  out vec4 fragColor;

  void main() {
    vec3 sum = texture(uSourceTex, vUv).rgb * 4.0;
    sum += texture(uSourceTex, vUv - uHalfPixel.xy).rgb;
    sum += texture(uSourceTex, vUv + uHalfPixel.xy).rgb;
    sum += texture(uSourceTex, vUv + vec2(uHalfPixel.x, -uHalfPixel.y)).rgb;
    sum += texture(uSourceTex, vUv - vec2(uHalfPixel.x, -uHalfPixel.y)).rgb;
    fragColor = vec4(sum * 0.125, 1.0);
  }
`;

const KAWASE_UPSAMPLE_SHADER = /* glsl */ `
  uniform sampler2D uSourceTex;
  uniform sampler2D uDetailTex;
  uniform vec2 uHalfPixel;
  uniform float uScatter;

  in vec2 vUv;
  out vec4 fragColor;

  void main() {
    vec2 hp = uHalfPixel * uScatter;
    vec3 sum = vec3(0.0);
    sum += texture(uSourceTex, vUv + vec2(-hp.x * 2.0, 0.0)).rgb;
    sum += texture(uSourceTex, vUv + vec2(-hp.x, hp.y)).rgb * 2.0;
    sum += texture(uSourceTex, vUv + vec2(0.0, hp.y * 2.0)).rgb;
    sum += texture(uSourceTex, vUv + vec2(hp.x, hp.y)).rgb * 2.0;
    sum += texture(uSourceTex, vUv + vec2(hp.x * 2.0, 0.0)).rgb;
    sum += texture(uSourceTex, vUv + vec2(hp.x, -hp.y)).rgb * 2.0;
    sum += texture(uSourceTex, vUv + vec2(0.0, -hp.y * 2.0)).rgb;
    sum += texture(uSourceTex, vUv + vec2(-hp.x, -hp.y)).rgb * 2.0;
    vec3 blurred = sum / 12.0;
    vec3 detail = texture(uDetailTex, vUv).rgb;
    fragColor = vec4(blurred * 0.72 + detail * 0.65, 1.0);
  }
`;

/**
 * Final Display Composite Shader:
 * - Subtle optical Macro Depth-of-Field (9-tap Poisson bokeh in linear HDR space)
 *   focused sharply on hero waterfront towers, gently softening far rear blocks & closest plinth rim
 * - Combines HDR scene, Dual-Kawase optical bloom, and wide subtle halation
 * - Applies AgX-calibrated Filmic Tone Mapping for natural highlight desaturation
 * - Converts linear color to sRGB gamma + Jimenez IGN anti-banding dithering
 */
const FINAL_COMPOSITE_SHADER = /* glsl */ `
  uniform sampler2D uHdrSceneTex;
  uniform sampler2D uBloomTex;
  uniform sampler2D uHalationTex;
  uniform sampler2D uDepthTex;
  uniform vec2 uTexelSize;
  uniform float uFocusDistance;
  uniform float uDofEnabled;
  uniform float uExposure;
  uniform float uBloomStrength;
  uniform float uHalationStrength;
  uniform float uDitherScale;
  uniform float uTime;

  in vec2 vUv;
  out vec4 fragColor;

  float decodeDepth(vec2 uv) {
    vec2 rg = texture(uDepthTex, clamp(uv, vec2(0.001), vec2(0.999))).rg;
    return (rg.r + rg.g / 255.0) * 85.0;
  }

  float computeCoc(float viewDist) {
    // Background sky/void (depth ~0 or >=84) stays gently soft or matches distant edge
    if (viewDist <= 0.1 || viewDist >= 84.0) return 0.30;
    // Razor-sharp focal corridor on hero waterfront towers & Brookfield domes,
    // whisper-subtle optical softening on far rear blocks & nearest front plinth rim
    float delta = abs(viewDist - uFocusDistance);
    return clamp((delta - 5.4) / 15.5, 0.0, 1.0);
  }

  vec4 sampleOpticalDof(vec2 uv) {
    vec4 centerSample = texture(uHdrSceneTex, uv);
    centerSample.rgb = max(centerSample.rgb, vec3(0.0));
    centerSample.a = clamp(centerSample.a, 0.0, 1.0);
    if (uDofEnabled < 0.5 || centerSample.a <= 0.01) return centerSample;

    float centerDist = decodeDepth(uv);
    float centerCoc = computeCoc(centerDist);
    if (centerCoc <= 0.015) return centerSample;

    // Whisper-subtle optical macro lens radius (max ~2.0 CSS-equivalent px)
    float radiusPx = centerCoc * 2.05;
    vec2 radiusUv = uTexelSize * radiusPx;

    const vec2 POISSON_8[8] = vec2[8](
      vec2(-0.7071,  0.7071),
      vec2( 0.7071,  0.7071),
      vec2(-0.7071, -0.7071),
      vec2( 0.7071, -0.7071),
      vec2( 0.0000,  1.0000),
      vec2( 1.0000,  0.0000),
      vec2( 0.0000, -1.0000),
      vec2(-1.0000,  0.0000)
    );

    vec4 accum = centerSample;
    float totalW = 1.0;

    for (int i = 0; i < 8; i++) {
      vec2 tapUv = clamp(uv + POISSON_8[i] * radiusUv, vec2(0.001), vec2(0.999));
      float tapDist = decodeDepth(tapUv);
      float tapCoc = computeCoc(tapDist);
      vec4 tapSample = texture(uHdrSceneTex, tapUv);
      tapSample.rgb = max(tapSample.rgb, vec3(0.0));
      tapSample.a = clamp(tapSample.a, 0.0, 1.0);
      // Prevent sharp midground hero towers from bleeding into defocused rear/foreground
      // and prevent transparent sky taps from darkening solid geometry edges
      float bleedGuard = (tapDist < centerDist - 2.0) ? clamp(tapCoc + 0.25, 0.0, 1.0) : max(tapCoc, centerCoc * 0.65);
      float alphaGuard = mix(1.0, tapSample.a, step(0.5, centerSample.a));
      float w = 0.75 * bleedGuard * alphaGuard;
      accum += tapSample * w;
      totalW += w;
    }
    return accum / totalW;
  }

  vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
  }

  vec3 agxToneMap(vec3 col) {
    const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
      vec3(0.6274, 0.0691, 0.0164),
      vec3(0.3293, 0.9195, 0.0880),
      vec3(0.0433, 0.0113, 0.8956)
    );
    const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
      vec3(1.6605, -0.1246, -0.0182),
      vec3(-0.5876, 1.1329, -0.1006),
      vec3(-0.0728, -0.0083, 1.1187)
    );
    const mat3 AgXInsetMatrix = mat3(
      vec3(0.856627153315983,  0.137318972929847,  0.11189821299995),
      vec3(0.0951212405381588, 0.761241990602591,  0.0767994186031903),
      vec3(0.0482516061458583, 0.101439036467562,  0.811302368396859)
    );
    const mat3 AgXOutsetMatrix = mat3(
      vec3( 1.1271005818144368,  -0.1413297634984383,  -0.14132976349843826),
      vec3(-0.11060664309660323,  1.157823702216272,   -0.11060664309660294),
      vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405)
    );
    const float minEv = -12.47393;
    const float maxEv = 4.026069;

    vec3 val = LINEAR_SRGB_TO_LINEAR_REC2020 * max(col, vec3(0.0));
    val = AgXInsetMatrix * val;
    val = clamp((log2(max(val, vec3(1e-10))) - minEv) / (maxEv - minEv), 0.0, 1.0);
    val = agxDefaultContrastApprox(val);
    val = AgXOutsetMatrix * val;
    val = pow(max(vec3(0.0), val), vec3(2.2));
    val = LINEAR_REC2020_TO_LINEAR_SRGB * val;
    return clamp(val, 0.0, 1.0);
  }

  vec3 linearToSRGB(vec3 lin) {
    vec3 low = lin * 12.92;
    vec3 high = 1.055 * pow(max(lin, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    bvec3 cutoff = lessThanEqual(lin, vec3(0.0031308));
    return mix(high, low, vec3(cutoff));
  }

  float interleavedGradientNoise(vec2 pixelPos) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pixelPos, magic.xy)));
  }

  void main() {
    vec4 hdrSample = sampleOpticalDof(vUv);
    vec3 bloom = max(texture(uBloomTex, vUv).rgb, vec3(0.0));
    vec3 halation = max(texture(uHalationTex, vUv).rgb, vec3(0.0));

    vec3 glowHdr = bloom * uBloomStrength
      + halation * vec3(0.95, 0.97, 1.02) * uHalationStrength;
    float glowLum = max(glowHdr.r, max(glowHdr.g, glowHdr.b));

    float sceneAlpha = clamp(hdrSample.a, 0.0, 1.0);
    float outAlpha = clamp(max(sceneAlpha, glowLum * 1.4), 0.0, 1.0);

    // Un-premultiply linear HDR over transparent sky before AgX tone mapping (guarded against 8-bit halo banding)
    float unpremulDiv = max(sceneAlpha, glowLum * 1.4 + 0.12 * (1.0 - sceneAlpha));
    vec3 combinedHdr = outAlpha > 0.002
      ? (hdrSample.rgb + glowHdr) / max(unpremulDiv, 0.002)
      : vec3(0.0);

    vec3 toneMapped = agxToneMap(combinedHdr * uExposure);
    vec3 srgb = linearToSRGB(toneMapped);

    // Animated blue-noise-style spatio-temporal sensor grain (~1.5% opacity across dark gradients)
    vec2 frameShift = vec2(
      fract(uTime * 11.731) * 131.0,
      fract(uTime * 17.397) * 197.0
    );
    vec2 grainCoord = gl_FragCoord.xy + frameShift;
    float n1 = interleavedGradientNoise(grainCoord);
    float n2 = interleavedGradientNoise(grainCoord + vec2(37.0, 17.0));
    float blueNoise = (n1 - n2); // zero-mean high-frequency triangular distribution in [-1, 1]
    float dither = blueNoise * 0.75 * uDitherScale;
    float lum = dot(srgb, vec3(0.2126, 0.7152, 0.0722));
    float darkToneWeight = mix(1.0, 0.38, smoothstep(0.04, 0.55, lum));
    float sensorGrain = blueNoise * 0.015 * darkToneWeight * step(0.02, outAlpha);

    // Gentle 1.5% border safety feather only at the extreme viewport perimeter
    float borderFeather =
      smoothstep(0.0, 0.015, vUv.x) *
      smoothstep(1.0, 0.985, vUv.x) *
      smoothstep(0.0, 0.015, vUv.y) *
      smoothstep(1.0, 0.985, vUv.y);

    fragColor = vec4(clamp(srgb + vec3(dither + sensorGrain), 0.0, 1.0), outAlpha * borderFeather);
  }
`;

export function createHdrPostProcessor(
  renderer: THREE.WebGLRenderer,
  initialWidth: number,
  initialHeight: number,
  initialPixelRatio: number,
  isSoftwareGl: boolean,
): HdrPostProcessor {
  const gl = renderer.getContext() as WebGL2RenderingContext;
  const supportsHalfFloatFbo = Boolean(
    renderer.capabilities.isWebGL2 &&
      (gl.getExtension('EXT_color_buffer_float') ||
        gl.getExtension('EXT_color_buffer_half_float')),
  );

  const fboType = supportsHalfFloatFbo ? THREE.HalfFloatType : THREE.UnsignedByteType;
  const msaaSamples = !isSoftwareGl && renderer.capabilities.isWebGL2
    ? Math.min(4, renderer.capabilities.maxSamples || 0)
    : 0;

  const wPx = Math.max(2, Math.floor(initialWidth * initialPixelRatio));
  const hPx = Math.max(2, Math.floor(initialHeight * initialPixelRatio));

  const hdrTarget = new THREE.WebGLRenderTarget(wPx, hPx, {
    type: fboType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: true,
    stencilBuffer: false,
    samples: msaaSamples,
  });

  // Half-res non-MSAA target for linear camera-space depth (Macro Tilt-Shift DoF)
  const depthTarget = new THREE.WebGLRenderTarget(
    Math.max(2, Math.floor(wPx * 0.5)),
    Math.max(2, Math.floor(hPx * 0.5)),
    {
      type: THREE.UnsignedByteType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      stencilBuffer: false,
    },
  );

  const dofDepthMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: DOF_DEPTH_VERTEX_SHADER,
    fragmentShader: DOF_DEPTH_FRAGMENT_SHADER,
  });

  const maxPyramidLevels = 4;
  const bloomDownTargets: THREE.WebGLRenderTarget[] = [];
  const bloomUpTargets: THREE.WebGLRenderTarget[] = [];

  for (let i = 0; i < maxPyramidLevels; i++) {
    const scale = 2 ** (i + 1);
    const lw = Math.max(2, Math.floor(wPx / scale));
    const lh = Math.max(2, Math.floor(hPx / scale));
    const opts: THREE.RenderTargetOptions = {
      type: fboType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
    };
    bloomDownTargets.push(new THREE.WebGLRenderTarget(lw, lh, opts));
    bloomUpTargets.push(new THREE.WebGLRenderTarget(lw, lh, opts));
  }

  const quadGeo = new THREE.BufferGeometry();
  const vertices = new Float32Array([
    -1, -1, 0,
     3, -1, 0,
    -1,  3, 0,
  ]);
  const uvs = new Float32Array([
    0, 0,
    2, 0,
    0, 2,
  ]);
  quadGeo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  quadGeo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

  const postScene = new THREE.Scene();
  const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  const prefilterMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    depthTest: false,
    depthWrite: false,
    uniforms: {
      uSourceTex: { value: null },
      uTexelSize: { value: new THREE.Vector2(1 / wPx, 1 / hPx) },
      uThreshold: { value: 0.88 },
      uSoftKnee: { value: 0.22 },
    },
    vertexShader: FULLSCREEN_VERTEX_SHADER,
    fragmentShader: PREFILTER_FRAGMENT_SHADER,
  });

  const downsampleMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    depthTest: false,
    depthWrite: false,
    uniforms: {
      uSourceTex: { value: null },
      uHalfPixel: { value: new THREE.Vector2(0.001, 0.001) },
    },
    vertexShader: FULLSCREEN_VERTEX_SHADER,
    fragmentShader: KAWASE_DOWNSAMPLE_SHADER,
  });

  const upsampleMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    depthTest: false,
    depthWrite: false,
    uniforms: {
      uSourceTex: { value: null },
      uDetailTex: { value: null },
      uHalfPixel: { value: new THREE.Vector2(0.001, 0.001) },
      uScatter: { value: 0.52 },
    },
    vertexShader: FULLSCREEN_VERTEX_SHADER,
    fragmentShader: KAWASE_UPSAMPLE_SHADER,
  });

  const compositeMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    depthTest: false,
    depthWrite: false,
    uniforms: {
      uHdrSceneTex: { value: hdrTarget.texture },
      uBloomTex: { value: bloomUpTargets[0].texture },
      uHalationTex: { value: bloomDownTargets[2].texture },
      uDepthTex: { value: depthTarget.texture },
      uTexelSize: { value: new THREE.Vector2(1 / wPx, 1 / hPx) },
      uFocusDistance: { value: 44.0 },
      uDofEnabled: { value: 1.0 },
      uExposure: { value: 1.65 },
      uBloomStrength: { value: 0.19 },
      uHalationStrength: { value: 0.018 },
      uDitherScale: { value: 1.0 / 255.0 },
      uTime: { value: 0.0 },
    },
    vertexShader: FULLSCREEN_VERTEX_SHADER,
    fragmentShader: FINAL_COMPOSITE_SHADER,
  });

  const quadMesh = new THREE.Mesh(quadGeo, compositeMat);
  quadMesh.frustumCulled = false;
  postScene.add(quadMesh);

  let activeBloomLevels = isSoftwareGl ? 2 : 4;
  let slowFrameStreak = 0;

  const updateFrameBudget = (dtSeconds: number) => {
    if (isSoftwareGl) return;
    const dtMs = dtSeconds * 1000;
    if (dtMs > 36) {
      slowFrameStreak += 1;
      if (slowFrameStreak >= 4 && activeBloomLevels > 2) {
        activeBloomLevels = 2;
        slowFrameStreak = 0;
      } else if (slowFrameStreak >= 8 && activeBloomLevels > 1) {
        activeBloomLevels = 1;
        slowFrameStreak = 0;
      }
    } else {
      slowFrameStreak = 0;
    }
  };

  const render = (
    scene: THREE.Scene,
    camera: THREE.Camera,
    dtSeconds: number,
    focusDistance = 44.0,
    depthExcludeObjects: THREE.Object3D[] = [],
    timeSeconds = 0,
  ) => {
    updateFrameBudget(dtSeconds);

    // 1. Render scene into linear HDR FBO (RGBA16F) with transparent backdrop alpha = 0
    renderer.setRenderTarget(hdrTarget);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(scene, camera);

    // 1b. Render lightweight half-res linear depth map for Macro Tilt-Shift DoF
    const prevOverride = scene.overrideMaterial;
    const prevBg = scene.background;
    const prevVisibility = depthExcludeObjects.map((o) => o.visible);
    for (const o of depthExcludeObjects) o.visible = false;
    scene.background = null;
    scene.overrideMaterial = dofDepthMat;
    renderer.setRenderTarget(depthTarget);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(scene, camera);
    scene.overrideMaterial = prevOverride;
    scene.background = prevBg;
    for (let i = 0; i < depthExcludeObjects.length; i++) {
      depthExcludeObjects[i].visible = prevVisibility[i];
    }

    // 2. Extract HDR peaks (>1.0) into Bloom Level 0
    prefilterMat.uniforms.uSourceTex.value = hdrTarget.texture;
    prefilterMat.uniforms.uTexelSize.value.set(1 / hdrTarget.width, 1 / hdrTarget.height);
    quadMesh.material = prefilterMat;
    renderer.setRenderTarget(bloomDownTargets[0]);
    renderer.render(postScene, postCamera);

    // 3. Dual-Kawase Downsample chain
    quadMesh.material = downsampleMat;
    for (let i = 1; i < activeBloomLevels; i++) {
      const prev = bloomDownTargets[i - 1];
      const dst = bloomDownTargets[i];
      downsampleMat.uniforms.uSourceTex.value = prev.texture;
      downsampleMat.uniforms.uHalfPixel.value.set(0.5 / prev.width, 0.5 / prev.height);
      renderer.setRenderTarget(dst);
      renderer.render(postScene, postCamera);
    }

    // 4. Dual-Kawase Upsample chain
    quadMesh.material = upsampleMat;
    let lastUpsampled = bloomDownTargets[activeBloomLevels - 1];
    for (let i = activeBloomLevels - 2; i >= 0; i--) {
      const dst = bloomUpTargets[i];
      const detail = bloomDownTargets[i];
      upsampleMat.uniforms.uSourceTex.value = lastUpsampled.texture;
      upsampleMat.uniforms.uDetailTex.value = detail.texture;
      upsampleMat.uniforms.uHalfPixel.value.set(
        0.5 / lastUpsampled.width,
        0.5 / lastUpsampled.height,
      );
      renderer.setRenderTarget(dst);
      renderer.render(postScene, postCamera);
      lastUpsampled = dst;
    }

    // 5. Final Composite: Macro DoF + HDR + Bloom + Halation -> AgX -> sRGB -> IGN Dithering
    compositeMat.uniforms.uHdrSceneTex.value = hdrTarget.texture;
    compositeMat.uniforms.uBloomTex.value = lastUpsampled.texture;
    compositeMat.uniforms.uHalationTex.value =
      bloomDownTargets[Math.min(2, activeBloomLevels - 1)].texture;
    compositeMat.uniforms.uDepthTex.value = depthTarget.texture;
    compositeMat.uniforms.uTexelSize.value.set(1 / hdrTarget.width, 1 / hdrTarget.height);
    compositeMat.uniforms.uFocusDistance.value = focusDistance;
    compositeMat.uniforms.uTime.value = timeSeconds;
    quadMesh.material = compositeMat;
    renderer.setRenderTarget(null);
    renderer.render(postScene, postCamera);
  };

  const setSize = (width: number, height: number, pixelRatio: number) => {
    const newW = Math.max(2, Math.floor(width * pixelRatio));
    const newH = Math.max(2, Math.floor(height * pixelRatio));
    hdrTarget.setSize(newW, newH);
    depthTarget.setSize(Math.max(2, Math.floor(newW * 0.5)), Math.max(2, Math.floor(newH * 0.5)));
    for (let i = 0; i < maxPyramidLevels; i++) {
      const scale = 2 ** (i + 1);
      const lw = Math.max(2, Math.floor(newW / scale));
      const lh = Math.max(2, Math.floor(newH / scale));
      bloomDownTargets[i].setSize(lw, lh);
      bloomUpTargets[i].setSize(lw, lh);
    }
  };

  const setBloomStrength = (strength: number) => {
    compositeMat.uniforms.uBloomStrength.value = strength;
  };

  const dispose = () => {
    hdrTarget.dispose();
    depthTarget.dispose();
    dofDepthMat.dispose();
    for (const rt of bloomDownTargets) rt.dispose();
    for (const rt of bloomUpTargets) rt.dispose();
    quadGeo.dispose();
    prefilterMat.dispose();
    downsampleMat.dispose();
    upsampleMat.dispose();
    compositeMat.dispose();
  };

  return {
    render,
    setSize,
    setBloomStrength,
    dispose,
    isHdrFloatBuffer: supportsHalfFloatFbo,
    get activeBloomLevels() {
      return activeBloomLevels;
    },
  };
}
