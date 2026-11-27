// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L191–246 (the settings: names, meanings and defaults).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: engine-free (K6.1); `froxelMix` added (design §3.4). Divergences: DIVERGES.md V33, V67, K6.1 rows.

/**
 * The settings of the core frame renderer (K6.1, design §2.2): his `frameRenderer.settings` (the same names, meanings
 * and v0.26 defaults), plus the K6 fields. Since K6.8 `@pmndrs/vdb/three`'s `VDBFrameRenderer.settings` is this type
 * (one declaration).
 */

/** Phase functions (his `phase`). */
export type VDBFramePhaseName = "isotropic" | "henyey-greenstein" | "dual-lobe" | "lorenz-mie" | "lorenz-mie-rgb";

/**
 * Multiple scattering modes (his `multipleScattering`, plus K6.3: `'mlp'` and `'multi-scale'` (K7.3, his 4e2d46f)).
 * `'neural'` and `'mlp'` are EXPERIMENTAL: both run the core's generic small-MLP evaluator on `renderer.mlp` (his
 * trained network by default), whose data shape may change before 1.0.
 */
export type VDBFrameMultipleScatteringMode = boolean | "two-sided" | "octaves" | "neural" | "mlp" | "multi-scale";

/**
 * The renderer's settings (`vdbFrameDefaultSettings()`; his `frameRenderer.settings`, K6.10a: each field's doc ported
 * from the deleted `./three` `VDBFrameRendererSettings`). Values out of the device's range clamp with a one-time
 * warning (`shadowResolution`, `castShadowMapSize`).
 */
export interface VDBFrameSettings {
  /** Froxel grid width (default 512). */
  width: number;
  /** Froxel grid height (default 512). */
  height: number;
  /** N > 0: at most one froxel per N x N viewport pixels. 0: no cap. */
  pixelsPerFroxel: number;
  /** Slices (default 24). */
  depth: number;
  /** Most sub-steps across one slice (default 32). */
  subSteps: number;
  /** Near distance of the froxel slices in view units (default 0.1). */
  near: number;
  /** Far distance of the froxel slices in view units (default Infinity: the camera far plane). */
  far: number;
  /** View distance where the density fades to 0 (FogExp2-like). Infinity: no fade. */
  fadeRadius: number;
  /** Golden-ratio jitter of the sub-steps, shadow start and grid marches (default false). */
  jitter: boolean;
  /** His froxel reprojection with a neighbourhood clamp (default true, his; V33). */
  temporal: boolean;
  /** History weight of the temporal pass (0.01–1; his default 0.2). */
  temporalBlend: number;
  /** `'volume'`: each volume lights itself with its own shadow march. `'frame'`: the world shadow grid (default, his; V33). */
  lighting: "volume" | "frame";
  /**
   * `'frame'` lighting: voxels on each side of each shadow grid cascade (default 64). Clamped (warned once) to what the
   * device holds: the transmittance stack must tile in `maxTextureDimension3D` and fit `maxBufferSize` (K6.8, K6.10a).
   */
  shadowResolution: number;
  /** `'frame'` lighting: shadow grid cascades, 1–4 (default 1). */
  shadowCascades: number;
  /** `'frame'` lighting: half size of the finest cascades around the camera (default 512). */
  shadowDistance: number;
  /**
   * The phase of every volume this renderer draws (default 'henyey-greenstein', g 0.8; his demos set 'dual-lobe'), and
   * of `phaseG` … `phaseGExponent`: `'frame'` lighting lights all volumes with it in one pass; in `'volume'` lighting
   * a volume's own phase parameters override it where set.
   */
  phase: VDBFramePhaseName;
  /** Henyey-Greenstein asymmetry `g` of the forward lobe (default 0.8). */
  phaseG: number;
  /** `g` of the backward lobe of the dual-lobe phases (default -0.3). */
  phaseBackG: number;
  /** Weight of the backward lobe of the dual-lobe phases (default 0.2). */
  phaseBackWeight: number;
  /** Wavelength exponent of `g` in spectral rendering (default 0: the same `g` in every channel). */
  phaseGExponent: number;
  /** Blend the volumes over the scene in sRGB, as WebGLRenderer (default true). false blends in linear space. */
  srgbBlending: boolean;
  /**
   * `'frame'` lighting: the environment as lights: `'outward'` (SH irradiance along each volume's outward direction),
   * `'directional'` (≤ 8 lights from a 64 × 32 map, default) or `'spherical-harmonics'` (SH quadrature lights).
   */
  environmentLighting: "outward" | "directional" | "spherical-harmonics";
  /** Lights of `'directional'` (1–8, default 8). */
  environmentSamples: number;
  /** SH order of `'spherical-harmonics'` (1 or 2, default 1). */
  environmentShOrder: number;
  /**
   * Multiple scattering in the shadow marches: false (single scattering), true (1D diffusion toward the light),
   * `'two-sided'` (2D slab diffusion toward and away from the light), `'octaves'` (indirect light octaves,
   * `scatteringOctave*`), `'multi-scale'` (diffusion from mips of the shadow grid, `multiScaleLevels`; 'frame'
   * lighting) or `'neural'` (default, his: the scattered light of the scene light slots from his small trained network
   * in `'frame'` lighting, diffusion elsewhere). `'neural'` and `'mlp'` (the same evaluator by its own name) are
   * EXPERIMENTAL: they run the generic small-MLP evaluator on `renderer.mlp` (null: 1D diffusion), whose data shape may
   * change before 1.0.
   */
  multipleScattering: VDBFrameMultipleScatteringMode;
  /** `'octaves'`: octaves N (1–8, default 4). Octave n has extinction × aⁿ, light × bⁿ, phase g × cⁿ. */
  scatteringOctaves: number;
  /** `'octaves'`: a (0–1, default 0.5). */
  scatteringOctaveExtinction: number;
  /** `'octaves'`: b (0–1, default 0.5). */
  scatteringOctaveContribution: number;
  /** `'octaves'`: c (0–1, default 0.5). */
  scatteringOctavePhase: number;
  /**
   * Spectral media (his `spectral`, default false): per-channel extinction (each volume's `extinctionScale` ×
   * (λ / 550)^−`angstromExponent`), per-channel albedo and multiple scattering. 'frame' lighting draws one
   * transmittance pass per slot group; the froxels get a second texture. `'neural'` / `'mlp'` run as 1D diffusion.
   */
  spectral: boolean;
  /**
   * Volumes cast shadows on meshes (his `castShadows`, default false), for directional, point and spot lights that ask
   * for a map: Beer shadow maps from the world shadow grid's last cascade, read by the host's receivers
   * (`vdbVolumeShadowWgsl`; `./three` multiplies them into each light's shadow).
   */
  castShadows: boolean;
  /**
   * Texels on each side of a volume shadow map (each cascade tile and cube face; default 512). Clamped (warned once)
   * to `maxTextureDimension2D` and to a map whose zero-fill staging fits `maxBufferSize` (a point light's 6 faces
   * count; K6.10a).
   */
  castShadowMapSize: number;
  /** Directional lights: cascades of the volume shadow map (1–4, default 1). */
  castShadowCascades: number;
  /** Directional lights: the largest view distance the cascades cover (default 4000). */
  castShadowDistance: number;
  /** Grid samples on each light ray of a volume shadow map (1–256, default 32). */
  castShadowSteps: number;
  /** Map pieces drawn again in one update (a cascade tile or spot map 1, a cube map 6; default 1). */
  castShadowUpdatesPerFrame: number;
  /** Very small maps (64) and few samples (8): soft blob shadows at almost no cost (default false). */
  castShadowCoarse: boolean;
  /** K6.3 `'multi-scale'`: levels read, base included (2–8; his `multiScaleLevels`, default 6). */
  multiScaleLevels: number;
  /**
   * K6.5b: the host's depth shadow maps (`VDBFrameInput.occluders`) shadow the volumes (mesh → volume shadows: the
   * 'frame' transmittance and gather, the 'volume' march). Default true; it does nothing without maps.
   */
  occlusion: boolean;
  /**
   * K6.Q1 (V85): shadowed ambient. Ambient, hemisphere and light-probe light (and 'frame' lighting's 'outward'
   * environment) is attenuated by the medium: transmittance marched along `ambientOcclusionPairs` opposite direction
   * pairs, averaged per hemisphere around the sky direction, each the lighting mode's isotropic-light total (white
   * furnace with multiple scattering). 'frame' lighting marches them in the transmittance pass (a grid lookup in the
   * gather); 'volume' lighting per froxel through the volume's shadow march. Default true (D35): the unshadowed light
   * of his renderer was +35 to +148 % too bright in single scattering (the K6.R reference). `false` gives his unshadowed
   * ambient, hemisphere and probe light.
   */
  ambientOcclusion: boolean;
  /** K6.Q1: direction pairs of the shadowed ambient (1–8; 1: ± the sky direction only). */
  ambientOcclusionPairs: number;
  /** K6.Q1: march steps of each 'volume'-lighting ambient direction (0: the volume's `shadowSteps`). */
  ambientOcclusionSteps: number;
}

/**
 * K6.8: the settings the renderer also reads that are not public in 0.2.0 (A/B settings of 'multi-scale'; K6.3a
 * recommends dropping multi-scale as a default candidate, the owner decides). The renderer's settings object has them
 * (`vdbFrameDefaultSettings()` makes them); set them through this type (`core/internal.ts`).
 */
export interface VDBFrameInternalSettings extends VDBFrameSettings {
  /**
   * K6.3 `'multi-scale'`: the mip edge rule (V75): `'zero'` (default; density outside the grid is 0) or his
   * `'clamp'` (clamp-to-edge). An A/B setting for K6.R1.
   */
  multiScaleEdge: "zero" | "clamp";
  /**
   * K6.3a `'multi-scale'`: the blur of each level (V75 item 5). `'his'` (default; his 4e2d46f): level k reads the
   * binomial mip k, σ ≈ 2^(k−1) voxels. `'paper'`: level k reads mip k + 1, σ ≈ 2^k voxels (Disney "Deep Scattering"
   * §4.2: the Gaussian width equals the level's stencil spacing); the chain holds one more mip. An A/B setting.
   */
  multiScaleSigma: "his" | "paper";
  /**
   * K6.Q1a: how 'frame' lighting (and 'volume' lighting with `ambientOcclusion`) reads the light probe along ± the sky
   * direction, the probe's share of the sky up / down terms. `'irradiance'` (default; his): three's cosine-weighted
   * irradiance there. `'radiance'`: π × the probe's mean radiance over that hemisphere (band 0 × 1, band 1 × ½, band 2 ×
   * 0), the weight isotropic in-scattering gives each hemisphere. An A/B setting (K6.Q1a handoff).
   */
  ambientProbe: "irradiance" | "radiance";
  /**
   * K6.Na: the evaluator of `'neural'` / `'mlp'` (EXPERIMENTAL). `'generic'` (default, K6.3): scalar loops over a flat
   * weight array. `'packed'`: the layer layout of the pmndrs neural tooling (pmndrs/denoiser `denoiser-wgsl`: mat4x4 per
   * output group and input chunk, `mat4x4f * vec4f`, vec4 activations), the same network to f32 rounding. An A/B setting
   * (K6-Na handoff); DECISIONS changes only after the A/B.
   */
  mlpRuntime: "generic" | "packed";
}

/** His v0.26 defaults (`FrameRenderer.js` L191–246, as `VDBFrameRenderer.settings`), plus `occlusion: true` and the K6.Q1 ambient occlusion (on since D35). */
export function vdbFrameDefaultSettings(): VDBFrameSettings {
  return vdbFrameDefaultInternalSettings();
}

/** The defaults with the internal fields typed ({@link VDBFrameInternalSettings}). */
export function vdbFrameDefaultInternalSettings(): VDBFrameInternalSettings {
  return {
    width: 512,
    height: 512,
    pixelsPerFroxel: 0,
    depth: 24,
    subSteps: 32,
    near: 0.1,
    far: Infinity,
    fadeRadius: Infinity,
    jitter: false,
    temporal: true,
    temporalBlend: 0.2,
    lighting: "frame",
    shadowResolution: 64,
    shadowCascades: 1,
    shadowDistance: 512,
    phase: "henyey-greenstein",
    phaseG: 0.8,
    phaseBackG: -0.3,
    phaseBackWeight: 0.2,
    phaseGExponent: 0.0,
    srgbBlending: true,
    environmentLighting: "directional",
    environmentSamples: 8,
    environmentShOrder: 1,
    multipleScattering: "neural",
    scatteringOctaves: 4,
    scatteringOctaveExtinction: 0.5,
    scatteringOctaveContribution: 0.5,
    scatteringOctavePhase: 0.5,
    spectral: false,
    castShadows: false,
    castShadowMapSize: 512,
    castShadowCascades: 1,
    castShadowDistance: 4000,
    castShadowSteps: 32,
    castShadowUpdatesPerFrame: 1,
    castShadowCoarse: false,
    multiScaleLevels: 6,
    multiScaleEdge: "zero",
    multiScaleSigma: "his",
    ambientProbe: "irradiance",
    mlpRuntime: "generic",
    occlusion: true,
    ambientOcclusion: true,
    ambientOcclusionPairs: 4,
    ambientOcclusionSteps: 0,
  };
}
