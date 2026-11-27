// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameShadowDensityMaterial.js L12–131 (the shadow density pass),
//   FrameTransmittanceMaterial.js L43–276 (light slots, gridExtinction, march, skyTotal, the near-extinction skip,
//   the slot packing), FrameDensityMaterial.js L172–357 (the grey density pass), FrameLightingMaterial.js L277–437
//   (the grey lighting pass); frameShaderChunks.js L1061–1194 (createGridNodes: sampleShadows).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: compute kernels on stacked 3D textures (K6.2, design §3.3, §3.5): the shadow grid and the transmittance
// are sub-volumes of one 3D texture each, read with hardware trilinear (his 2D slice atlases with a manual z lerp);
// the density and lighting passes are one kernel (the density sums live in registers while a thread loops its
// source group; his 4 rgba16float density targets are gone); several source groups accumulate in packed partials.
// The environment terms (K6.2b) read the GPU environment buffer: the dominant 'directional' light in the next free
// slot, the environment grid, or the 'outward' SH irradiance. K6.3: octaves and 'two-sided' (FrameTransmittanceMaterial
// L180–214, FrameDensityMaterial L272–294, FrameLightingMaterial L310–325); the grid mips and the network of 'neural'
// (FrameGridMipMaterial L10–66, frameNeuralChunks.js L14–129) run as a generic MLP evaluator (shapes as uniforms,
// weights in a storage buffer); K7.3's 'multi-scale' lookup (mjurczyk 4e2d46f). Spectral waits for K6.4. CPU twins:
// tools/frame (grid.ts, stack.ts, mlp.ts, multi-scale.ts). Divergences: DIVERGES.md.

/**
 * The 'frame' lighting kernels of the core renderer (K6.2):
 *
 * - `vdbFrameGridDensityCs` (per caster source group, 4×4×4): the extinction and absorption of the group's
 *   casters at each shadow grid voxel of every cascade (his shadow density pass, additive).
 * - `vdbFrameTransmittanceCs` (source-free, 4×4×4): from each voxel the march through the grid toward the light
 *   slots and the sky up and down; four sub-volumes per cascade (his 4 MRT targets).
 * - `vdbFrameGatherCs` 'frame' (per source group, 8×8×1): his density pass per volume (the slice pre-integration,
 *   the visible position, the shadow slots there) summed extinction-weighted in registers, then (last group) his
 *   lighting pass on the sums: premultiplied (radiance·α, α) into the scatter texture.
 */

import type { VDBVolumeChannels } from "../../source/types.js";
import { frameChunksWgsl } from "./chunks.js";
import { VDB_FRAME_SLICE_WGSL, vdbFrameSliceWgsl, vdbFrameSourceFunctionsWgsl } from "./gather.js";
import { vdbFrameMultiScaleWgsl } from "../../compute/multi-scale.js";
import { vdbFrameOccluderWgsl } from "./occluders.js";
import { VDB_FRAME_AMBIENT_MAX_PAIRS } from "../../../tools/frame/ambient.js";

/** Group-1 bindings of the shadow grid density kernel (K6.4 spectral: the absorption stack beside the extinction one). */
export const GRID_DENSITY_BINDINGS = { volumes: 0, group: 1, grid: 2, partials: 3, absorption: 4 } as const;
/** Group-1 bindings of the transmittance kernel (K6.4 spectral: the absorption stack). */
export const TRANSMITTANCE_BINDINGS = { grid: 0, transmittance: 1, absorption: 2 } as const;
/** Group-1 bindings of the 'frame' gather kernel (K6.4 spectral: the second scatter texture). */
export const FRAME_GATHER_BINDINGS = { volumes: 0, group: 1, scatter: 2, partials: 3, transmittance: 4, environmentGrid: 5, emissionLut: 6, scatterAlpha: 7 } as const;
/** K6.4: words per voxel of the spectral shadow grid's multi-group partials (extinction rgb, absorption rgb). */
export const VDB_FRAME_SPECTRAL_GRID_WORDS = 3;

/** Values per froxel the 'frame' gather sums (4 × vec4f), packed as 8 `pack2x16float` words in the partials. */
export const VDB_FRAME_SUM_WORDS = 8;

const GRID_CELL = `  let r = u32(vdbFrame.gridLayout.y);
  if (gid.x >= r || gid.y >= r || gid.z >= r * u32(vdbFrame.gridLayout.x)) {
    return;
  }
  let cascade = gid.z / r;
  let cell = vec3u(gid.xy, gid.z - cascade * r);
  let uvw = (vec3f(cell) + 0.5) / f32(r);
  let world = vdbFrame.gridMin[cascade].xyz + uvw * vdbFrame.gridSize[cascade].xyz;`;

/**
 * K6.4: the spectral shadow density pass (his FrameShadowDensityMaterial spectral branch, L111–121): two stacks, the
 * extinction (spectrum × extinction, rgb) and the absorption of each channel (extinction × (1 − albedo)); multi-group
 * partials hold 3 words per voxel at texel z, z + Z and z + 2Z of a partials texture 3× as deep (Z the stack depth).
 */
function vdbFrameSpectralGridDensityWgsl(o: { multi: boolean; channels: VDBVolumeChannels }): string {
  const b = GRID_DENSITY_BINDINGS;
  const store = o.multi
    ? `  let role = vdbFrameGroup.casters.z;
  let layer = vec3u(0u, 0u, textureDimensions(vdbFrameShadowGrid).z);
  if ((role & 1u) == 0u) {
    let w0 = unpack2x16float(textureLoad(vdbFrameGridPartials, texel).x);
    let w1 = unpack2x16float(textureLoad(vdbFrameGridPartials, texel + layer).x);
    let w2 = unpack2x16float(textureLoad(vdbFrameGridPartials, texel + 2u * layer).x);
    sumE += vec3f(w0, w1.x);
    sumA += vec3f(w1.y, w2);
  }
  if ((role & 2u) != 0u) {
    textureStore(vdbFrameShadowGrid, texel, vec4f(sumE, 0.0));
    textureStore(vdbFrameAbsorptionGrid, texel, vec4f(sumA, 0.0));
  } else {
    textureStore(vdbFrameGridPartials, texel, vec4u(pack2x16float(sumE.xy), 0u, 0u, 0u));
    textureStore(vdbFrameGridPartials, texel + layer, vec4u(pack2x16float(vec2f(sumE.z, sumA.x)), 0u, 0u, 0u));
    textureStore(vdbFrameGridPartials, texel + 2u * layer, vec4u(pack2x16float(sumA.yz), 0u, 0u, 0u));
  }`
    : `  textureStore(vdbFrameShadowGrid, texel, vec4f(sumE, 0.0));
  textureStore(vdbFrameAbsorptionGrid, texel, vec4f(sumA, 0.0));`;
  return `${frameChunksWgsl(["constants", "mat3", "froxel", "lod", "shadow"])}
@group(1) @binding(${b.volumes}) var<uniform> vdbFrameVolumes: VdbFrameVolumes;
@group(1) @binding(${b.group}) var<uniform> vdbFrameGroup: VdbFrameGroup;
@group(1) @binding(${b.grid}) var vdbFrameShadowGrid: texture_storage_3d<rgba16float, write>;
@group(1) @binding(${b.absorption}) var vdbFrameAbsorptionGrid: texture_storage_3d<rgba16float, write>;
${o.multi ? `@group(1) @binding(${b.partials}) var vdbFrameGridPartials: texture_storage_3d<r32uint, read_write>;\n` : ""}
${vdbFrameSourceFunctionsWgsl(o.channels)}
// His spectral shadow density pass at one voxel: the summed extinction and absorption of each channel of the group's
// casters whose box holds the voxel centre, read at the voxel's lod (absorbance scales the shadow, not the look).
@compute @workgroup_size(4, 4, 4)
fn vdbFrameGridDensityCs(@builtin(global_invocation_id) gid: vec3u) {
${GRID_CELL}
  let voxel = vdbFrame.gridVoxel[cascade].xyz;
  let voxelSize = max(voxel.x, max(voxel.y, voxel.z));
  var sumE = vec3f(0.0);
  var sumA = vec3f(0.0);
  let casters = vdbFrameGroup.casters;
  for (var i = 0u; i < casters.y; i++) {
    let vi = casters.x + i;
    let local = (vdbFrameVolumes.v[vi].worldToLocal * vec4f(world, 1.0)).xyz;
    let a = abs(local);
    if (max(a.x, max(a.y, a.z)) > 0.5) {
      continue;
    }
    let texels = vdbSrcTexels(vdbFrameVolumes.v[vi].slices.z);
    let maxTexels = max(texels.x, max(texels.y, texels.z));
    let lod = vdbFrameLod(voxelSize / vdbFrameVolumes.v[vi].density.z, maxTexels);
    let readLod = select(0.0, lod, vdbFrameVolumes.v[vi].march.y != 0u);
    let extinction = vdbFrameExtinction(vi, local, readLod) * vdbFrameVolumes.v[vi].density.w;
    if (extinction <= 0.0) {
      continue;
    }
    let spectral = vdbFrameVolumes.v[vi].extinctionSpectrum.xyz * extinction;
    sumE += spectral;
    sumA += spectral * (vec3f(1.0) - clamp(vdbFrameAlbedo(vi, local, readLod), vec3f(0.0), vec3f(1.0)));
  }
  let texel = vec3u(vdbFrame.shadowOrigin[cascade].xyz) + cell;
${store}
}
`;
}

/** The shadow grid density kernel (after the source module, the source WGSL and `vdbFrameUniformsWgsl()`). */
export function vdbFrameGridDensityWgsl(o: { multi: boolean; channels: VDBVolumeChannels; spectral?: boolean }): string {
  if (o.spectral) return vdbFrameSpectralGridDensityWgsl(o);
  const b = GRID_DENSITY_BINDINGS;
  const store = o.multi
    ? `  let role = vdbFrameGroup.casters.z;
  if ((role & 1u) == 0u) {
    sum += unpack2x16float(textureLoad(vdbFrameGridPartials, texel).x);
  }
  if ((role & 2u) != 0u) {
    textureStore(vdbFrameShadowGrid, texel, vec4f(sum, 0.0, 0.0));
  } else {
    textureStore(vdbFrameGridPartials, texel, vec4u(pack2x16float(sum), 0u, 0u, 0u));
  }`
    : `  textureStore(vdbFrameShadowGrid, texel, vec4f(sum, 0.0, 0.0));`;
  return `${frameChunksWgsl(["constants", "mat3", "froxel", "lod", "shadow"])}
@group(1) @binding(${b.volumes}) var<uniform> vdbFrameVolumes: VdbFrameVolumes;
@group(1) @binding(${b.group}) var<uniform> vdbFrameGroup: VdbFrameGroup;
@group(1) @binding(${b.grid}) var vdbFrameShadowGrid: texture_storage_3d<rgba16float, write>;
${o.multi ? `@group(1) @binding(${b.partials}) var vdbFrameGridPartials: texture_storage_3d<r32uint, read_write>;\n` : ""}
${vdbFrameSourceFunctionsWgsl(o.channels)}
// His shadow density pass at one voxel: the summed (extinction, absorption) of the group's casters whose box holds
// the voxel centre, read at the voxel's lod (absorbance scales the shadow, not the look).
@compute @workgroup_size(4, 4, 4)
fn vdbFrameGridDensityCs(@builtin(global_invocation_id) gid: vec3u) {
${GRID_CELL}
  let voxel = vdbFrame.gridVoxel[cascade].xyz;
  let voxelSize = max(voxel.x, max(voxel.y, voxel.z));
  var sum = vec2f(0.0);
  let casters = vdbFrameGroup.casters;
  for (var i = 0u; i < casters.y; i++) {
    let vi = casters.x + i;
    let local = (vdbFrameVolumes.v[vi].worldToLocal * vec4f(world, 1.0)).xyz;
    let a = abs(local);
    if (max(a.x, max(a.y, a.z)) > 0.5) {
      continue;
    }
    let texels = vdbSrcTexels(vdbFrameVolumes.v[vi].slices.z);
    let maxTexels = max(texels.x, max(texels.y, texels.z));
    let lod = vdbFrameLod(voxelSize / vdbFrameVolumes.v[vi].density.z, maxTexels);
    let readLod = select(0.0, lod, vdbFrameVolumes.v[vi].march.y != 0u);
    let extinction = vdbFrameExtinction(vi, local, readLod) * vdbFrameVolumes.v[vi].density.w;
    if (extinction <= 0.0) {
      continue;
    }
    sum += vec2f(extinction, vdbFrameShadowAbsorption(extinction, vdbFrameAlbedo(vi, local, readLod)));
  }
  let texel = vec3u(vdbFrame.shadowOrigin[cascade].xyz) + cell;
${store}
}
`;
}

/**
 * K6.3: the scattered-light term of the transmittance kernel's light slots, a pipeline key (design §3.6, §5.2). `'none'`
 * covers single scattering, 1D diffusion, 'two-sided' and 'octaves' (uniforms); `'mlp'` and `'multi-scale'` bind their
 * resources in group 2 (the kernel is source-free: the MLP weights are its one storage buffer, D10).
 */
export type VDBFrameScatteringKernel = "none" | "mlp" | "multi-scale";

/** K6.3: group-2 bindings of the transmittance kernel's `'mlp'` and `'multi-scale'` variants. */
export const TRANSMITTANCE_MLP_BINDINGS = { weights: 0, shape: 1, mips: 2 } as const;
export const TRANSMITTANCE_MULTI_SCALE_BINDINGS = { params: 0, mips: 1 } as const;

/** Most MLP layers and the widest layer the generic evaluator takes (K6.3, design §3.6). */
export const VDB_FRAME_MLP_MAX_LAYERS = 8;
export const VDB_FRAME_MLP_MAX_WIDTH = 64;

/**
 * K6.3: the experimental generic MLP evaluator (design §3.6, Q5) and its inputs, his `'neural'` features (frameNeuralChunks
 * L60–129): log(1 + σ h 2^k) of the feature levels at the voxel, toward the light and away from it, log(1 + direct optical
 * depth) and √(absorption / extinction) of level 1. Level 0 is the last shadow grid cascade, level k ≥ 1 the box mip
 * k − 1 of `vdbFrameNeuralMips` (the last mip past the chain). The shape is a uniform, the weights a storage buffer.
 */
const MLP_WGSL = `struct VdbFrameMlp {
  // layer count, feature levels, output (0 linear, 1 exp(min(o, 4))), input count
  info: vec4u,
  // per layer: inputs, outputs, activation (0 linear, 1 ReLU), weight offset (outputs x inputs row-major, then the biases)
  layers: array<vec4u, ${VDB_FRAME_MLP_MAX_LAYERS}>,
};
@group(2) @binding(${TRANSMITTANCE_MLP_BINDINGS.weights}) var<storage, read> vdbFrameMlpWeights: array<f32>;
@group(2) @binding(${TRANSMITTANCE_MLP_BINDINGS.shape}) var<uniform> vdbFrameMlp: VdbFrameMlp;
@group(2) @binding(${TRANSMITTANCE_MLP_BINDINGS.mips}) var vdbFrameNeuralMips: texture_3d<f32>;

// The forward pass: each layer's outputs from the last one's (two ${VDB_FRAME_MLP_MAX_WIDTH}-wide locals), then the output transform
fn vdbFrameMlpForward(input: array<f32, ${VDB_FRAME_MLP_MAX_WIDTH}>) -> f32 {
  var a = input;
  var b: array<f32, ${VDB_FRAME_MLP_MAX_WIDTH}>;
  let count = min(vdbFrameMlp.info.x, ${VDB_FRAME_MLP_MAX_LAYERS}u);
  for (var l = 0u; l < count; l++) {
    let shape = vdbFrameMlp.layers[l];
    let n = min(shape.x, ${VDB_FRAME_MLP_MAX_WIDTH}u);
    let m = min(shape.y, ${VDB_FRAME_MLP_MAX_WIDTH}u);
    let bias = shape.w + shape.y * shape.x;
    for (var j = 0u; j < m; j++) {
      var s = vdbFrameMlpWeights[bias + j];
      let row = shape.w + j * shape.x;
      for (var i = 0u; i < n; i++) {
        s += vdbFrameMlpWeights[row + i] * a[i];
      }
      b[j] = select(s, max(s, 0.0), shape.z == 1u);
    }
    a = b;
  }
  return select(a[0], exp(min(a[0], 4.0)), vdbFrameMlp.info.z == 1u);
}

// sampleLevel: (extinction, absorption) of feature level k at a world position, 0 outside the last cascade
fn vdbFrameNeuralSample(level: u32, world: vec3f) -> vec2f {
  let cascade = u32(vdbFrame.gridLayout.x) - 1u;
  let uvw = vdbFrameGridUvw(cascade, world);
  if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) {
    return vec2f(0.0);
  }
  if (level == 0u) {
    return vdbFrameStackSample(vdbFrameShadowGrid, vdbFrame.shadowOrigin[cascade].xyz, uvw).xy;
  }
  let mip = min(level - 1u, textureNumLevels(vdbFrameNeuralMips) - 1u);
  let dims = vec3f(textureDimensions(vdbFrameNeuralMips, mip));
  let p = clamp(uvw * dims, vec3f(0.5), dims - vec3f(0.5));
  return textureSampleLevel(vdbFrameNeuralMips, vdbFrameLinear, p / dims, f32(mip)).xy;
}

// h: the largest voxel side of the last cascade
fn vdbFrameNeuralUnit() -> f32 {
  let v = vdbFrame.gridVoxel[u32(vdbFrame.gridLayout.x) - 1u].xyz;
  return max(v.x, max(v.y, v.z));
}

// getLocalInputs: the inputs that do not depend on the light (the voxel's levels at [0, L), the albedo input at 3L + 1)
fn vdbFrameMlpLocal(world: vec3f) -> array<f32, ${VDB_FRAME_MLP_MAX_WIDTH}> {
  var input: array<f32, ${VDB_FRAME_MLP_MAX_WIDTH}>;
  let levels = min(vdbFrameMlp.info.y, 21u);
  let h = vdbFrameNeuralUnit();
  for (var k = 0u; k < levels; k++) {
    let s = vdbFrameNeuralSample(k, world);
    input[k] = log(s.x * (h * f32(1u << k)) + 1.0);
    if (k == 1u && s.x > 0.0) {
      input[3u * levels + 1u] = sqrt(clamp(s.y / max(s.x, 1e-20), 0.0, 1.0));
    }
  }
  return input;
}

// neuralLight: the network's scattered light (total - direct) of a slot for the unit direction to the light
fn vdbFrameMlpLight(world: vec3f, direction: vec3f, opticalDepth: f32, local: array<f32, ${VDB_FRAME_MLP_MAX_WIDTH}>) -> f32 {
  var input = local;
  let levels = min(vdbFrameMlp.info.y, 21u);
  let h = vdbFrameNeuralUnit();
  for (var k = 0u; k < levels; k++) {
    let reach = h * f32(1u << k);
    input[levels + k] = log(vdbFrameNeuralSample(k, world + direction * reach).x * reach + 1.0);
    input[2u * levels + k] = log(vdbFrameNeuralSample(k, world - direction * reach).x * reach + 1.0);
  }
  input[3u * levels] = log(opticalDepth + 1.0);
  return vdbFrameMlpForward(input);
}
`;

/** K6.3: the box mip kernels of the `'mlp'` feature levels (his grid mip pass, design §3.3 `gridMips`). */
export const GRID_MIP_BINDINGS = { source: 0, out: 1 } as const;

/**
 * K6.3 `gridMips` (his FrameGridMipMaterial, the box branch): texel x of mip k is the mean of the 2 × 2 × 2 texels 2x + i
 * of level k − 1, each clamped to that level's last texel. `vdbFrameGridMipFirstCs` reads the last shadow grid cascade
 * in the stack (level 0), `vdbFrameGridMipCs` a one-mip view of the level before. rgba16float (his RG half float in .xy).
 */
export function vdbFrameGridMipWgsl(): string {
  const b = GRID_MIP_BINDINGS;
  return `@group(1) @binding(${b.source}) var vdbFrameMipSource: texture_3d<f32>;
@group(1) @binding(${b.out}) var vdbFrameMipOut: texture_storage_3d<rgba16float, write>;

fn vdbFrameMipBox(gid: vec3u, origin: vec3u, size: vec3u) {
  if (any(gid >= textureDimensions(vdbFrameMipOut))) {
    return;
  }
  var sum = vec4f(0.0);
  for (var i = 0u; i < 8u; i++) {
    let p = min(gid * 2u + vec3u(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u), size - vec3u(1u));
    sum += textureLoad(vdbFrameMipSource, origin + p, 0);
  }
  textureStore(vdbFrameMipOut, gid, vec4f(sum.xy / 8.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn vdbFrameGridMipFirstCs(@builtin(global_invocation_id) gid: vec3u) {
  let cascade = u32(vdbFrame.gridLayout.x) - 1u;
  vdbFrameMipBox(gid, vec3u(vdbFrame.shadowOrigin[cascade].xyz), vec3u(u32(vdbFrame.gridLayout.y)));
}

@compute @workgroup_size(4, 4, 4)
fn vdbFrameGridMipCs(@builtin(global_invocation_id) gid: vec3u) {
  vdbFrameMipBox(gid, vec3u(0u), textureDimensions(vdbFrameMipSource));
}
`;
}

/**
 * K6.4: the spectral grid sample and march (his FrameTransmittanceMaterial buildSpectralNodes: gridSample, march): the
 * grey march with the depths of each channel, stopping when every channel is past march depth 6. Needs the
 * `vdbFrameShadowGrid` (extinction rgb) and `vdbFrameAbsorptionGrid` (absorption rgb) stacks.
 */
const SPECTRAL_GRID_MARCH_WGSL = `struct VdbFrameSpectralSample {
  e: vec3f,
  a: vec3f,
};
// (extinction, absorption) of each channel of the finest cascade at the position, 0 outside the grid
fn vdbFrameSpectralGridSample(world: vec3f) -> VdbFrameSpectralSample {
  let c = vdbFrameFindCascade(world, 0u);
  if (c < 0) {
    return VdbFrameSpectralSample(vec3f(0.0), vec3f(0.0));
  }
  let cascade = u32(c);
  let origin = vdbFrame.shadowOrigin[cascade].xyz;
  let uvw = vdbFrameGridUvw(cascade, world);
  return VdbFrameSpectralSample(vdbFrameStackSample(vdbFrameShadowGrid, origin, uvw).xyz, vdbFrameStackSample(vdbFrameAbsorptionGrid, origin, uvw).xyz);
}
fn vdbFrameSpectralGridMarch(origin: vec3f, direction: vec3f, maxDistance: f32, jitter: f32) -> VdbFrameSpectralMarch {
  let ms = vdbFrame.lighting.x;
  let octaves = vdbFrame.octaves;
  var d = VdbFrameSpectralDepths(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var traveled = 0.0;
  var probe = origin;
  for (var i = 0u; i < 1024u; i++) {
    let c = vdbFrameFindCascade(probe, 0u);
    if (c < 0 || traveled >= maxDistance || vdbFrameMinComponent(vdbFrameSpectralShadowMarchDepth(d, ms, octaves)) > 6.0) {
      break;
    }
    let perUnit = abs(direction) / vdbFrame.gridVoxel[u32(c)].xyz;
    let stepLength = min(1.0 / max(perUnit.x, max(perUnit.y, perUnit.z)), maxDistance - traveled);
    let s = vdbFrameSpectralGridSample(probe + direction * stepLength * jitter);
    let st = vdbFrameSpectralShadowStep(s.e, s.a, ms);
    d.x += st.x * stepLength;
    d.y += st.y * stepLength;
    d.z += st.z * stepLength;
    probe += direction * stepLength;
    traveled += stepLength;
  }
  let t = vdbFrameSpectralShadowTransmittance(d, ms, octaves);
  return VdbFrameSpectralMarch(t.x, t.y, d.z, d.y);
}
`;

/**
 * K6.4: the spectral transmittance kernel (his buildSpectralNodes, one pass per slot group): one dispatch over the
 * cascades × the slot groups in use (`VdbFrameUniforms.spectral.y`). Group 0 holds (sky up, sky down) and slot 0, group g
 * slots 2g − 1 and 2g; a slot entry is (total, direct) of each channel, packed into 3 targets per group and cascade at
 * `transmittanceOrigin[3 (g C + c) + k]`. 'two-sided' and the K6.5b occluder maps as the grey kernel, per channel.
 */
function vdbFrameSpectralTransmittanceWgsl(): string {
  const b = TRANSMITTANCE_BINDINGS;
  return `${frameChunksWgsl(["constants", "noise", "shadow", "spectral", "grid"])}
@group(1) @binding(${b.grid}) var vdbFrameShadowGrid: texture_3d<f32>;
@group(1) @binding(${b.absorption}) var vdbFrameAbsorptionGrid: texture_3d<f32>;
@group(1) @binding(${b.transmittance}) var vdbFrameTransmittanceOut: texture_storage_3d<rgba16float, write>;
${vdbFrameOccluderWgsl()}
${SPECTRAL_GRID_MARCH_WGSL}
// The unit direction to light slot i and its march length (w < 0: no slot); slot = scene slots: the environment light
fn vdbFrameSlotDirection(i: u32, world: vec3f) -> vec4f {
  let slots = u32(vdbFrame.gridLayout.w);
  let envSlot = vdbFrame.environment.x > 0.5;
  if (i >= 6u || i >= slots + select(0u, 1u, envSlot)) {
    return vec4f(0.0, 1.0, 0.0, -1.0);
  }
  if (i == slots) {
    if (vdbFrameEnv.info.z != 0u && vdbFrameEnv.info.x != 0u) {
      return vec4f(normalize(vdbFrameEnv.direction[0].xyz), 1e20);
    }
    return vec4f(0.0, 1.0, 0.0, -1.0);
  }
  let light = vdbFrameLights.position[i];
  let toLight = light.xyz - world;
  let isDirection = light.w < 1.5;
  return vec4f(select(normalize(toLight), normalize(light.xyz), isDirection), select(length(toLight), 1e20, isDirection));
}

@compute @workgroup_size(4, 4, 4)
fn vdbFrameTransmittanceCs(@builtin(global_invocation_id) gid: vec3u) {
  let r = u32(vdbFrame.gridLayout.y);
  let cascades = u32(vdbFrame.gridLayout.x);
  let groups = u32(vdbFrame.spectral.y);
  if (gid.x >= r || gid.y >= r || gid.z >= r * cascades * groups) {
    return;
  }
  let layer = gid.z / r;
  let cascade = layer % cascades;
  let group = layer / cascades;
  let cell = vec3u(gid.xy, gid.z - layer * r);
  let uvw = (vec3f(cell) + 0.5) / f32(r);
  let world = vdbFrame.gridMin[cascade].xyz + uvw * vdbFrame.gridSize[cascade].xyz;
  var t0 = vec4f(1.0);
  var t1 = vec4f(1.0);
  var t2 = vec4f(1.0);
  let voxel = vdbFrame.gridVoxel[cascade].xyz;
  var near = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let corner = vec3f(f32(i & 1u), f32((i >> 1u) & 1u), f32((i >> 2u) & 1u)) - 0.5;
    near += dot(vdbFrameSpectralGridSample(world + corner * voxel).e, vec3f(1.0));
  }
  if (near > 0.0) {
    let ms = vdbFrame.lighting.x;
    let twoSided = vdbFrame.lighting.y > 0.5;
    let jitter = vdbFrameNoise(vdbFrame.jitterFade.x, vdbFrameGridFragCoord(cascade, cell));
    let sky = normalize(vdbFrameLights.sky.xyz);
    let isSky = group == 0u;
    // the two entries' slots (group 0: entry a is the sky, b slot 0)
    let slotA = select(2u * group - 1u, 0u, isSky);
    let slotB = select(2u * group, 0u, isSky);
    let a = select(vdbFrameSlotDirection(slotA, world), vec4f(sky, 1e20), isSky);
    let bDir = vdbFrameSlotDirection(slotB, world);
    // marches 0, 1: the entries toward the light (or sky up); 2, 3: away from it (sky down; 'two-sided' backs)
    var directions = array<vec4f, 4>(a, bDir, vec4f(-a.xyz, select(-1.0, 1e20, a.w >= 0.0 && (isSky || twoSided))), vec4f(-bDir.xyz, select(-1.0, 1e20, bDir.w >= 0.0 && twoSided)));
    var results: array<VdbFrameSpectralMarch, 4>;
    for (var m = 0u; m < 4u; m++) {
      results[m] = vdbFrameSpectralUnmarched();
      let d = directions[m];
      if (d.w < 0.0) {
        continue;
      }
      results[m] = vdbFrameSpectralGridMarch(world, d.xyz, d.w, jitter);
    }
    // the slot entries: two-sided totals, then (scene slots) the host occluder visibility of the scattered part
    for (var e = 0u; e < 2u; e++) {
      if ((e == 0u && isSky) || directions[e].w < 0.0) {
        continue;
      }
      let slot = select(slotA, slotB, e == 1u);
      if (twoSided) {
        results[e].y = vdbFrameTwoSidedDiffusion3(results[e].x, results[e].z, results[e].w, results[e + 2u].z);
      }
      if (slot < u32(vdbFrame.gridLayout.w) && vdbFrame.octaves.x <= 0.0) {
        let visibility = vdbFrameOccluderVisibility(world, slot);
        if (visibility < 1.0) {
          results[e].y = results[e].x + (results[e].y - results[e].x) * visibility;
        }
      }
    }
    var ax = results[0].y;
    var ay = results[0].x;
    if (isSky) {
      let up = results[0];
      let down = results[2];
      ax = select(up.y, vdbFrameIsotropicDiffusion3(up.z, down.z, up.w), ms > 0.0);
      ay = select(down.y, vdbFrameIsotropicDiffusion3(down.z, up.z, down.w), ms > 0.0);
      // K6.Q1a shadowed ambient: the hemispherical visibilities of each channel (V up, V down) replace the sky totals
      // (the gather reads them where it read the totals); pair 0 is the sky marches. Octaves: each direction's spectral
      // octave total over an open direction's, at the voxel's grid albedo (as the environment grid)
      let ambientPairs = min(vdbFrameLights.counts.z, ${VDB_FRAME_AMBIENT_MAX_PAIRS}u);
      if (ambientPairs > 0u) {
        let octaves = vdbFrame.octaves;
        let origin = vdbFrame.shadowOrigin[cascade].xyz;
        let albedo = vdbFrameSpectralGridAlbedo(vdbFrameStackSample(vdbFrameShadowGrid, origin, uvw).xyz, vdbFrameStackSample(vdbFrameAbsorptionGrid, origin, uvw).xyz);
        let open = vdbFrameSpectralShadowTotal(vec3f(1.0), octaves, albedo);
        var ambientUp = select(vdbFrameSpectralShadowTotal(up.y, octaves, albedo) / open, ax, ms > 0.0);
        var ambientDown = select(vdbFrameSpectralShadowTotal(down.y, octaves, albedo) / open, ay, ms > 0.0);
        // the other pairs: march 2(k − 1) toward ω_k, 2(k − 1) + 1 away (one march call site)
        var front = vdbFrameSpectralUnmarched();
        for (var m = 0u; m < 2u * (ambientPairs - 1u); m++) {
          let d = vdbFrameLights.ambientDirections[m / 2u + 1u].xyz;
          let marched = vdbFrameSpectralGridMarch(world, select(d, -d, (m & 1u) == 1u), 1e20, jitter);
          if ((m & 1u) == 0u) {
            front = marched;
            continue;
          }
          if (ms > 0.0) {
            ambientUp += vdbFrameIsotropicDiffusion3(front.z, marched.z, front.w);
            ambientDown += vdbFrameIsotropicDiffusion3(marched.z, front.z, marched.w);
          } else {
            ambientUp += vdbFrameSpectralShadowTotal(front.y, octaves, albedo) / open;
            ambientDown += vdbFrameSpectralShadowTotal(marched.y, octaves, albedo) / open;
          }
        }
        ax = ambientUp / f32(ambientPairs);
        ay = ambientDown / f32(ambientPairs);
      }
    }
    let bx = results[1].y;
    let by = results[1].x;
    t0 = vec4f(ax, ay.x);
    t1 = vec4f(ay.y, ay.z, bx.x, bx.y);
    t2 = vec4f(bx.z, by);
  }
  let base = 3u * (group * cascades + cascade);
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base].xyz) + cell, t0);
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base + 1u].xyz) + cell, t1);
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base + 2u].xyz) + cell, t2);
}
`;
}

/** The transmittance kernel (after `vdbFrameUniformsWgsl()`; source-free). K6.3: `scattering` picks the slot term. */
export function vdbFrameTransmittanceWgsl(o: { scattering?: VDBFrameScatteringKernel; spectral?: boolean } = {}): string {
  if (o.spectral) return vdbFrameSpectralTransmittanceWgsl();
  const b = TRANSMITTANCE_BINDINGS;
  const scattering = o.scattering ?? "none";
  const msb = TRANSMITTANCE_MULTI_SCALE_BINDINGS;
  const extra =
    scattering === "mlp"
      ? MLP_WGSL
      : scattering === "multi-scale"
        ? `${vdbFrameMultiScaleWgsl({ baseChannels: [0, 1], mipChannels: 2 })}
@group(2) @binding(${msb.params}) var<uniform> vdbFrameMsParams: VdbFrameMultiScale;
@group(2) @binding(${msb.mips}) var vdbFrameMsMips: texture_3d<f32>;
`
        : "";
  const local = scattering === "mlp" ? `    let local = vdbFrameMlpLocal(world);\n` : "";
  const slotTerm =
    scattering === "mlp"
      ? `      // 'mlp' (experimental; his 'neural'): total = direct + the network's scattered light
      results[i].y = results[i].x + vdbFrameMlpLight(world, s.xyz, opticalDepths[i], local);\n`
      : scattering === "multi-scale"
        ? `      // 'multi-scale' (K7.3): total = direct + the multi-scale scattered light, at most 1
      results[i].y = min(results[i].x + vdbFrameMultiScaleLight(vdbFrameShadowGrid, vdbFrameMsMips, vdbFrameLinear, vdbFrameMsParams, world, s.xyz, s.w), 1.0);\n`
        : "";
  return `${frameChunksWgsl(["constants", "noise", "shadow", "octaveTotal", "grid"])}
@group(1) @binding(${b.grid}) var vdbFrameShadowGrid: texture_3d<f32>;
@group(1) @binding(${b.transmittance}) var vdbFrameTransmittanceOut: texture_storage_3d<rgba16float, write>;
${extra}
${vdbFrameOccluderWgsl()}
// gridExtinction: (extinction, absorption) of the finest cascade at the position, 0 outside the grid
fn vdbFrameGridExtinction(world: vec3f) -> vec2f {
  let c = vdbFrameFindCascade(world, 0u);
  if (c < 0) {
    return vec2f(0.0);
  }
  let cascade = u32(c);
  return vdbFrameStackSample(vdbFrameShadowGrid, vdbFrame.shadowOrigin[cascade].xyz, vdbFrameGridUvw(cascade, world)).xy;
}

// His march: one voxel of the finest cascade at the probe per step on the direction's fastest axis (coarser
// cascades after leaving a fine one), each step sampled jitter of the way along it; stops outside the grid, at
// maxDistance or past march depth 6 (K6.3: octaves march to the last octave's depth). t: (direct, total, reduced depth,
// effective depth); depth: the direct optical depth (the MLP's input).
struct VdbFrameGridMarchResult {
  t: vec4f,
  depth: f32,
};
fn vdbFrameGridMarch(origin: vec3f, direction: vec3f, maxDistance: f32, jitter: f32) -> VdbFrameGridMarchResult {
  let ms = vdbFrame.lighting.x;
  let octaves = vdbFrame.octaves;
  var depths = vec3f(0.0);
  var traveled = 0.0;
  var probe = origin;
  for (var i = 0u; i < 1024u; i++) {
    let c = vdbFrameFindCascade(probe, 0u);
    if (c < 0 || traveled >= maxDistance || vdbFrameShadowMarchDepth(depths, ms, octaves) > 6.0) {
      break;
    }
    let perUnit = abs(direction) / vdbFrame.gridVoxel[u32(c)].xyz;
    let stepLength = min(1.0 / max(perUnit.x, max(perUnit.y, perUnit.z)), maxDistance - traveled);
    let s = vdbFrameGridExtinction(probe + direction * stepLength * jitter);
    depths += vdbFrameShadowStep(s.x, s.y, ms) * stepLength;
    probe += direction * stepLength;
    traveled += stepLength;
  }
  let t = vdbFrameShadowTransmittance(depths, ms, octaves);
  return VdbFrameGridMarchResult(vec4f(t.x, t.y, depths.z, depths.y), depths.x);
}

// His transmittance pass at one voxel: 4 targets (sky up total, sky down total, slot 0, 1 total), (slots 2–5
// total), (slots 0–3 direct), (slots 4–5 direct, K6.Q1 ambient V up, V down (1: off)); 1 where no extinction is within
// half a voxel.
@compute @workgroup_size(4, 4, 4)
fn vdbFrameTransmittanceCs(@builtin(global_invocation_id) gid: vec3u) {
${GRID_CELL}
  var t0 = vec4f(1.0);
  var t1 = vec4f(1.0);
  var t2 = vec4f(1.0);
  var t3 = vec4f(1.0);
  let voxel = vdbFrame.gridVoxel[cascade].xyz;
  var near = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let corner = vec3f(f32(i & 1u), f32((i >> 1u) & 1u), f32((i >> 2u) & 1u)) - 0.5;
    near += vdbFrameGridExtinction(world + corner * voxel).x;
  }
  if (near > 0.0) {
    let ms = vdbFrame.lighting.x;
    let twoSided = vdbFrame.lighting.y > 0.5;
    let jitter = vdbFrameNoise(vdbFrame.jitterFade.x, vdbFrameGridFragCoord(cascade, cell));
    let sky = normalize(vdbFrameLights.sky.xyz);
    let slots = u32(vdbFrame.gridLayout.w);
    // K6.2b: the dominant 'directional' environment light takes the next free slot (its direction from the buffer)
    let envSlot = vdbFrame.environment.x > 0.5;
    // the slots' unit directions to the light and march lengths (w < 0: no slot)
    var slotDirection: array<vec4f, 6>;
    for (var i = 0u; i < 6u; i++) {
      slotDirection[i] = vec4f(0.0, 1.0, 0.0, -1.0);
      if (i >= slots + select(0u, 1u, envSlot)) {
        continue;
      }
      if (i == slots) {
        if (vdbFrameEnv.info.z != 0u && vdbFrameEnv.info.x != 0u) {
          slotDirection[i] = vec4f(normalize(vdbFrameEnv.direction[0].xyz), 1e20);
        }
      } else {
        let light = vdbFrameLights.position[i];
        let toLight = light.xyz - world;
        let isDirection = light.w < 1.5;
        slotDirection[i] = vec4f(select(normalize(toLight), normalize(light.xyz), isDirection), select(length(toLight), 1e20, isDirection));
      }
    }
    // marches 0..5: the light slots; 6, 7: sky up and down; K6.3 'two-sided' 8..13: the slots away from the light
    // (one march call site)
    var results: array<vec4f, 14>;
    var opticalDepths: array<f32, 6>;
    let marches = select(8u, 14u, twoSided);
    for (var m = 0u; m < marches; m++) {
      var direction = sky;
      var maxDistance = 1e20;
      results[m] = vec4f(1.0, 1.0, 0.0, 0.0);
      if (m < 6u || m >= 8u) {
        let s = slotDirection[select(m, m - 8u, m >= 8u)];
        if (s.w < 0.0) {
          continue;
        }
        direction = select(s.xyz, -s.xyz, m >= 8u);
        maxDistance = select(s.w, 1e20, m >= 8u);
      } else if (m == 7u) {
        direction = -sky;
      }
      let r = vdbFrameGridMarch(world, direction, maxDistance, jitter);
      results[m] = r.t;
      if (m < 6u) {
        opticalDepths[m] = r.depth;
      }
    }
${local}    for (var i = 0u; i < 6u; i++) {
      let s = slotDirection[i];
      if (s.w < 0.0) {
        continue;
      }
      if (twoSided) {
        // 'two-sided': diffusion of the slab between the light side and the far side
        results[i].y = vdbFrameTwoSidedDiffusion(results[i].x, results[i].z, results[i].w, results[i + 8u].z);
      }
${slotTerm}      // K6.5b: a host occluder map scales the scene slot's scattered part (total − direct) by the voxel's visibility;
      // the direct part stays unoccluded here (the gather multiplies it by the visibility at its finer position). K6.3:
      // after the mode's own total (two-sided, the MLP, multi-scale); octaves keep 1 / (1 + depth) here (the gather
      // occludes their light at the visible position)
      if (i < slots && vdbFrame.octaves.x <= 0.0) {
        let visibility = vdbFrameOccluderVisibility(world, i);
        if (visibility < 1.0) {
          results[i].y = results[i].x + (results[i].y - results[i].x) * visibility;
        }
      }
    }
    let up = results[6];
    let down = results[7];
    var skyUp = up.y;
    var skyDown = down.y;
    if (ms > 0.0) {
      skyUp = vdbFrameIsotropicDiffusion(up.z, down.z, up.w);
      skyDown = vdbFrameIsotropicDiffusion(down.z, up.z, down.w);
    }
    // K6.Q1 shadowed ambient: the hemispherical visibilities (V up, V down), the mean isotropic-light total of each
    // pair's two sides (pair 0: the sky up / down marches above; octaves: their octave totals)
    var ambientUp = 1.0;
    var ambientDown = 1.0;
    let ambientPairs = min(vdbFrameLights.counts.z, ${VDB_FRAME_AMBIENT_MAX_PAIRS}u);
    if (ambientPairs > 0u) {
      ambientUp = select(vdbFrameShadowTotal(skyUp, vdbFrame.octaves) / vdbFrameShadowTotal(1.0, vdbFrame.octaves), skyUp, ms > 0.0);
      ambientDown = select(vdbFrameShadowTotal(skyDown, vdbFrame.octaves) / vdbFrameShadowTotal(1.0, vdbFrame.octaves), skyDown, ms > 0.0);
      for (var k = 1u; k < ambientPairs; k++) {
        let d = vdbFrameLights.ambientDirections[k].xyz;
        let front = vdbFrameGridMarch(world, d, 1e20, jitter).t;
        let back = vdbFrameGridMarch(world, -d, 1e20, jitter).t;
        if (ms > 0.0) {
          ambientUp += vdbFrameIsotropicDiffusion(front.z, back.z, front.w);
          ambientDown += vdbFrameIsotropicDiffusion(back.z, front.z, back.w);
        } else {
          ambientUp += vdbFrameShadowTotal(front.y, vdbFrame.octaves) / vdbFrameShadowTotal(1.0, vdbFrame.octaves);
          ambientDown += vdbFrameShadowTotal(back.y, vdbFrame.octaves) / vdbFrameShadowTotal(1.0, vdbFrame.octaves);
        }
      }
      ambientUp /= f32(ambientPairs);
      ambientDown /= f32(ambientPairs);
    }
    t0 = vec4f(skyUp, skyDown, results[0].y, results[1].y);
    t1 = vec4f(results[2].y, results[3].y, results[4].y, results[5].y);
    t2 = vec4f(results[0].x, results[1].x, results[2].x, results[3].x);
    t3 = vec4f(results[4].x, results[5].x, ambientUp, ambientDown);
  }
  let base = cascade * 4u;
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base].xyz) + cell, t0);
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base + 1u].xyz) + cell, t1);
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base + 2u].xyz) + cell, t2);
  textureStore(vdbFrameTransmittanceOut, vec3u(vdbFrame.transmittanceOrigin[base + 3u].xyz) + cell, t3);
}
`;
}

/**
 * K6.4: the spectral 'frame' gather (his FrameDensityMaterial / FrameLightingMaterial spectral branches): per volume the
 * spectral density sums (albedo × extinction rgb, weight = channel mean; emission and visible t × weight; extinction
 * rgb), then the lighting at the summed visible position samples the slot groups (3 targets each) there. Writes the
 * scatter (radiance × α of each channel, green α) and scatterAlpha (red α, blue α, 0, green α). Multi-group partials:
 * the 12 sums as 6 words.
 */
function vdbFrameSpectralLightingGatherWgsl(o: { multi: boolean; channels: VDBVolumeChannels; hook?: string | null }): string {
  const b = FRAME_GATHER_BINDINGS;
  const write = `  textureStore(vdbFrameScatter, cell, out.scatter);
  textureStore(vdbFrameScatterAlpha, cell, out.alpha);`;
  const store = o.multi
    ? `  let role = vdbFrameGroup.range.z;
  let isFirst = (role & 1u) != 0u;
  let isLast = (role & 2u) != 0u;
  if (!inside) {
    if (isLast) {
      textureStore(vdbFrameScatter, cell, vec4f(0.0));
      textureStore(vdbFrameScatterAlpha, cell, vec4f(0.0));
    }
    return;
  }
  let depth = u32(vdbFrame.grid.z);
  var words: array<u32, 6>;
  if (!isFirst) {
    for (var k = 0u; k < 6u; k++) {
      words[k] = textureLoad(vdbFramePartials, vec3u(cell.xy, cell.z + k * depth)).x;
    }
    sums.a += vec4f(unpack2x16float(words[0]), unpack2x16float(words[1]));
    sums.e += vec4f(unpack2x16float(words[2]), unpack2x16float(words[3]));
    sums.x += vec4f(unpack2x16float(words[4]), unpack2x16float(words[5]));
  }
  if (isLast) {
    let out = vdbFrameSpectralLightFroxel(sums, f);
${write}
  } else {
    words = array<u32, 6>(pack2x16float(sums.a.xy), pack2x16float(sums.a.zw), pack2x16float(sums.e.xy), pack2x16float(sums.e.zw), pack2x16float(sums.x.xy), pack2x16float(sums.x.zw));
    for (var k = 0u; k < 6u; k++) {
      textureStore(vdbFramePartials, vec3u(cell.xy, cell.z + k * depth), vec4u(words[k], 0u, 0u, 0u));
    }
  }`
    : `  var out = VdbFrameSpectralOut(vec4f(0.0), vec4f(0.0));
  if (inside) {
    out = vdbFrameSpectralLightFroxel(sums, f);
  }
${write}`;

  return `${frameChunksWgsl(["constants", "mat3", "froxel", "noise", "fade", "box", "lod", "phase", "shadow", "octaves", "spectral", "sh", "envSh", "grid"])}
@group(1) @binding(${b.volumes}) var<uniform> vdbFrameVolumes: VdbFrameVolumes;
@group(1) @binding(${b.group}) var<uniform> vdbFrameGroup: VdbFrameGroup;
@group(1) @binding(${b.scatter}) var vdbFrameScatter: texture_storage_3d<rgba16float, write>;
@group(1) @binding(${b.scatterAlpha}) var vdbFrameScatterAlpha: texture_storage_3d<rgba16float, write>;
${o.multi ? `@group(1) @binding(${b.partials}) var vdbFramePartials: texture_storage_3d<r32uint, read_write>;\n` : ""}@group(1) @binding(${b.transmittance}) var vdbFrameTransmittanceStack: texture_3d<f32>;
@group(1) @binding(${b.environmentGrid}) var vdbFrameEnvGrid: texture_3d<f32>;
${vdbFrameOccluderWgsl()}
${vdbFrameSourceFunctionsWgsl(o.channels, { lutBinding: b.emissionLut, hook: o.hook ?? null })}
${vdbFrameSliceWgsl(true)}
// His spectral density targets as sums: (albedo·extinction rgb, weight), (emission·weight, visible t·weight), (extinction rgb, 0)
struct VdbFrameSums {
  a: vec4f,
  e: vec4f,
  x: vec4f,
};
struct VdbFrameSpectralOut {
  scatter: vec4f,
  alpha: vec4f,
};

// sampleShadows of one slot group's 3 targets at a world position, 1 outside the grid (or without shadows), blended
// toward the next cascade within 10% of a cascade's sides
fn vdbFrameSpectralGroupAt(world: vec3f, group: u32) -> array<vec4f, 3> {
  var out = array<vec4f, 3>(vec4f(1.0), vec4f(1.0), vec4f(1.0));
  if (vdbFrame.lighting.z < 0.5) {
    return out;
  }
  let found = vdbFrameFindCascade(world, 0u);
  if (found < 0) {
    return out;
  }
  let cascades = u32(vdbFrame.gridLayout.x);
  let cascade = u32(found);
  let uvw = vdbFrameGridUvw(cascade, world);
  for (var k = 0u; k < 3u; k++) {
    out[k] = vdbFrameStackSample(vdbFrameTransmittanceStack, vdbFrame.transmittanceOrigin[3u * (group * cascades + cascade) + k].xyz, uvw);
  }
  let side = min(uvw, vec3f(1.0) - uvw);
  let blend = 1.0 - smoothstep(0.0, 0.1, min(side.x, min(side.y, side.z)));
  if (blend > 0.0) {
    let next = vdbFrameFindCascade(world, cascade + 1u);
    if (next >= 0) {
      let n = u32(next);
      let nextUvw = vdbFrameGridUvw(n, world);
      for (var k = 0u; k < 3u; k++) {
        out[k] = mix(out[k], vdbFrameStackSample(vdbFrameTransmittanceStack, vdbFrame.transmittanceOrigin[3u * (group * cascades + n) + k].xyz, nextUvw), vec4f(blend));
      }
    }
  }
  return out;
}

// frameUnpackSpectralGroup: entry a (sky: x up, y down; a slot: x total, y direct) or b of a group's 3 targets
fn vdbFrameSpectralEntry(m: array<vec4f, 3>, second: bool) -> VdbFrameSpectralShadow {
  if (second) {
    return VdbFrameSpectralShadow(vec3f(m[1].z, m[1].w, m[2].x), m[2].yzw);
  }
  return VdbFrameSpectralShadow(m[0].xyz, vec3f(m[0].w, m[1].x, m[1].y));
}

// K6.2b: the environment of his density pass (0 none, 1 'outward', 2 the environment grid)
fn vdbFrameEnvDensityMode() -> u32 {
  if (vdbFrameEnv.info.z == 0u) {
    return 0u;
  }
  let useShadows = vdbFrame.lighting.z > 0.5;
  let count = vdbFrameEnv.info.x;
  if (useShadows && vdbFrame.environment.z > 0.5 && count > u32(vdbFrame.environment.y)) {
    return 2u;
  }
  if (useShadows && vdbFrame.environment.x > 0.5 && count > 0u) {
    return 0u;
  }
  return 1u;
}

// His spectral density pass for one volume in one froxel; zero when the weight is not > 0
fn vdbFrameDensitySums(vi: u32, cell: vec3u, f: VdbFrameFroxel) -> VdbFrameSums {
  var out: VdbFrameSums;
  let s = vdbFrameSliceOf(vi, cell, f);
  let extinction = vdbFrameVolumes.v[vi].extinctionSpectrum.xyz * s.extinction;
  let weight = vdbFrameMeanComponent(extinction);
  if (!(weight > 0.0)) {
    return out;
  }
  let visibleT = select(0.5, s.visibleT, s.visibleT >= 0.0);
  let local = mix(s.localNear, s.localFar, vec3f(visibleT));
  let albedo = vdbFrameAlbedo(vi, local, s.readLod);
  var emission = vdbFrameEmission(vi, local, s.readLod) + vdbFrameVolumes.v[vi].scatterOpacity.xyz;
  let world = (vdbFrame.cameraWorld * vec4f(mix(f.viewNear, f.viewFar, vec3f(visibleT)), 1.0)).xyz;
  let octaves = vdbFrame.octaves;
  let envMode = vdbFrameEnvDensityMode();
  if (envMode == 1u) {
    // along the outward direction from the volume centre; its shadow between the sky down and up totals of each channel
    let sky = vdbFrameSpectralEntry(vdbFrameSpectralGroupAt(world, 0u), false);
    let skyDirection = normalize(vdbFrameLights.sky.xyz);
    var outward = skyDirection;
    if (length(local) > 1e-4) {
      outward = normalize(vdbFrameMat3(vdbFrame.cameraWorld) * (vdbFrameMat3(vdbFrameVolumes.v[vi].localToView) * local));
    }
    let skyBlend = dot(outward, skyDirection) * 0.5 + 0.5;
    // K6.Q1a: with the shadowed ambient the entry holds the hemispherical visibilities (totals already)
    let ambientOcclusion = vdbFrameLights.counts.z > 0u;
    let upTotal = select(vdbFrameSpectralShadowTotal(sky.x, octaves, albedo), sky.x, ambientOcclusion);
    let downTotal = select(vdbFrameSpectralShadowTotal(sky.y, octaves, albedo), sky.y, ambientOcclusion);
    let envShadow = mix(downTotal, upTotal, vec3f(skyBlend));
    if (vdbFrame.lighting.w > 0.5) {
      let inward = mix(upTotal, downTotal, vec3f(skyBlend));
      emission += albedo * ((vdbFrameEnvIrradiance(outward) * envShadow + vdbFrameEnvIrradiance(-outward) * inward) * 0.5) * (1.0 / VDB_FRAME_PI);
    } else {
      emission += albedo * vdbFrameEnvIrradiance(outward) * envShadow * (1.0 / VDB_FRAME_PI);
    }
  } else if (envMode == 2u) {
    let cascade = u32(vdbFrame.gridLayout.x) - 1u;
    let uvw = vdbFrameGridUvw(cascade, world);
    var environment = vec3f(0.0);
    if (vdbFrameInsideGrid(uvw)) {
      environment = vdbFrameStackSample(vdbFrameEnvGrid, vec3f(0.0), uvw).xyz;
    } else {
      for (var i = u32(vdbFrame.environment.y); i < vdbFrameEnv.info.x; i++) {
        environment += vdbFrameEnv.color[i].xyz;
      }
      environment = max(environment, vec3f(0.0)) * vdbFrameSpectralShadowTotal(vec3f(1.0), octaves, albedo);
    }
    emission += albedo * environment * (1.0 / VDB_FRAME_PI);
  }
  out.a = vec4f(albedo * extinction, weight);
  out.e = vec4f(emission * weight, visibleT * weight);
  out.x = vec4f(extinction, 0.0);
  return out;
}

// three's getDistanceAttenuation
fn vdbFrameDistanceAttenuation(distance: f32, cutoff: f32, decay: f32) -> f32 {
  let falloff = 1.0 / max(pow(distance, decay), 0.01);
  if (cutoff > 0.0) {
    let x = clamp(1.0 - pow(distance / cutoff, 4.0), 0.0, 1.0);
    return falloff * x * x;
  }
  return falloff;
}

// His spectral lighting pass on the summed density: the slot groups sampled at the summed visible position
fn vdbFrameSpectralLightFroxel(sums: VdbFrameSums, f: VdbFrameFroxel) -> VdbFrameSpectralOut {
  let weight = sums.a.w;
  if (!(weight > 0.0)) {
    return VdbFrameSpectralOut(vec4f(0.0), vec4f(0.0));
  }
  let extinction = sums.x.xyz;
  let visibleT = clamp(sums.e.w / weight, 0.0, 1.0);
  let world = (vdbFrame.cameraWorld * vec4f(mix(f.viewNear, f.viewFar, vec3f(visibleT)), 1.0)).xyz;
  let ray = normalize(vdbFrameMat3(vdbFrame.cameraWorld) * (f.viewFar - f.viewNear));
  let albedo = sums.a.xyz / max(extinction, vec3f(1e-8));
  let emission = sums.e.xyz / weight;
  let octaves = vdbFrame.octaves;
  let groups = u32(vdbFrame.spectral.y);
  // the slot entries (total, direct) of slots 0..5 and the sky (up, down) totals; 1 for groups not in use
  var slots: array<VdbFrameSpectralShadow, 6>;
  var skyMaps = array<vec4f, 3>(vec4f(1.0), vec4f(1.0), vec4f(1.0));
  for (var g = 0u; g < 4u; g++) {
    var maps = array<vec4f, 3>(vec4f(1.0), vec4f(1.0), vec4f(1.0));
    if (g < groups) {
      maps = vdbFrameSpectralGroupAt(world, g);
    }
    if (g == 0u) {
      skyMaps = maps;
      slots[0] = vdbFrameSpectralEntry(maps, true);
    } else {
      slots[2u * g - 1u] = vdbFrameSpectralEntry(maps, false);
      if (2u * g < 6u) {
        slots[2u * g] = vdbFrameSpectralEntry(maps, true);
      }
    }
  }
  let sky = vdbFrameSpectralEntry(skyMaps, false);
  let skyDirection = normalize(vdbFrameLights.sky.xyz);
  // K6.Q1a shadowed ambient: group 0's sky entry holds the hemispherical visibilities of each channel (the sky totals
  // replaced); ambient x their mean (off: unshadowed, his)
  let ambientOcclusion = vdbFrameLights.counts.z > 0u;
  let skyUp = select(vdbFrameSpectralShadowTotal(sky.x, octaves, albedo), sky.x, ambientOcclusion) * 0.5;
  let skyDown = select(vdbFrameSpectralShadowTotal(sky.y, octaves, albedo), sky.y, ambientOcclusion) * 0.5;
  var irradiance = vdbFrameLights.ambient.xyz * select(vec3f(1.0), skyUp + skyDown, ambientOcclusion);
  irradiance += vdbFrameShIrradiance(skyDirection) * skyUp;
  irradiance += vdbFrameShIrradiance(-skyDirection) * skyDown;
  irradiance += vdbFrameLights.hemiSkySum.xyz * skyUp;
  irradiance += vdbFrameLights.hemiGroundSum.xyz * skyDown;
  let shadowLights = u32(vdbFrame.gridLayout.w);
  for (var i = 0u; i < vdbFrameLights.counts.x; i++) {
    let position = vdbFrameLights.position[i];
    let color = vdbFrameLights.color[i];
    let spot = vdbFrameLights.spot[i];
    let falloff = vdbFrameLights.falloff[i];
    let lightVector = position.xyz - world;
    let spotAttenuation = select(1.0, smoothstep(spot.w, color.w, dot(normalize(lightVector), spot.xyz)), position.w > 2.5);
    let distanceAttenuation = select(1.0, vdbFrameDistanceAttenuation(length(lightVector), falloff.x, falloff.y), position.w > 1.5);
    let direction = select(normalize(lightVector), position.xyz, position.w < 1.5);
    let cosTheta = dot(ray, direction);
    let hasSlot = i < shadowLights && i < 6u;
    // K6.5b: the host occluder visibility at the lit position: a slot's direct part (its scattered part was occluded in
    // the transmittance; octaves: the whole light), a light without a slot whole
    let visibility = vdbFrameOccluderVisibility(world, i);
    var light: vec3f;
    if (hasSlot && octaves.x <= 0.0) {
      let entry = slots[min(i, 5u)];
      light = vdbFrameSpectralShadowedLight(vdbFrame.framePhase, vdbFrame.framePhaseSpectrum.xyz, octaves, cosTheta, VdbFrameSpectralShadow(entry.y * visibility, entry.x - entry.y * (1.0 - visibility)), albedo);
    } else {
      var shadow = VdbFrameSpectralShadow(vec3f(1.0), vec3f(1.0));
      if (hasSlot) {
        let entry = slots[min(i, 5u)];
        shadow = VdbFrameSpectralShadow(entry.y, entry.x);
      }
      light = vdbFrameSpectralShadowedLight(vdbFrame.framePhase, vdbFrame.framePhaseSpectrum.xyz, octaves, cosTheta, shadow, albedo) * visibility;
    }
    irradiance += color.rgb * light * spotAttenuation * distanceAttenuation;
  }
  // K6.2b: the dominant 'directional' environment light in its slot (isotropic, no phase)
  if (vdbFrame.lighting.z > 0.5 && vdbFrame.environment.x > 0.5 && vdbFrameEnv.info.z != 0u && vdbFrameEnv.info.x > 0u) {
    irradiance += vdbFrameEnv.color[0].xyz * vdbFrameSpectralShadowTotal(slots[min(shadowLights, 5u)].x, octaves, albedo);
  }
  let radiance = albedo * irradiance * (1.0 / VDB_FRAME_PI) + emission;
  let alpha = vec3f(1.0) - exp(-extinction * length(f.viewFar - f.viewNear));
  return VdbFrameSpectralOut(vec4f(radiance * alpha, alpha.y), vec4f(alpha.x, alpha.z, 0.0, alpha.y));
}

@compute @workgroup_size(8, 8, 1)
fn vdbFrameGatherCs(@builtin(global_invocation_id) gid: vec3u) {
  let box = vdbFrame.dispatchBox;
  if (gid.x >= box.z || gid.y >= box.w || f32(gid.z) >= vdbFrame.grid.z) {
    return;
  }
  let cell = vec3u(box.x + gid.x, box.y + gid.y, gid.z);
  let area = vdbFrame.areaRect;
  let areaSlices = vdbFrame.areaSlices;
  let p = vec2f(cell.xy);
  let z = f32(cell.z);
  let inside = p.x >= area.x && p.y >= area.y && p.x < area.z && p.y < area.w && z >= areaSlices.x && z < areaSlices.y;
  var sums: VdbFrameSums;
  var f: VdbFrameFroxel;
  if (inside) {
    f = vdbFrameFroxelAt(cell);
    let group = vdbFrameGroup.range;
    for (var i = 0u; i < group.y; i++) {
      let r = vdbFrameDensitySums(group.x + i, cell, f);
      sums.a += r.a;
      sums.e += r.e;
      sums.x += r.x;
    }
  }
${store}
}
`;
}

/** The 'frame' gather kernel (after the source module, the source WGSL and `vdbFrameUniformsWgsl()`). */
export function vdbFrameLightingGatherWgsl(o: { multi: boolean; channels: VDBVolumeChannels; hook?: string | null; spectral?: boolean }): string {
  if (o.spectral) return vdbFrameSpectralLightingGatherWgsl(o);
  const b = FRAME_GATHER_BINDINGS;
  const store = o.multi
    ? `  let role = vdbFrameGroup.range.z;
  let isFirst = (role & 1u) != 0u;
  let isLast = (role & 2u) != 0u;
  if (!inside) {
    if (isLast) {
      textureStore(vdbFrameScatter, cell, vec4f(0.0));
    }
    return;
  }
  let depth = u32(vdbFrame.grid.z);
  var words: array<u32, ${VDB_FRAME_SUM_WORDS}>;
  if (!isFirst) {
    for (var k = 0u; k < ${VDB_FRAME_SUM_WORDS}u; k++) {
      words[k] = textureLoad(vdbFramePartials, vec3u(cell.xy, cell.z + k * depth)).x;
    }
    sums.a += vec4f(unpack2x16float(words[0]), unpack2x16float(words[1]));
    sums.e += vec4f(unpack2x16float(words[2]), unpack2x16float(words[3]));
    sums.s0 += vec4f(unpack2x16float(words[4]), unpack2x16float(words[5]));
    sums.s1 += vec4f(unpack2x16float(words[6]), unpack2x16float(words[7]));
  }
  if (isLast) {
    textureStore(vdbFrameScatter, cell, vdbFrameLightFroxel(sums, f));
  } else {
    words = array<u32, ${VDB_FRAME_SUM_WORDS}>(
      pack2x16float(sums.a.xy), pack2x16float(sums.a.zw), pack2x16float(sums.e.xy), pack2x16float(sums.e.zw),
      pack2x16float(sums.s0.xy), pack2x16float(sums.s0.zw), pack2x16float(sums.s1.xy), pack2x16float(sums.s1.zw));
    for (var k = 0u; k < ${VDB_FRAME_SUM_WORDS}u; k++) {
      textureStore(vdbFramePartials, vec3u(cell.xy, cell.z + k * depth), vec4u(words[k], 0u, 0u, 0u));
    }
  }`
    : `  var out = vec4f(0.0);
  if (inside) {
    out = vdbFrameLightFroxel(sums, f);
  }
  textureStore(vdbFrameScatter, cell, out);`;

  return `${frameChunksWgsl(["constants", "mat3", "froxel", "noise", "fade", "box", "lod", "phase", "shadow", "octaves", "sh", "envSh", "grid"])}
@group(1) @binding(${b.volumes}) var<uniform> vdbFrameVolumes: VdbFrameVolumes;
@group(1) @binding(${b.group}) var<uniform> vdbFrameGroup: VdbFrameGroup;
@group(1) @binding(${b.scatter}) var vdbFrameScatter: texture_storage_3d<rgba16float, write>;
${o.multi ? `@group(1) @binding(${b.partials}) var vdbFramePartials: texture_storage_3d<r32uint, read_write>;\n` : ""}@group(1) @binding(${b.transmittance}) var vdbFrameTransmittanceStack: texture_3d<f32>;
@group(1) @binding(${b.environmentGrid}) var vdbFrameEnvGrid: texture_3d<f32>;
${vdbFrameOccluderWgsl()}
${vdbFrameSourceFunctionsWgsl(o.channels, { lutBinding: b.emissionLut, hook: o.hook ?? null })}
${VDB_FRAME_SLICE_WGSL}
// His density targets as sums: (albedo·σ, σ), (emission·σ, visible t·σ), (sky up, sky down, slot 0, 1)·σ, (slots 2–5)·σ
struct VdbFrameSums {
  a: vec4f,
  e: vec4f,
  s0: vec4f,
  s1: vec4f,
};

// sampleShadows: the 4 transmittance targets at a world position, 1 outside the grid (or without shadows), blended
// toward the next cascade within 10% of a cascade's sides
fn vdbFrameShadowsAt(world: vec3f) -> array<vec4f, 4> {
  var out = array<vec4f, 4>(vec4f(1.0), vec4f(1.0), vec4f(1.0), vec4f(1.0));
  if (vdbFrame.lighting.z < 0.5) {
    return out;
  }
  let found = vdbFrameFindCascade(world, 0u);
  if (found < 0) {
    return out;
  }
  let cascade = u32(found);
  let uvw = vdbFrameGridUvw(cascade, world);
  for (var k = 0u; k < 4u; k++) {
    out[k] = vdbFrameStackSample(vdbFrameTransmittanceStack, vdbFrame.transmittanceOrigin[cascade * 4u + k].xyz, uvw);
  }
  let side = min(uvw, vec3f(1.0) - uvw);
  let blend = 1.0 - smoothstep(0.0, 0.1, min(side.x, min(side.y, side.z)));
  if (blend > 0.0) {
    let next = vdbFrameFindCascade(world, cascade + 1u);
    if (next >= 0) {
      let n = u32(next);
      let nextUvw = vdbFrameGridUvw(n, world);
      for (var k = 0u; k < 4u; k++) {
        out[k] = mix(out[k], vdbFrameStackSample(vdbFrameTransmittanceStack, vdbFrame.transmittanceOrigin[n * 4u + k].xyz, nextUvw), vec4f(blend));
      }
    }
  }
  return out;
}

// K6.2b: the environment of his density pass. 0: none (no environment, or only the slot light, which the lighting
// adds), 1: 'outward' (SH irradiance along the outward direction), 2: the environment grid.
fn vdbFrameEnvDensityMode() -> u32 {
  if (vdbFrameEnv.info.z == 0u) {
    return 0u;
  }
  let useShadows = vdbFrame.lighting.z > 0.5;
  let count = vdbFrameEnv.info.x;
  if (useShadows && vdbFrame.environment.z > 0.5 && count > u32(vdbFrame.environment.y)) {
    return 2u;
  }
  if (useShadows && vdbFrame.environment.x > 0.5 && count > 0u) {
    return 0u;
  }
  return 1u;
}

// His density pass for one volume in one froxel: the slice's mean extinction σ (over the whole slice), the
// visibility-weighted position, albedo and emission there, and the shadow slots at that world position (scene
// light slots: green phase × direct + scattered); σ-weighted. Zero when σ ≤ 0.
fn vdbFrameDensitySums(vi: u32, cell: vec3u, f: VdbFrameFroxel) -> VdbFrameSums {
  var out: VdbFrameSums;
  let s = vdbFrameSliceOf(vi, cell, f);
  let extinction = s.extinction;
  if (!(extinction > 0.0)) {
    return out;
  }
  let visibleT = select(0.5, s.visibleT, s.visibleT >= 0.0);
  let local = mix(s.localNear, s.localFar, vec3f(visibleT));
  let albedo = vdbFrameAlbedo(vi, local, s.readLod);
  var emission = vdbFrameEmission(vi, local, s.readLod) + vdbFrameVolumes.v[vi].scatterOpacity.xyz;
  let world = (vdbFrame.cameraWorld * vec4f(mix(f.viewNear, f.viewFar, vec3f(visibleT)), 1.0)).xyz;
  let shadows = vdbFrameShadowsAt(world);
  // K6.3: isotropic slots (sky up and down, the environment slot) hold total transmittance (octaves: their sum)
  let octaves = vdbFrame.octaves;
  var skyUp = vdbFrameShadowTotal(shadows[0].x, octaves);
  var skyDown = vdbFrameShadowTotal(shadows[0].y, octaves);
  // K6.Q1 shadowed ambient: the hemispherical visibilities replace the sky totals (hemisphere, probe, 'outward', ambient)
  if (vdbFrameLights.counts.z > 0u) {
    skyUp = shadows[3].z;
    skyDown = shadows[3].w;
  }
  let envMode = vdbFrameEnvDensityMode();
  if (envMode == 1u) {
    // along the outward direction from the volume centre; its shadow between the sky down and up totals
    let skyDirection = normalize(vdbFrameLights.sky.xyz);
    var outward = skyDirection;
    if (length(local) > 1e-4) {
      outward = normalize(vdbFrameMat3(vdbFrame.cameraWorld) * (vdbFrameMat3(vdbFrameVolumes.v[vi].localToView) * local));
    }
    let skyBlend = dot(outward, skyDirection) * 0.5 + 0.5;
    let envShadow = mix(skyDown, skyUp, skyBlend);
    if (vdbFrame.lighting.w > 0.5) {
      // diffusion: the outward and the inward irradiance, each with its two-sided total
      let inward = mix(skyUp, skyDown, skyBlend);
      emission += albedo * ((vdbFrameEnvIrradiance(outward) * envShadow + vdbFrameEnvIrradiance(-outward) * inward) * 0.5) * (1.0 / VDB_FRAME_PI);
    } else {
      emission += albedo * vdbFrameEnvIrradiance(outward) * envShadow * (1.0 / VDB_FRAME_PI);
    }
  } else if (envMode == 2u) {
    // the environment grid at the visible position; outside it the unshadowed light of its lights
    let cascade = u32(vdbFrame.gridLayout.x) - 1u;
    let uvw = vdbFrameGridUvw(cascade, world);
    var environment = vec3f(0.0);
    if (vdbFrameInsideGrid(uvw)) {
      environment = vdbFrameStackSample(vdbFrameEnvGrid, vec3f(0.0), uvw).xyz;
    } else {
      for (var i = u32(vdbFrame.environment.y); i < vdbFrameEnv.info.x; i++) {
        environment += vdbFrameEnv.color[i].xyz;
      }
      environment = max(environment, vec3f(0.0)) * vdbFrameShadowTotal(1.0, octaves);
    }
    emission += albedo * environment * (1.0 / VDB_FRAME_PI);
  }
  let ray = normalize(vdbFrameMat3(vdbFrame.cameraWorld) * (f.viewFar - f.viewNear));
  var slot = array<f32, 6>(shadows[0].z, shadows[0].w, shadows[1].x, shadows[1].y, shadows[1].z, shadows[1].w);
  var direct = array<f32, 6>(shadows[2].x, shadows[2].y, shadows[2].z, shadows[2].w, shadows[3].x, shadows[3].y);
  let sceneSlots = min(u32(vdbFrame.gridLayout.w), 6u);
  for (var i = 0u; i < 6u; i++) {
    if (i < sceneSlots) {
      // frameShadowedLight(...).g: the green phase x direct + the scattered light (octaves: their phased sum). K6.5b: the
      // direct part x the host occluder visibility at the visible position (sharp at froxel resolution); the scattered
      // part was occluded in the transmittance, except the octaves' (1 / (1 + depth) there): occluded here
      let light = vdbFrameLights.position[i];
      let direction = select(normalize(light.xyz - world), light.xyz, light.w < 1.5);
      let visibility = vdbFrameOccluderVisibility(world, i);
      let cosTheta = dot(ray, direction);
      if (octaves.x > 0.0) {
        slot[i] = vdbFrameShadowedLight(vdbFrame.framePhase, vdbFrame.framePhaseSpectrum.xyz, octaves, cosTheta, vec2f(direct[i], slot[i])).y * visibility;
      } else {
        slot[i] = vdbFrameShadowedLight(vdbFrame.framePhase, vdbFrame.framePhaseSpectrum.xyz, octaves, cosTheta, vec2f(direct[i] * visibility, slot[i] - direct[i] * (1.0 - visibility))).y;
      }
    } else {
      slot[i] = vdbFrameShadowTotal(slot[i], octaves);
    }
  }
  out.a = vec4f(albedo * extinction, extinction);
  out.e = vec4f(emission * extinction, visibleT * extinction);
  out.s0 = min(vec4f(skyUp, skyDown, slot[0], slot[1]) * extinction, vec4f(6.0e4));
  out.s1 = min(vec4f(slot[2], slot[3], slot[4], slot[5]) * extinction, vec4f(6.0e4));
  return out;
}

// three's getDistanceAttenuation
fn vdbFrameDistanceAttenuation(distance: f32, cutoff: f32, decay: f32) -> f32 {
  let falloff = 1.0 / max(pow(distance, decay), 0.01);
  if (cutoff > 0.0) {
    let x = clamp(1.0 - pow(distance / cutoff, 4.0), 0.0, 1.0);
    return falloff * x * x;
  }
  return falloff;
}

// His lighting pass on the summed density of all volumes in the froxel: premultiplied (radiance·α, α)
fn vdbFrameLightFroxel(sums: VdbFrameSums, f: VdbFrameFroxel) -> vec4f {
  let extinction = sums.a.w;
  if (!(extinction > 0.0)) {
    return vec4f(0.0);
  }
  let visibleT = clamp(sums.e.w / extinction, 0.0, 1.0);
  let world = (vdbFrame.cameraWorld * vec4f(mix(f.viewNear, f.viewFar, vec3f(visibleT)), 1.0)).xyz;
  let ray = normalize(vdbFrameMat3(vdbFrame.cameraWorld) * (f.viewFar - f.viewNear));
  let s0 = sums.s0 / extinction;
  let s1 = sums.s1 / extinction;
  var slot = array<f32, 6>(s0.z, s0.w, s1.x, s1.y, s1.z, s1.w);
  // per-channel phases: the slots hold the green phase; the direct transmittance adds the red and blue difference
  let colorPhase = vdbFrame.framePhaseSpectrum.z > 0.5;
  var direct = array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  if (colorPhase) {
    let d = vdbFrameShadowsAt(world);
    direct = array<f32, 6>(d[2].x, d[2].y, d[2].z, d[2].w, d[3].x, d[3].y);
  }
  let sky = normalize(vdbFrameLights.sky.xyz);
  let skyUp = s0.x * 0.5;
  let skyDown = s0.y * 0.5;
  // K6.Q1: ambient x the mean of the hemispherical visibilities (off: unshadowed, his)
  var irradiance = vdbFrameLights.ambient.xyz * select(1.0, skyUp + skyDown, vdbFrameLights.counts.z > 0u);
  irradiance += vdbFrameShIrradiance(sky) * skyUp;
  irradiance += vdbFrameShIrradiance(-sky) * skyDown;
  irradiance += vdbFrameLights.hemiSkySum.xyz * skyUp;
  irradiance += vdbFrameLights.hemiGroundSum.xyz * skyDown;
  let shadowLights = u32(vdbFrame.gridLayout.w);
  for (var i = 0u; i < vdbFrameLights.counts.x; i++) {
    let position = vdbFrameLights.position[i];
    let color = vdbFrameLights.color[i];
    let spot = vdbFrameLights.spot[i];
    let falloff = vdbFrameLights.falloff[i];
    let lightVector = position.xyz - world;
    let spotAttenuation = select(1.0, smoothstep(spot.w, color.w, dot(normalize(lightVector), spot.xyz)), position.w > 2.5);
    let distanceAttenuation = select(1.0, vdbFrameDistanceAttenuation(length(lightVector), falloff.x, falloff.y), position.w > 1.5);
    let direction = select(normalize(lightVector), position.xyz, position.w < 1.5);
    let phase = vdbFramePhaseColor(vdbFrame.framePhase, vdbFrame.framePhaseSpectrum.xyz, dot(ray, direction));
    let hasSlot = i < shadowLights && i < 6u;
    // K6.5b: the host occluder visibility at the lit position (the slots hold it already; their per-channel phase
    // difference and the lights without a slot take it here). No slot: no shadow (K6.3: octaves still add their light)
    var factor = vdbFrameShadowedLight(vdbFrame.framePhase, vdbFrame.framePhaseSpectrum.xyz, vdbFrame.octaves, dot(ray, direction), vec2f(1.0));
    if (hasSlot) {
      factor = vec3f(slot[i]);
      if (colorPhase) {
        factor += (phase - phase.y) * (direct[i] * vdbFrameOccluderVisibility(world, i));
      }
    } else {
      factor *= vdbFrameOccluderVisibility(world, i);
    }
    irradiance += color.rgb * factor * spotAttenuation * distanceAttenuation;
  }
  // K6.2b: the dominant 'directional' environment light in its slot (isotropic, no phase)
  if (vdbFrame.lighting.z > 0.5 && vdbFrame.environment.x > 0.5 && vdbFrameEnv.info.z != 0u && vdbFrameEnv.info.x > 0u) {
    irradiance += vdbFrameEnv.color[0].xyz * slot[min(shadowLights, 5u)];
  }
  let albedo = sums.a.rgb / extinction;
  let emission = sums.e.rgb / extinction;
  let radiance = albedo * irradiance * (1.0 / VDB_FRAME_PI) + emission;
  let alpha = 1.0 - exp(-extinction * length(f.viewFar - f.viewNear));
  return vec4f(radiance * alpha, alpha);
}

@compute @workgroup_size(8, 8, 1)
fn vdbFrameGatherCs(@builtin(global_invocation_id) gid: vec3u) {
  let box = vdbFrame.dispatchBox;
  if (gid.x >= box.z || gid.y >= box.w || f32(gid.z) >= vdbFrame.grid.z) {
    return;
  }
  let cell = vec3u(box.x + gid.x, box.y + gid.y, gid.z);
  let area = vdbFrame.areaRect;
  let areaSlices = vdbFrame.areaSlices;
  let p = vec2f(cell.xy);
  let z = f32(cell.z);
  let inside = p.x >= area.x && p.y >= area.y && p.x < area.z && p.y < area.w && z >= areaSlices.x && z < areaSlices.y;
  var sums: VdbFrameSums;
  var f: VdbFrameFroxel;
  if (inside) {
    f = vdbFrameFroxelAt(cell);
    let group = vdbFrameGroup.range;
    for (var i = 0u; i < group.y; i++) {
      let r = vdbFrameDensitySums(group.x + i, cell, f);
      sums.a += r.a;
      sums.e += r.e;
      sums.s0 += r.s0;
      sums.s1 += r.s1;
    }
  }
${store}
}
`;
}
