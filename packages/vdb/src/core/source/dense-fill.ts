// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/wvdb/wvdbKernel.js upsample L122–188 (a multi-page LOD shows the previous LOD upsampled in the
//   bricks not decoded yet), src/three-openvdb/frame/webgpu/frameMipmaps.js downsample L20–62 and packAtlas L65–88.
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: WGSL compute instead of his worker / CPU loops; integer weights (8 fractional bits) so the CPU twin
// (`tools/wvdb/partial.ts` wvdbUpsampleTexels, `wvdbMipAtlas`) agrees byte for byte; only pending bricks upsample.
// Divergences: DIVERGES.md (K4.22 row).

/**
 * K4.22: the dense source's GPU fill of a partially streamed `.wvdb` LOD (`WvdbDenseTexture.fill`). One storage
 * buffer holds, at byte offsets: the pending-brick bits, the coarser LOD's texels, the partial LOD's texels (both
 * tight, as the host has them), then the texture's rows and the mip atlas's rows padded to 256 bytes (copy layout).
 * One compute pass:
 *
 * 1. `vdbDenseFillCs`, one thread per texel: a pending brick's texel is the coarse texels' trilinear upsample
 *    (his `axis`: centres aligned, edges clamped; weights in 1/256, the sum rounded half up once), any other texel
 *    is the host's; written into the cleared row region with `atomicOr` per byte.
 * 2. `vdbDenseMipCs`, one dispatch per mip level, one thread per texel: his box downsample (`(sum of 8 + 4) >> 3`,
 *    edge taps clamped) of the level below, into the atlas at his x offset (`atlasWidth − atlasWidth / 2^(l−1)`).
 *    With `mips: 'coarse'` (the default fill) level 1 of a pending brick is the coarser LOD's texel instead: the LOD
 *    the bake box-filtered from this one, so a far view of a pending brick reads it, not a blur of its upsample.
 *
 * Then `copyBufferToTexture` into the source's texture and atlas. The work is submitted at once (the source's
 * `replace` acts on the queue, as `writeTexture` does), so a later `replace` lands after it. r8unorm is not a
 * storage-texture format in core WebGPU, hence the buffer and the copies (Traps: the f16 multi-scale chain does
 * the same). A texture whose buffer would pass `maxStorageBufferBindingSize` or `maxBufferSize` is filled on the
 * CPU by the twin instead (`vdbDenseFillFits`).
 */

import { VDBError } from "../errors.js";
import { USAGE, WORKGROUP_AXIS } from "../compute/shared.js";
import type { WvdbDenseTexture } from "../../tools/wvdb/dense.js";
import { wvdbFillCoarseMips } from "../../tools/wvdb/partial.js";

const W = WORKGROUP_AXIS;
const STAGE_COMPUTE = 0x4;
/** Uniform slot stride (minUniformBufferOffsetAlignment's default). */
const SLOT = 256;
const PARAMS_BYTES = 96;
const BRICK = 8;

const FILL_WGSL = /* wgsl */ `
struct VdbDenseFillParams {
    dims: vec3u,
    texelBytes: u32,
    srcDims: vec3u,
    pendingWord: u32,
    bricks: vec3u,
    coarseByte: u32,
    fineByte: u32,
    dstByte: u32,
    dstRow: u32,
    dstRows: u32,
    dstX: u32,
    srcByte: u32,
    srcRow: u32,
    srcRows: u32,
    srcX: u32,
    coarseMip: u32,
    pad1: u32,
    pad2: u32,
};
@group(0) @binding(0) var<uniform> vdbFillP: VdbDenseFillParams;
@group(0) @binding(1) var<storage, read_write> vdbFillBuf: array<atomic<u32>>;

// Byte b of the buffer, and an OR of v into byte b (the row regions are cleared before the pass).
fn vdbFillLoad(b: u32) -> u32 {
    return (atomicLoad(&vdbFillBuf[b >> 2u]) >> ((b & 3u) * 8u)) & 0xffu;
}
fn vdbFillStore(b: u32, v: u32) {
    atomicOr(&vdbFillBuf[b >> 2u], (v & 0xffu) << ((b & 3u) * 8u));
}

// His upsample axis in integers: target texel i of size reads clamp((i + 1/2) src / size - 1/2, 0, src - 1);
// returns (lower tap, upper tap, upper weight in 1/256, the exact fraction rounded half up).
fn vdbFillAxis(i: u32, size: u32, src: u32) -> vec3u {
    let den = 2u * size;
    let num = min(u32(max(i32((2u * i + 1u) * src) - i32(size), 0)), (src - 1u) * den);
    let lo = num / den;
    let r = num - lo * den;
    return vec3u(lo, min(lo + 1u, src - 1u), (r * 256u + size) / den);
}

fn vdbFillTap(i: u32, c: u32) -> u32 {
    return vdbFillLoad(vdbFillP.coarseByte + i * vdbFillP.texelBytes + c);
}

@compute @workgroup_size(${W}, ${W}, ${W})
fn vdbDenseFillCs(@builtin(global_invocation_id) id: vec3u) {
    let p = vdbFillP;
    if any(id >= p.dims) {
        return;
    }
    let brick = (id.x >> 3u) + p.bricks.x * ((id.y >> 3u) + p.bricks.y * (id.z >> 3u));
    let pending = (atomicLoad(&vdbFillBuf[p.pendingWord + (brick >> 5u)]) >> (brick & 31u)) & 1u;
    let dst = p.dstByte + (id.z * p.dstRows + id.y) * p.dstRow + (p.dstX + id.x) * p.texelBytes;
    if pending == 0u {
        let src = p.fineByte + ((id.z * p.dims.y + id.y) * p.dims.x + id.x) * p.texelBytes;
        for (var c = 0u; c < p.texelBytes; c++) {
            vdbFillStore(dst + c, vdbFillLoad(src + c));
        }
        return;
    }
    let ax = vdbFillAxis(id.x, p.dims.x, p.srcDims.x);
    let ay = vdbFillAxis(id.y, p.dims.y, p.srcDims.y);
    let az = vdbFillAxis(id.z, p.dims.z, p.srcDims.z);
    let r00 = (ay.x + az.x * p.srcDims.y) * p.srcDims.x;
    let r10 = (ay.y + az.x * p.srcDims.y) * p.srcDims.x;
    let r01 = (ay.x + az.y * p.srcDims.y) * p.srcDims.x;
    let r11 = (ay.y + az.y * p.srcDims.y) * p.srcDims.x;
    for (var c = 0u; c < p.texelBytes; c++) {
        let c00 = vdbFillTap(ax.x + r00, c) * (256u - ax.z) + vdbFillTap(ax.y + r00, c) * ax.z;
        let c10 = vdbFillTap(ax.x + r10, c) * (256u - ax.z) + vdbFillTap(ax.y + r10, c) * ax.z;
        let c01 = vdbFillTap(ax.x + r01, c) * (256u - ax.z) + vdbFillTap(ax.y + r01, c) * ax.z;
        let c11 = vdbFillTap(ax.x + r11, c) * (256u - ax.z) + vdbFillTap(ax.y + r11, c) * ax.z;
        let c0 = c00 * (256u - ay.z) + c10 * ay.z;
        let c1 = c01 * (256u - ay.z) + c11 * ay.z;
        vdbFillStore(dst + c, (c0 * (256u - az.z) + c1 * az.z + 0x800000u) >> 24u);
    }
}

fn vdbMipTap(x: u32, y: u32, z: u32, c: u32) -> u32 {
    let p = vdbFillP;
    return vdbFillLoad(p.srcByte + (z * p.srcRows + y) * p.srcRow + (p.srcX + x) * p.texelBytes + c);
}

@compute @workgroup_size(${W}, ${W}, ${W})
fn vdbDenseMipCs(@builtin(global_invocation_id) id: vec3u) {
    let p = vdbFillP;
    if any(id >= p.dims) {
        return;
    }
    let s = p.srcDims - vec3u(1u);
    let x0 = min(id.x * 2u, s.x);
    let x1 = min(id.x * 2u + 1u, s.x);
    let y0 = min(id.y * 2u, s.y);
    let y1 = min(id.y * 2u + 1u, s.y);
    let z0 = min(id.z * 2u, s.z);
    let z1 = min(id.z * 2u + 1u, s.z);
    let dst = p.dstByte + (id.z * p.dstRows + id.y) * p.dstRow + (p.dstX + id.x) * p.texelBytes;
    if p.coarseMip != 0u {
        // Level 1 of a pending brick: the coarser LOD's texel (its size is level 1's). Children 2x, 2x + 1 share a brick.
        let brick = ((id.x * 2u) >> 3u) + p.bricks.x * (((id.y * 2u) >> 3u) + p.bricks.y * ((id.z * 2u) >> 3u));
        if ((atomicLoad(&vdbFillBuf[p.pendingWord + (brick >> 5u)]) >> (brick & 31u)) & 1u) != 0u {
            let src = p.coarseByte + ((id.z * p.dims.y + id.y) * p.dims.x + id.x) * p.texelBytes;
            for (var c = 0u; c < p.texelBytes; c++) {
                vdbFillStore(dst + c, vdbFillLoad(src + c));
            }
            return;
        }
    }
    for (var c = 0u; c < p.texelBytes; c++) {
        let sum = vdbMipTap(x0, y0, z0, c) + vdbMipTap(x1, y0, z0, c) + vdbMipTap(x0, y1, z0, c) + vdbMipTap(x1, y1, z0, c)
            + vdbMipTap(x0, y0, z1, c) + vdbMipTap(x1, y0, z1, c) + vdbMipTap(x0, y1, z1, c) + vdbMipTap(x1, y1, z1, c);
        vdbFillStore(dst + c, (sum + 4u) >> 3u);
    }
}
`;

/** The WGSL of the fill and mip kernels (exported for probes and tests). */
export function vdbDenseFillWgsl(): string {
  return FILL_WGSL;
}

const align = (n: number, a: number) => Math.ceil(n / a) * a;
const levelSize = (size: number, level: number) => Math.max(1, Math.ceil(size / 2 ** level));

/** One dispatch: its parameters (u32 words of `VdbDenseFillParams`) and size. */
interface Dispatch {
  entry: "fill" | "mip";
  dims: [number, number, number];
  params: Uint32Array;
}

/** Where everything sits in the fill buffer of one texture (bytes), and the dispatches. */
export interface VDBDenseFillLayout {
  readonly key: string;
  readonly texelBytes: number;
  readonly dims: readonly [number, number, number];
  readonly srcDims: readonly [number, number, number];
  readonly pendingByte: number;
  readonly pendingWords: number;
  readonly coarseByte: number;
  readonly fineByte: number;
  readonly mapByte: number;
  readonly mapRow: number;
  /** The atlas rows (null without a mip atlas). */
  readonly atlas: { readonly byte: number; readonly row: number; readonly dimensions: readonly [number, number, number]; readonly levels: number } | null;
  /** Level 1 of the pending bricks from the coarse texels (`fill.mips` 'coarse' and a halving size). */
  readonly coarseMips: boolean;
  /** Buffer bytes. */
  readonly size: number;
}

/** The fill buffer layout of `tex` (which must carry a `fill`). */
export function vdbDenseFillLayout(tex: WvdbDenseTexture): VDBDenseFillLayout {
  const fill = tex.fill;
  if (!fill) throw new VDBError("VDB_BAD_OPTION", "vdbDenseFillLayout: the texture has no fill.");
  const tb = tex.texelBytes;
  const dims = [...tex.dimensions] as [number, number, number];
  const srcDims = [...fill.from.dimensions] as [number, number, number];
  const bricks = dims.map((d) => Math.ceil(d / BRICK));
  const pendingWords = Math.ceil((bricks[0]! * bricks[1]! * bricks[2]!) / 32);
  const coarseByte = align(pendingWords * 4, 16);
  const fineByte = coarseByte + align(srcDims[0] * srcDims[1] * srcDims[2] * tb, 16);
  const mapByte = align(fineByte + dims[0] * dims[1] * dims[2] * tb, 256);
  const mapRow = align(dims[0] * tb, 256);
  let end = mapByte + mapRow * dims[1] * dims[2];
  let atlas: VDBDenseFillLayout["atlas"] = null;
  if (tex.mip) {
    const a = tex.mip.dimensions;
    atlas = { byte: align(end, 256), row: align(a[0] * tb, 256), dimensions: [a[0], a[1], a[2]], levels: tex.mip.levels };
    end = atlas.byte + atlas.row * a[1] * a[2];
  }
  const coarseMips = !!atlas && wvdbFillCoarseMips(tex);
  return {
    key: `${dims.join("x")}|${srcDims.join("x")}|${tb}|${atlas ? `${atlas.dimensions.join("x")}:${atlas.levels}` : "-"}|${coarseMips ? "coarse" : "box"}`,
    texelBytes: tb,
    dims,
    srcDims,
    pendingByte: 0,
    pendingWords,
    coarseByte,
    fineByte,
    mapByte,
    mapRow,
    atlas,
    coarseMips,
    size: align(end, 4),
  };
}

/** The dispatches of a layout: the fill, then one per mip level (level 1 reads the texture rows, then the atlas). */
function dispatchesOf(l: VDBDenseFillLayout): Dispatch[] {
  const [w, h, d] = l.dims;
  const bricks = l.dims.map((v) => Math.ceil(v / BRICK));
  const params = (o: Partial<Record<string, number | readonly number[]>>): Uint32Array => {
    const u = new Uint32Array(PARAMS_BYTES / 4);
    const vec = (at: number, v: readonly number[]) => u.set(v, at);
    vec(0, (o.dims as readonly number[]) ?? [0, 0, 0]);
    u[3] = l.texelBytes;
    vec(4, (o.srcDims as readonly number[]) ?? [0, 0, 0]);
    u[7] = l.pendingByte / 4;
    vec(8, bricks);
    u[11] = l.coarseByte;
    u[12] = l.fineByte;
    u[13] = o.dstByte as number;
    u[14] = o.dstRow as number;
    u[15] = o.dstRows as number;
    u[16] = (o.dstX as number) ?? 0;
    u[17] = (o.srcByte as number) ?? 0;
    u[18] = (o.srcRow as number) ?? 0;
    u[19] = (o.srcRows as number) ?? 0;
    u[20] = (o.srcX as number) ?? 0;
    u[21] = (o.coarseMip as number) ?? 0;
    return u;
  };
  const out: Dispatch[] = [{ entry: "fill", dims: [w, h, d], params: params({ dims: l.dims, srcDims: l.srcDims, dstByte: l.mapByte, dstRow: l.mapRow, dstRows: h, dstX: 0 }) }];
  const a = l.atlas;
  if (a) {
    const aw = a.dimensions[0];
    const xOf = (level: number) => aw - aw / 2 ** (level - 1);
    for (let level = 1; level <= a.levels; level++) {
      const dims = [levelSize(w, level), levelSize(h, level), levelSize(d, level)] as [number, number, number];
      const src = level === 1 ? { srcByte: l.mapByte, srcRow: l.mapRow, srcRows: h, srcX: 0 } : { srcByte: a.byte, srcRow: a.row, srcRows: a.dimensions[1], srcX: xOf(level - 1) };
      const srcDims = [levelSize(w, level - 1), levelSize(h, level - 1), levelSize(d, level - 1)];
      const coarseMip = level === 1 && l.coarseMips ? 1 : 0;
      out.push({ entry: "mip", dims, params: params({ dims, srcDims, dstByte: a.byte, dstRow: a.row, dstRows: a.dimensions[1], dstX: xOf(level), coarseMip, ...src }) });
    }
  }
  return out;
}

/** True when `layout`'s buffer fits the device's storage binding and buffer limits (else fill on the CPU). */
export function vdbDenseFillFits(device: GPUDevice, layout: VDBDenseFillLayout): boolean {
  const limits = (device as { limits?: Partial<GPUSupportedLimits> }).limits ?? {};
  const binding = limits.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const buffer = limits.maxBufferSize ?? 256 * 1024 * 1024;
  return layout.size <= binding && layout.size <= buffer;
}

interface Pipelines {
  fill: GPUComputePipeline;
  mip: GPUComputePipeline;
  group0: GPUBindGroupLayout;
  error: Promise<GPUError | null>;
}

const pipelines = new WeakMap<GPUDevice, Pipelines>();

function fillPipelines(device: GPUDevice): Pipelines {
  const cached = pipelines.get(device);
  if (cached) return cached;
  device.pushErrorScope("validation");
  const group0 = device.createBindGroupLayout({
    label: "vdb dense fill group 0",
    entries: [
      { binding: 0, visibility: STAGE_COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAMS_BYTES } },
      { binding: 1, visibility: STAGE_COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const module = device.createShaderModule({ label: "vdb dense fill", code: FILL_WGSL });
  const layout = device.createPipelineLayout({ label: "vdb dense fill", bindGroupLayouts: [group0] });
  const make = (entryPoint: string) => device.createComputePipeline({ label: `vdb dense fill ${entryPoint}`, layout, compute: { module, entryPoint } });
  const error = device.popErrorScope().then((e) => {
    if (e) console.error(`@pmndrs/vdb vdbDenseSource: the fill kernels are invalid: ${e.message}`);
    return e;
  });
  const p: Pipelines = { fill: make("vdbDenseFillCs"), mip: make("vdbDenseMipCs"), group0, error };
  pipelines.set(device, p);
  return p;
}

/** Writes `data` at `offset` (bytes) with `writeBuffer`'s 4-byte size rule: the tail goes through a padded word. */
function writeBytes(device: GPUDevice, buffer: GPUBuffer, offset: number, data: Uint8Array<ArrayBuffer>): void {
  const body = data.byteLength & ~3;
  if (body > 0) device.queue.writeBuffer(buffer, offset, data, 0, body);
  if (body < data.byteLength) {
    const tail = new Uint8Array(4);
    tail.set(data.subarray(body));
    device.queue.writeBuffer(buffer, offset + body, tail);
  }
}

/**
 * The GPU fill of one dense channel's textures (see the file comment). Keeps its buffer while the layout stays
 * (the pages of one LOD), and the coarse texels it last wrote (pages share one coarse array).
 */
export class VDBDenseFiller {
  readonly #device: GPUDevice;
  readonly #label: string;
  #layout: VDBDenseFillLayout | null = null;
  #buffer: GPUBuffer | null = null;
  #params: GPUBuffer | null = null;
  #group: GPUBindGroup | null = null;
  #dispatches: Dispatch[] = [];
  #coarse: Uint8Array | null = null;
  /** Fills run (for tests and probes). */
  runs = 0;

  constructor(device: GPUDevice, label: string) {
    this.#device = device;
    this.#label = label;
  }

  /** Pipeline validation of the kernels (null = valid). */
  get error(): Promise<GPUError | null> {
    return fillPipelines(this.#device).error;
  }

  /** Bytes of this filler's buffers. */
  get gpuBytes(): number {
    return (this.#buffer?.size ?? 0) + (this.#params?.size ?? 0);
  }

  /**
   * Fills `map` (and `mip`, the atlas texture, when `tex` has one) with `tex` and its pending bricks upsampled, and
   * submits the work. `tex.fill` must be set; `map`/`mip` must have `tex`'s sizes and `COPY_DST`.
   * @returns the buffers it replaced (the caller retires them).
   */
  fill(map: GPUTexture, mip: GPUTexture | null, tex: WvdbDenseTexture): GPUBuffer[] {
    const fill = tex.fill!;
    const device = this.#device;
    const layout = vdbDenseFillLayout(tex);
    const retired: GPUBuffer[] = [];
    if (!this.#layout || this.#layout.key !== layout.key) {
      if (this.#buffer) retired.push(this.#buffer, this.#params!);
      this.#layout = layout;
      this.#dispatches = dispatchesOf(layout);
      this.#buffer = device.createBuffer({ label: `${this.#label} fill`, size: layout.size, usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC });
      this.#params = device.createBuffer({ label: `${this.#label} fill params`, size: this.#dispatches.length * SLOT, usage: USAGE.UNIFORM | USAGE.COPY_DST });
      const bytes = new Uint8Array(this.#dispatches.length * SLOT);
      this.#dispatches.forEach((d, i) => bytes.set(new Uint8Array(d.params.buffer), i * SLOT));
      device.queue.writeBuffer(this.#params, 0, bytes);
      const p = fillPipelines(device);
      this.#group = device.createBindGroup({
        label: `${this.#label} fill group`,
        layout: p.group0,
        entries: [
          { binding: 0, resource: { buffer: this.#params, size: PARAMS_BYTES } },
          { binding: 1, resource: { buffer: this.#buffer } },
        ],
      });
      this.#coarse = null;
    }
    const buffer = this.#buffer!;
    const l = this.#layout;
    device.queue.writeBuffer(buffer, l.pendingByte, new Uint32Array(fill.pending.subarray(0, l.pendingWords)));
    if (this.#coarse !== fill.from.data) {
      writeBytes(device, buffer, l.coarseByte, fill.from.data);
      this.#coarse = fill.from.data;
    }
    writeBytes(device, buffer, l.fineByte, tex.data);
    const p = fillPipelines(device);
    const encoder = device.createCommandEncoder({ label: `${this.#label} fill` });
    encoder.clearBuffer(buffer, l.mapByte, l.size - l.mapByte);
    const pass = encoder.beginComputePass({ label: `${this.#label} fill` });
    this.#dispatches.forEach((d, i) => {
      pass.setPipeline(d.entry === "fill" ? p.fill : p.mip);
      pass.setBindGroup(0, this.#group!, [i * SLOT]);
      pass.dispatchWorkgroups(Math.ceil(d.dims[0] / W), Math.ceil(d.dims[1] / W), Math.ceil(d.dims[2] / W));
    });
    pass.end();
    const [w, h, d] = l.dims;
    encoder.copyBufferToTexture({ buffer, offset: l.mapByte, bytesPerRow: l.mapRow, rowsPerImage: h }, { texture: map }, [w, h, d]);
    if (l.atlas && mip) {
      const [aw, ah, ad] = l.atlas.dimensions;
      encoder.copyBufferToTexture({ buffer, offset: l.atlas.byte, bytesPerRow: l.atlas.row, rowsPerImage: ah }, { texture: mip }, [aw, ah, ad]);
    }
    device.queue.submit([encoder.finish()]);
    this.runs++;
    return retired;
  }

  /** The buffers this filler holds (the caller retires them), and forget them. */
  release(): GPUBuffer[] {
    const out = this.#buffer ? [this.#buffer, this.#params!] : [];
    this.#buffer = this.#params = null;
    this.#group = null;
    this.#layout = null;
    this.#coarse = null;
    return out;
  }
}
