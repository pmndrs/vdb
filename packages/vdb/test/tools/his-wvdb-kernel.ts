// mjurczyk's `.wvdb` worker kernel re-implemented from his source (read, not executed), shared by the K3.0
// upload test (wvdb-dense.test.ts) and the K4.22 partial-LOD test (wvdb-partial.test.ts). Not a test file.
// Source: mjurczyk/openvdb 75e7345, src/openvdb/wvdb/format.js (decodeWVDBPage L145–209, decodeWVDBMask L319–328)
// and src/three-openvdb/wvdb/wvdbKernel.js (L4–344); see the line citations below.
import type { WvdbVec3 } from "../../src/tools/internal.js";

export async function inflateLikeHisKernel(bytes: Uint8Array): Promise<Uint8Array> {
  // wvdbKernel.js L14–18
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// format.js L145–209
export function hisDecodePage(bytes: Uint8Array, { layout, bits, firstIndex = 0, channels = 1 }: { layout: string; bits: number; firstIndex?: number; channels?: number }) {
  const voxels = 512;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = view.getUint32(0, true);
  const maxCode = (1 << bits) - 1;
  let offset = 4;
  const indices = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    if (layout === "sparse") {
      indices[i] = view.getUint32(offset, true);
      offset += 4;
    } else indices[i] = firstIndex + i;
  }
  const ranges = count * channels;
  const mins = new Float32Array(ranges);
  const maxs = new Float32Array(ranges);
  for (let i = 0; i < ranges; i++, offset += 4) mins[i] = view.getFloat32(offset, true);
  for (let i = 0; i < ranges; i++, offset += 4) maxs[i] = view.getFloat32(offset, true);
  const values = new Float32Array(ranges * voxels);
  for (let i = 0; i < ranges; i++) {
    const min = mins[i]!;
    const max = maxs[i]!;
    const target = i * voxels;
    if (min === max) {
      values.fill(min, target, target + voxels);
      continue;
    }
    const step = (max - min) / maxCode;
    for (let voxel = 0; voxel < voxels; voxel++) {
      let code: number;
      if (bits === 16) {
        code = view.getUint16(offset, true);
        offset += 2;
      } else {
        code = bytes[offset]!;
        offset += 1;
      }
      values[target + voxel] = code === maxCode ? max : min + code * step;
    }
  }
  return { indices, values };
}

// format.js L319–328
export function hisDecodeMask(bytes: Uint8Array): Uint32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const words = new Uint32Array(bytes.byteLength >>> 2);
  for (let i = 0; i < words.length; i++) words[i] = view.getUint32(i * 4, true);
  return words;
}

export const hisToByte = (value: number) => (Math.min(Math.max(value, 0.0), 1.0) * 255.0) | 0; // WVDBReader.js L128, wvdbKernel.js L20

export interface HisKernelGrid {
  class: string;
  background: any;
  channels: number;
  scale: number;
  backgroundTexel: number[];
  lods: { dimensions: WvdbVec3; layout: string; pages: number }[];
}

/** wvdbKernel.js L4–344, as one closure per file. Returns the texture data each grid ends with per LOD. */
export function hisKernel(bits: number, grids: HisKernelGrid[]) {
  const brickDim = 8;
  const brickVoxels = 512;
  const levels: Record<string, any> = {};
  const previousLevels: Record<string, any> = {};
  // L24–58
  const toTexels = (values: Float32Array, count: number, { class: gridClass, background, channels, scale }: HisKernelGrid) => {
    if (channels === 3) {
      const texels = new Uint8Array(count * brickVoxels * 4);
      for (let brick = 0; brick < count; brick++) {
        const source = brick * brickVoxels * 3;
        const target = brick * brickVoxels * 4;
        for (let voxel = 0; voxel < brickVoxels; voxel++) {
          for (let channel = 0; channel < 3; channel++) texels[target + voxel * 4 + channel] = hisToByte(values[source + channel * brickVoxels + voxel]!);
          texels[target + voxel * 4 + 3] = 255;
        }
      }
      return texels;
    }
    const texels = new Uint8Array(values.length);
    const isLevelSet = gridClass === "level set";
    for (let i = 0; i < values.length; i++) texels[i] = isLevelSet ? (Math.abs(values[i]!) < background ? 255 : 0) : hisToByte(values[i]! / scale);
    return texels;
  };
  // L76–86
  const fillTexels = (target: Uint8Array, texel: number[]) => {
    if (texel.length === 1) return target.fill(texel[0]!);
    for (let i = 0; i < target.length; i += texel.length) target.set(texel, i);
    return target;
  };
  // L89–119
  const writeBricks = (target: Uint8Array, [width, height, depth]: number[], indices: ArrayLike<number>, texels: Uint8Array, texelSize = 1) => {
    const countX = Math.ceil(width! / brickDim);
    const countY = Math.ceil(height! / brickDim);
    for (let i = 0; i < indices.length; i++) {
      const index = indices[i]!;
      const originX = (index % countX) * brickDim;
      const originY = (Math.floor(index / countX) % countY) * brickDim;
      const originZ = Math.floor(index / (countX * countY)) * brickDim;
      let voxel = i * brickVoxels;
      for (let z = 0; z < brickDim; z++) {
        for (let y = 0; y < brickDim; y++) {
          const row = (originY + y) * width! + (originZ + z) * width! * height!;
          const inside = originY + y < height! && originZ + z < depth!;
          for (let x = 0; x < brickDim; x++, voxel++) {
            if (inside && originX + x < width!) {
              const targetOffset = (originX + x + row) * texelSize;
              const sourceOffset = voxel * texelSize;
              for (let component = 0; component < texelSize; component++) target[targetOffset + component] = texels[sourceOffset + component]!;
            }
          }
        }
      }
    }
  };
  // L122–188
  const upsample = (source: Uint8Array, [sw, sh, sd]: number[], target: Uint8Array, [width, height, depth]: number[], texelSize: number) => {
    const sourceSlice = sw! * sh!;
    const axis = (size: number, sourceSize: number) => {
      const lower = new Int32Array(size);
      const upper = new Int32Array(size);
      const weight = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const position = Math.min(Math.max(((i + 0.5) / size) * sourceSize - 0.5, 0.0), sourceSize - 1);
        lower[i] = Math.floor(position);
        upper[i] = Math.min(lower[i]! + 1, sourceSize - 1);
        weight[i] = position - lower[i]!;
      }
      return { lower, upper, weight };
    };
    const ax = axis(width!, sw!);
    const ay = axis(height!, sh!);
    const az = axis(depth!, sd!);
    let offset = 0;
    for (let z = 0; z < depth!; z++) {
      const z0 = az.lower[z]! * sourceSlice;
      const z1 = az.upper[z]! * sourceSlice;
      const wz = az.weight[z]!;
      for (let y = 0; y < height!; y++) {
        const y0 = ay.lower[y]! * sw!;
        const y1 = ay.upper[y]! * sw!;
        const wy = ay.weight[y]!;
        for (let x = 0; x < width!; x++) {
          const x0 = ax.lower[x]!;
          const x1 = ax.upper[x]!;
          const wx = ax.weight[x]!;
          for (let component = 0; component < texelSize; component++) {
            const at = (index: number) => source[index * texelSize + component]!;
            const c00 = at(x0 + y0 + z0) + (at(x1 + y0 + z0) - at(x0 + y0 + z0)) * wx;
            const c10 = at(x0 + y1 + z0) + (at(x1 + y1 + z0) - at(x0 + y1 + z0)) * wx;
            const c01 = at(x0 + y0 + z1) + (at(x1 + y0 + z1) - at(x0 + y0 + z1)) * wx;
            const c11 = at(x0 + y1 + z1) + (at(x1 + y1 + z1) - at(x0 + y1 + z1)) * wx;
            const c0 = c00 + (c10 - c00) * wy;
            const c1 = c01 + (c11 - c01) * wy;
            target[offset++] = c0 + (c1 - c0) * wz + 0.5;
          }
        }
      }
    }
  };
  const getTexelSize = (grid: HisKernelGrid) => grid.backgroundTexel.length;
  // L193–211 (no level-set distance texture: it feeds only the surface renderer)
  const clearMissingBricks = (level: any, grid: HisKernelGrid, dimensions: number[]) => {
    const texelSize = getTexelSize(grid);
    const background = fillTexels(new Uint8Array(brickVoxels * texelSize), grid.backgroundTexel);
    const total = Math.ceil(dimensions[0]! / brickDim) * Math.ceil(dimensions[1]! / brickDim) * Math.ceil(dimensions[2]! / brickDim);
    for (let index = 0; index < total; index++) {
      if (!(level.mask[index >>> 5] & (1 << (index & 31)))) writeBricks(level.data, dimensions, [index], background, texelSize);
    }
  };
  // L230–344 (volume chunks)
  const chunk = async ({ gridIndex, lod: lodIndex, kind, firstIndex, bytes }: { gridIndex: number; lod: number; kind: string; firstIndex?: number; bytes: Uint8Array }) => {
    const grid = grids[gridIndex]!;
    const lod = grid.lods[lodIndex]!;
    const gridKey = `${gridIndex}`;
    const key = `${gridKey}:${lodIndex}`;
    const texelSize = getTexelSize(grid);
    const [width, height, depth] = lod.dimensions;
    const data = await inflateLikeHisKernel(bytes);
    if (!levels[key]) {
      levels[key] = { data: fillTexels(new Uint8Array(width * height * depth * texelSize), grid.backgroundTexel), pagesLeft: lod.pages, mask: null };
    }
    const level = levels[key];
    if (kind === "mask") level.mask = hisDecodeMask(data);
    else {
      const previous = previousLevels[gridKey];
      if (previous && level.pagesLeft === lod.pages) {
        upsample(previous.data, previous.dimensions, level.data, lod.dimensions, texelSize);
        delete previousLevels[gridKey];
        if (level.mask) clearMissingBricks(level, grid, lod.dimensions);
      }
      const { indices, values } = hisDecodePage(data, { layout: lod.layout, bits, firstIndex: firstIndex ?? 0, channels: grid.channels });
      writeBricks(level.data, lod.dimensions, indices, toTexels(values, indices.length, grid), texelSize);
      level.pagesLeft--;
    }
    const complete = level.pagesLeft === 0;
    if (!complete && (kind === "mask" || lod.pages === 1)) return null;
    if (complete) {
      delete levels[key];
      const nextLod = grid.lods[lodIndex + 1];
      if (nextLod && nextLod.pages > 1) previousLevels[gridKey] = { data: level.data.slice(), dimensions: lod.dimensions };
    }
    return { gridIndex, lod: lodIndex, complete, data: complete ? level.data : level.data.slice() };
  };
  return { chunk };
}

