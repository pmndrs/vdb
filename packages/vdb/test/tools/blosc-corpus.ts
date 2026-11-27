/**
 * K7.C1: the deterministic inputs `blosc-write.test.ts` compresses with our LZ4 Blosc writer and, through
 * `fixtures/native-v14/generators/blosc_frames.c`, with the Blosc NanoVDB v14 links. Native frame hashes
 * live in `fixtures/native-v14/blosc-frames.json`; changing a case here changes its hash.
 */

import { readFileSync } from "node:fs";

import type { BloscCompressOptions } from "../../src/tools/vdb/blosc-write.js";
import { decompressBlosc } from "../../src/tools/vdb/blosc.js";
import { compressLz4Block, lz4CompressBound } from "../../src/tools/vdb/lz4.js";

export interface BloscCase {
  name: string;
  data: Uint8Array;
  opts: BloscCompressOptions;
}

/** One input compressed at every `destSize` listed (hashed together): sweeps LZ4's output budget across its checks. */
export interface BloscSweep {
  name: string;
  data: Uint8Array;
  opts: BloscCompressOptions;
  destSizes: number[];
}

/**
 * Budget sweeps: one unsplit block (typesize 1, no shuffle) so Blosc hands LZ4 `destSize − 24` bytes, swept
 * over every size (small inputs, 16-bit table) or over the last stretch below and above the block's
 * unlimited size (a 32-bit-table input). Where LZ4's conservative checks give up, Blosc stores the split
 * raw or returns no frame: the frames must equal native's at every size.
 */
export function bloscSweeps(): BloscSweep[] {
  const opts: BloscCompressOptions = { typesize: 1, shuffle: false, blocksize: 0 };
  const border = (n: number, p: number, seed: number): Uint8Array => {
    const r = rng(seed);
    const data = random(n, seed + 1);
    for (let i = 16; i < n - 16; i++) if (r() < p) data.copyWithin(i, i - 1 - ((r() * 12) | 0), i + 4 + ((r() * 6) | 0));
    return data;
  };
  const every = (n: number): number[] => Array.from({ length: n + 17 }, (_, i) => i);
  const out: BloscSweep[] = [
    { name: "sweep-border-3000", data: border(3000, 0.03, 901), opts, destSizes: every(3000) },
    { name: "sweep-text-3000", data: textLike(3000, 903), opts, destSizes: every(3000) },
    { name: "sweep-runs-3000", data: runs(3000, 905), opts, destSizes: every(3000) },
  ];
  // 32-bit table: 66000 bytes is one block (blocksize = nbytes) of one split.
  const big = border(66000, 0.05, 907);
  const unlimited = compressLz4Block(big, new Uint8Array(lz4CompressBound(big.length)));
  const sizes: number[] = [];
  for (let d = Math.min(unlimited, big.length) + 24 - 400; d <= Math.min(unlimited, big.length) + 24 + 40; d++) sizes.push(d);
  out.push({ name: "sweep-border-66000", data: big, opts, destSizes: sizes });
  return out;
}

/** mulberry32: a small seeded generator, so every run (and the native one) sees the same bytes. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const zeros = (n: number): Uint8Array => new Uint8Array(n);

function random(n: number, seed: number): Uint8Array {
  const r = rng(seed);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (r() * 256) | 0;
  return out;
}

/** Smooth f32 field (a fog volume's leaf values): long equal-exponent runs, noisy mantissas. */
function floatRamp(n: number, seed: number): Uint8Array {
  const r = rng(seed);
  const f = new Float32Array(Math.ceil(n / 4));
  for (let i = 0; i < f.length; i++) f[i] = Math.max(0, Math.sin(i / 97) * Math.cos(i / 13)) + r() * 1e-3;
  return new Uint8Array(f.buffer, 0, n);
}

/** Mostly zero f32 values with random clusters (a sparse grid's inactive space and leaves). */
function sparseFloats(n: number, seed: number): Uint8Array {
  const r = rng(seed);
  const f = new Float32Array(Math.ceil(n / 4));
  let i = 0;
  while (i < f.length) {
    i += (r() * 600) | 0;
    const len = (r() * 300) | 0;
    for (let k = 0; k < len && i < f.length; k++, i++) f[i] = r() < 0.2 ? 1 : r();
  }
  return new Uint8Array(f.buffer, 0, n);
}

/** One random period repeated: periods over 65535 can't be matched (LZ4's distance limit). */
function repeated(n: number, period: number, seed: number): Uint8Array {
  const base = random(period, seed);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = base[i % period]!;
  return out;
}

/** Runs of one byte (some longer than 15 + 255, the match-length chains) between random stretches. */
function runs(n: number, seed: number): Uint8Array {
  const r = rng(seed);
  const out = new Uint8Array(n);
  let i = 0;
  while (i < n) {
    const len = r() < 0.1 ? (r() * 5000) | 0 : (r() * 40) | 0;
    const v = r() < 0.5 ? 0 : (r() * 256) | 0;
    for (let k = 0; k < len && i < n; k++) out[i++] = v;
    const lit = (r() * 30) | 0;
    for (let k = 0; k < lit && i < n; k++) out[i++] = (r() * 256) | 0;
  }
  return out;
}

/** A small alphabet with short repeats: many short matches, catch-ups and literal runs. */
function textLike(n: number, seed: number): Uint8Array {
  const r = rng(seed);
  const words = Array.from({ length: 64 }, () => random(3 + ((r() * 9) | 0), (r() * 1e9) | 0));
  const out = new Uint8Array(n);
  let i = 0;
  while (i < n) {
    const w = words[(r() * r() * words.length) | 0]!;
    for (let k = 0; k < w.length && i < n; k++) out[i++] = 97 + (w[k]! % 26);
    if (i < n) out[i++] = 32;
  }
  return out;
}

/** The grid images inside a native BLOSC `.nvdb` (the committed fixtures). */
export function nativeImages(path: string): Uint8Array[] {
  const b = new Uint8Array(readFileSync(path));
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const n = dv.getUint16(12, true);
  let off = 16;
  for (let i = 0; i < n; i++) off += 176 + dv.getUint32(off + 136, true);
  const out: Uint8Array[] = [];
  for (let i = 0; i < n; i++) {
    const size = Number(dv.getBigUint64(off, true));
    out.push(decompressBlosc(b.subarray(off + 8, off + 8 + size)));
    off += 8 + size;
  }
  return out;
}

const MIB = 1 << 20;
/** Split sizes on either side of LZ4_64Klimit (65547): the 16-bit table below it, the 32-bit one from it. */
const SPLIT_EDGE = [65546, 65547];

export function bloscCorpus(fixtureDir: string): BloscCase[] {
  const cases: BloscCase[] = [];
  const add = (name: string, data: Uint8Array, opts: BloscCompressOptions = {}): void => {
    cases.push({ name, data, opts });
  };
  // NanoVDB's own call (clevel 9, shuffle, typesize 4, 2^18 requested) on sizes around every rule edge.
  const sizes = [0, 1, 3, 4, 5, 12, 13, 127, 128, 129, 511, 512, 513, 1000, 4099, 65536, ...SPLIT_EDGE.map((s) => 4 * s), 262144, MIB, MIB + 5, 3 * MIB + 1000];
  const kinds: [string, (n: number, seed: number) => Uint8Array][] = [
    ["zeros", (n) => zeros(n)],
    ["random", random],
    ["ramp", floatRamp],
    ["sparse", sparseFloats],
    ["runs", runs],
    ["text", textLike],
  ];
  let seed = 1;
  for (const n of sizes) {
    for (const [kind, make] of kinds) {
      if (n > MIB + 5 && (kind === "zeros" || kind === "text")) continue;
      add(`${kind}-${n}`, make(n, seed++));
    }
  }
  add("period-70000", repeated(3 * MIB, 70000, seed++));
  add("period-65535", repeated(MIB, 65535, seed++));
  add("period-1000", repeated(300000, 1000, seed++));
  // Other parameters Blosc takes: typesize (no split past 16), shuffle off, levels (accel 10 - clevel), auto blocks.
  const sparse = sparseFloats(2 * MIB + 12, seed++);
  const text = textLike(400000, seed++);
  for (const typesize of [1, 2, 8, 16, 17]) add(`sparse-ts${typesize}`, sparse, { typesize });
  add("sparse-noshuffle", sparse, { shuffle: false });
  add("text-ts1-noshuffle", text, { typesize: 1, shuffle: false });
  for (const clevel of [0, 1, 5, 8]) add(`sparse-clevel${clevel}`, sparse, { clevel });
  for (const clevel of [1, 3, 5, 9]) add(`text-auto-clevel${clevel}`, text, { clevel, blocksize: 0 });
  add("sparse-bs128", sparse.subarray(0, 50000), { blocksize: 128 });
  add("sparse-bs4096", sparse.subarray(0, 50000), { blocksize: 4096 });
  // Output budgets: a frame that only fits compressed, one that fits nowhere, a memcpy'd fallback.
  add("text-dest-half", text, { destSize: 200000 });
  add("random-dest-tight", random(100000, seed++), { destSize: 100000 + 15 });
  add("random-dest-exact", random(100000, seed++), { destSize: 100000 + 16 });
  add("ramp-dest-small", floatRamp(100000, seed++), { destSize: 1000 });
  // Borderline data (random bytes with a few short repeats): LZ4's output lands near the input size, so its
  // conservative budget checks (literals, match, last literals) decide between a block and a raw split.
  for (const p of [0.004, 0.01, 0.02, 0.04]) {
    for (const n of [20000, 70000]) {
      const r = rng(seed++);
      const data = random(n, seed++);
      for (let i = 16; i < n - 16; i++) if (r() < p) data.copyWithin(i, i - 1 - ((r() * 12) | 0), i + 4 + ((r() * 6) | 0));
      add(`border-${p}-${n}`, data, { typesize: 1, shuffle: false });
    }
  }
  for (const f of [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]) add(`text-dest-${f}`, text, { destSize: Math.round(400000 * f) });
  // Real grids: NanoVDB images from the committed native fixtures.
  nativeImages(`${fixtureDir}/two_grids_blosc.nvdb`).forEach((img, i) => add(`nanovdb-image-${i}`, img));
  return cases;
}
