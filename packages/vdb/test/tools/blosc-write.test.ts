/**
 * K7.C1: the LZ4 block compressor and the LZ4 Blosc1 frame writer.
 *
 * 1. Native parity: every case of `blosc-corpus.ts` compresses to the frame the Blosc NanoVDB v14 links
 *    (`nanovdb_blosc`, LZ4 1.9.4) writes, by SHA-256 against `fixtures/native-v14/blosc-frames.json`.
 *    With `VDB_NATIVE_BLOSC_FRAMES=<blosc_frames binary>` (generators/blosc_frames.c) the test runs the
 *    native compressor itself, compares bytes, has native decode our frames, and with
 *    `VDB_NATIVE_BLOSC_WRITE=1` rewrites the JSON.
 * 2. Round trips through our decoders (fuzzed: random and structured inputs, every size to 300, the
 *    64 KiB table switch, incompressible data, tight output budgets).
 * 3. `writeNvdb({ codec: 'blosc' })`: native's files from native's images, several grids, chunking.
 */

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { compressBlosc, shuffleBytes, type BloscCompressOptions } from "../../src/tools/vdb/blosc-write.js";
import { decompressBlosc, decompressLz4Block, readBloscHeader, unshuffle } from "../../src/tools/vdb/blosc.js";
import { compressLz4Block, lz4CompressBound } from "../../src/tools/vdb/lz4.js";
import { NvdbCodec, NvdbFile, NvdbStreamReader, nvdbBloscPayload, vdbBuildFromDense, writeNvdb } from "../../src/tools/internal.js";
import { bloscCorpus, bloscSweeps, nativeImages, rng, type BloscSweep } from "./blosc-corpus.js";

const FIXTURES = new URL("./fixtures/native-v14/", import.meta.url).pathname;
const JSON_PATH = join(FIXTURES, "blosc-frames.json");
const sha256 = (b: Uint8Array): string => createHash("sha256").update(b).digest("hex");
/** Byte equality without vitest's per-element diff (megabyte arrays). */
const same = (a: Uint8Array, b: Uint8Array): boolean => Buffer.from(a.buffer, a.byteOffset, a.byteLength).equals(Buffer.from(b.buffer, b.byteOffset, b.byteLength));

interface NativeFrame {
  /** Frame bytes, 0 when Blosc returned 0 (no frame). */
  size: number;
  sha256: string;
}

describe("LZ4 Blosc frames equal native NanoVDB v14 (nanovdb_blosc)", () => {
  const corpus = bloscCorpus(FIXTURES);
  const sweeps = bloscSweeps();
  const native = JSON.parse(readFileSync(JSON_PATH, "utf8")) as { frames: Record<string, NativeFrame> };
  /** A sweep's frames hashed together: per size a u32 frame length, then the frame (empty: Blosc returned 0). */
  const sweepDigest = (frames: Uint8Array[]): string => {
    const h = createHash("sha256");
    for (const f of frames) {
      const len = new Uint8Array(4);
      new DataView(len.buffer).setUint32(0, f.length, true);
      h.update(len).update(f);
    }
    return h.digest("hex");
  };
  const oursSweep = (s: BloscSweep): Uint8Array[] => s.destSizes.map((d) => compressBlosc(s.data, { ...s.opts, destSize: d }) ?? new Uint8Array(0));

  it("covers every corpus case and sweep", () => {
    expect(Object.keys(native.frames).sort()).toEqual([...corpus.map((c) => c.name), ...sweeps.map((s) => s.name)].sort());
  });

  it("every case: same frame (SHA-256) as native, and it decodes to the input", () => {
    const mismatched: string[] = [];
    for (const c of corpus) {
      const frame = compressBlosc(c.data, c.opts);
      const want = native.frames[c.name]!;
      const got = frame ?? new Uint8Array(0);
      if (got.length !== want.size || sha256(got) !== want.sha256) mismatched.push(`${c.name}: ${got.length} vs ${want.size}`);
      if (frame && !same(decompressBlosc(frame, { maxBytes: c.data.length }), c.data)) mismatched.push(`${c.name}: round trip`);
    }
    expect(mismatched).toEqual([]);
  });

  it("output budget sweeps: native's frame (or no frame) at every destSize", () => {
    for (const s of sweeps) {
      const frames = oursSweep(s);
      expect(sweepDigest(frames), s.name).toBe(native.frames[s.name]!.sha256);
      // The sweep crosses every outcome: no frame, a raw (memcpy'd) frame, and an LZ4 block.
      const kinds = new Set(frames.map((f) => (f.length === 0 ? "none" : f[2]! & 0x02 ? "memcpy" : f.length - 24 === s.data.length ? "raw" : "lz4")));
      expect([...kinds].sort(), s.name).toEqual(expect.arrayContaining(["lz4", "none"]));
    }
  });

  it("the corpus reaches every writer path: raw splits, memcpy'd frames, no frame, unsplit and split blocks", () => {
    const seen = new Set<string>();
    for (const c of corpus) {
      const frame = compressBlosc(c.data, c.opts);
      if (!frame) {
        seen.add("null");
        continue;
      }
      const h = readBloscHeader(frame);
      seen.add(h.flags & 0x02 ? "memcpy" : "compressed");
      if (!(h.flags & 0x02)) seen.add(h.flags & 0x10 ? "unsplit" : "split");
      if (!(h.flags & 0x02) && h.cbytes > 16 + 4 * Math.ceil(h.nbytes / h.blocksize) + h.nbytes * 0.9) seen.add("raw-splits");
    }
    expect([...seen].sort()).toEqual(["compressed", "memcpy", "null", "raw-splits", "split", "unsplit"]);
  });

  const bin = process.env.VDB_NATIVE_BLOSC_FRAMES;
  it.skipIf(!bin)("native compressor (VDB_NATIVE_BLOSC_FRAMES): byte-equal frames; native decodes ours", () => {
    const dir = mkdtempSync(join(tmpdir(), "vdb-blosc-"));
    try {
      const jobs: { key: string; data: Uint8Array; opts: BloscCompressOptions }[] = [
        ...corpus.map((c) => ({ key: c.name, data: c.data, opts: c.opts })),
        ...sweeps.flatMap((s) => s.destSizes.map((d) => ({ key: `${s.name}@${d}`, data: s.data, opts: { ...s.opts, destSize: d } }))),
      ];
      const inputs = new Map<Uint8Array, string>();
      const lines: string[] = [];
      jobs.forEach((j, i) => {
        let input = inputs.get(j.data);
        if (!input) {
          input = join(dir, `${inputs.size}.in`);
          writeFileSync(input, j.data);
          inputs.set(j.data, input);
        }
        const o = j.opts;
        const shuffle = o.shuffle ?? true ? 1 : 0;
        lines.push(`c ${o.clevel ?? 9} ${shuffle} ${o.typesize ?? 4} ${o.blocksize ?? 1 << 18} ${o.destSize ?? j.data.length + 16} ${input} ${dir}/${i}.native`);
        const ours = compressBlosc(j.data, j.opts);
        if (ours) {
          writeFileSync(join(dir, `${i}.ours`), ours);
          lines.push(`d ${dir}/${i}.ours ${dir}/${i}.decoded`);
        }
      });
      writeFileSync(join(dir, "manifest"), lines.join("\n") + "\n");
      execFileSync(bin!, [join(dir, "manifest")]);
      const nativeOf = new Map<string, Uint8Array>();
      jobs.forEach((j, i) => {
        const nat = new Uint8Array(readFileSync(join(dir, `${i}.native`)));
        nativeOf.set(j.key, nat);
        const ours = compressBlosc(j.data, j.opts) ?? new Uint8Array(0);
        expect(same(ours, nat), j.key).toBe(true);
        if (ours.length) expect(same(new Uint8Array(readFileSync(join(dir, `${i}.decoded`))), j.data), j.key).toBe(true);
      });
      if (process.env.VDB_NATIVE_BLOSC_WRITE === "1") {
        const frames: Record<string, NativeFrame> = {};
        for (const c of corpus) {
          const nat = nativeOf.get(c.name)!;
          frames[c.name] = { size: nat.length, sha256: sha256(nat) };
        }
        for (const s of sweeps) {
          frames[s.name] = { size: s.destSizes.length, sha256: sweepDigest(s.destSizes.map((d) => nativeOf.get(`${s.name}@${d}`)!)) };
        }
        const sorted = Object.fromEntries(Object.entries(frames).sort(([a], [b]) => (a < b ? -1 : 1)));
        const doc = {
          note:
            "K7.C1: SHA-256 of the Blosc frames nanovdb_blosc (OpenVDB v14.0.0, LZ4 1.9.4) writes for test/tools/blosc-corpus.ts " +
            "(a sweep: its frames hashed together, size = sizes swept); regenerate with VDB_NATIVE_BLOSC_FRAMES + VDB_NATIVE_BLOSC_WRITE=1",
          frames: sorted,
        };
        writeFileSync(JSON_PATH, JSON.stringify(doc, null, 1) + "\n");
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 600_000);
});

/** Our block compressor, then our decoder; checks the limited-output contract too. */
function roundTrip(src: Uint8Array, capacity = lz4CompressBound(src.length), accel = 1): number {
  const dest = new Uint8Array(capacity);
  const n = compressLz4Block(src, dest, accel);
  expect(n).toBeLessThanOrEqual(capacity);
  if (n === 0) {
    expect(capacity).toBeLessThan(lz4CompressBound(src.length));
    return 0;
  }
  const out = new Uint8Array(src.length);
  expect(decompressLz4Block(dest.subarray(0, n), out)).toBe(src.length);
  expect(same(out, src)).toBe(true);
  return n;
}

describe("compressLz4Block fuzz (round trips through decompressLz4Block)", () => {
  it("every size 0-300, random and constant, at every acceleration Blosc uses", () => {
    const r = rng(7);
    for (let n = 0; n <= 300; n++) {
      const rand = Uint8Array.from({ length: n }, () => (r() * 256) | 0);
      const cst = new Uint8Array(n).fill(n & 0xff);
      for (const accel of [1, 5, 9]) {
        roundTrip(rand, undefined, accel);
        roundTrip(cst, undefined, accel);
      }
    }
    // An empty input is the one-byte block 0; under 13 bytes everything is literals.
    const one = new Uint8Array(4);
    expect(compressLz4Block(new Uint8Array(0), one)).toBe(1);
    expect(one[0]).toBe(0);
    expect(roundTrip(new Uint8Array(12))).toBe(13);
  });

  it("structured inputs of random sizes, including both sides of the 64 KiB table switch", () => {
    const r = rng(11);
    for (let t = 0; t < 120; t++) {
      const n = t < 6 ? [65545, 65546, 65547, 65548, 131072, 200003][t]! : (r() * (t % 3 === 0 ? 300000 : 70000)) | 0;
      const src = new Uint8Array(n);
      const alphabet = 1 + ((r() * 255) | 0);
      let i = 0;
      while (i < n) {
        const mode = r();
        const len = (r() * (mode < 0.05 ? 3000 : 64)) | 0;
        if (mode < 0.4) {
          const v = (r() * alphabet) | 0;
          for (let k = 0; k < len && i < n; k++) src[i++] = v;
        } else if (mode < 0.7 && i > 0) {
          const from = (r() * i) | 0; // a copy from anywhere earlier (also > 65535 back)
          for (let k = 0; k < len && i < n; k++) src[i++] = src[from + k]!;
        } else {
          for (let k = 0; k < len && i < n; k++) src[i++] = (r() * alphabet) | 0;
        }
      }
      roundTrip(src, undefined, 1 + ((r() * 9) | 0));
    }
  });

  it("incompressible input: the bound holds, and a budget of the input size gives 0 (Blosc then stores it raw)", () => {
    for (const n of [13, 100, 4096, 65546, 65547, 300000]) {
      const src = new Uint8Array(n);
      const r = rng(n);
      for (let i = 0; i < n; i++) src[i] = (r() * 256) | 0;
      const size = roundTrip(src);
      expect(size).toBeGreaterThan(n);
      expect(size).toBeLessThanOrEqual(lz4CompressBound(n));
      expect(roundTrip(src, n)).toBe(0);
    }
  });

  it("tight budgets: every capacity either fits the block or returns 0, never a partial block", () => {
    const r = rng(3);
    const src = new Uint8Array(5000);
    for (let i = 0; i < src.length; i++) src[i] = i % 300 < 150 ? 7 : (r() * 8) | 0;
    const full = roundTrip(src);
    for (let cap = 0; cap <= full + 40; cap++) {
      const got = roundTrip(src, cap);
      expect(got === 0 || got === full).toBe(true);
    }
  });

  it("long matches and long literal runs (length chains past 15 + 255)", () => {
    const src = new Uint8Array(200000);
    const r = rng(5);
    for (let i = 0; i < 1000; i++) src[i] = (r() * 256) | 0; // 1000 literals: a 15 + 255 + 255 + ... run
    // then 199000 zeros: one match of length > 15 + 255 * 700
    const n = roundTrip(src);
    expect(n).toBeLessThan(2000);
  });
});

describe("Blosc writer pieces", () => {
  it("shuffleBytes is unshuffle's inverse, with a partial-element tail kept in place", () => {
    const r = rng(9);
    for (const typesize of [1, 2, 3, 4, 8, 16]) {
      for (const n of [0, 1, 7, 64, 1001]) {
        const src = Uint8Array.from({ length: n }, () => (r() * 256) | 0);
        const sh = new Uint8Array(n);
        shuffleBytes(src, sh, typesize);
        const back = new Uint8Array(n);
        unshuffle(sh, back, typesize);
        expect(back).toEqual(src);
      }
    }
    const sh = new Uint8Array(9);
    shuffleBytes(Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8]), sh, 4);
    expect([...sh]).toEqual([0, 4, 1, 5, 2, 6, 3, 7, 8]);
  });

  it("NanoVDB's request (2^18, typesize 4) gives 1 MiB blocks of four 256 KiB splits", () => {
    const src = new Uint8Array(3 * (1 << 20) + 1000);
    const h = readBloscHeader(compressBlosc(src)!);
    expect([h.version, h.versionlz, h.flags, h.typesize, h.nbytes, h.blocksize]).toEqual([2, 1, 0x21, 4, src.length, 1 << 20]);
  });

  it("rejects options Blosc rejects; no frame when destSize can't hold even the header", () => {
    expect(() => compressBlosc(new Uint8Array(10), { clevel: 10 })).toThrow(RangeError);
    expect(() => compressBlosc(new Uint8Array(10), { typesize: 0 })).toThrow(RangeError);
    expect(compressBlosc(new Uint8Array(10), { destSize: 15 })).toBeNull();
  });
});

describe("writeNvdb({ codec: 'blosc' }) (Codec::BLOSC as io::Internal::write)", () => {
  /** Offsets allowed to differ from a native file: FileHeader version; per grid FileMetaData nameKey and version. */
  function allowed(file: Uint8Array): Set<number> {
    const dv = new DataView(file.buffer, file.byteOffset, file.byteLength);
    const out = new Set([8, 9, 10, 11]);
    let off = 16;
    for (let g = 0; g < dv.getUint16(12, true); g++) {
      for (let k = 16; k < 24; k++) out.add(off + k);
      for (let k = 172; k < 176; k++) out.add(off + k);
      off += 176 + dv.getUint32(off + 136, true);
    }
    return out;
  }

  it.each(["sphere_fog_blosc.nvdb", "two_grids_blosc.nvdb"])(
    "%s: our file from native's images equals native's but for the versions and the name key",
    (name) => {
      const native = new Uint8Array(readFileSync(join(FIXTURES, name)));
      const images = nativeImages(join(FIXTURES, name)).map((b) => new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4));
      const ours = new Uint8Array(writeNvdb(images, { codec: "blosc" }));
      expect(ours.length).toBe(native.length);
      const skip = allowed(ours);
      const differ: number[] = [];
      for (let i = 0; i < ours.length; i++) if (ours[i] !== native[i] && !skip.has(i)) differ.push(i);
      expect(differ).toEqual([]);
    },
  );

  // A grid image over 1 MiB (two Blosc blocks, the second a leftover) next to a small one.
  const r = rng(21);
  const dense = new Float32Array(72 * 72 * 72).map((_, i) => (i % 72 < 40 ? Math.fround(Math.sin(i / 500) + r() * 1e-3) : 0));
  const big = vdbBuildFromDense(dense, [72, 72, 72], { background: 0, gridName: "big" });
  const small = vdbBuildFromDense(new Float32Array(8 * 8 * 8).fill(0.5), [8, 8, 8], { background: 0, gridName: "small" });
  const sameImage = (a: Uint32Array, b: Uint32Array): boolean => same(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), new Uint8Array(b.buffer, b.byteOffset, b.byteLength));

  it("round-trips several grids through NvdbFile and the stream reader; headers say BLOSC; fileSize is the stored bytes", () => {
    expect(big.byteLength).toBeGreaterThan(1 << 20);
    const file = new Uint8Array(writeNvdb([big, small], { codec: "blosc" }));
    const dv = new DataView(file.buffer);
    expect(dv.getUint16(14, true)).toBe(NvdbCodec.BLOSC);
    const parsed = NvdbFile.fromArrayBuffer(file.slice().buffer);
    expect(parsed.grids.map((g) => g.name)).toEqual(["big", "small"]);
    expect(sameImage(parsed.gridImage(0), big) && sameImage(parsed.gridImage(1), small)).toBe(true);
    // fileSize of each grid = its chunks' bytes; they add up to the file.
    const md1 = 16 + 176 + dv.getUint32(16 + 136, true);
    const sizes = [Number(dv.getBigUint64(16 + 8, true)), Number(dv.getBigUint64(md1 + 8, true))];
    expect(dv.getUint16(16 + 168, true)).toBe(NvdbCodec.BLOSC);
    expect(md1 + 176 + dv.getUint32(md1 + 136, true) + sizes[0]! + sizes[1]!).toBe(file.length);
    const reader = new NvdbStreamReader();
    const got = [];
    for (let at = 0; at < file.length; at += 4093) got.push(...reader.push(file.subarray(at, at + 4093)));
    reader.end();
    expect(got.map((g) => g.metadata.name)).toEqual(["big", "small"]);
    expect(sameImage(got[0]!.image, big)).toBe(true);
    // Smaller than codec NONE (the smooth grid compresses; native would write the same bytes).
    expect(file.length).toBeLessThan(writeNvdb([big, small]).byteLength / 2);
  });

  it("chunks: each is min(rest, chunkBytes) as a u64 size and one frame; any chunking reads back", () => {
    const bytes = new Uint8Array(big.buffer, big.byteOffset, big.byteLength);
    const chunk = 300_001;
    const payload = nvdbBloscPayload(bytes, chunk);
    const pv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    let off = 0;
    let filled = 0;
    while (off < payload.length) {
      const size = Number(pv.getBigUint64(off, true));
      const want = compressBlosc(bytes.subarray(filled, Math.min(bytes.length, filled + chunk)))!;
      expect(same(payload.subarray(off + 8, off + 8 + size), want)).toBe(true);
      filled += Math.min(chunk, bytes.length - filled);
      off += 8 + size;
    }
    expect([off, filled]).toEqual([payload.length, bytes.length]);
    // Splice the chunked payload into a BLOSC file in place of the one-chunk grid data.
    const file = new Uint8Array(writeNvdb([big], { codec: "blosc" }));
    const head = 16 + 176 + new DataView(file.buffer).getUint32(16 + 136, true);
    const spliced = new Uint8Array(head + payload.length);
    spliced.set(file.subarray(0, head));
    spliced.set(payload, head);
    new DataView(spliced.buffer).setBigUint64(16 + 8, BigInt(payload.length), true);
    expect(sameImage(NvdbFile.fromArrayBuffer(spliced.buffer).gridImage(0), big)).toBe(true);
    // Incompressible data: native's destSize (chunk + 16) leaves room only for a memcpy'd frame.
    const noise = Uint8Array.from({ length: 50_000 }, () => (r() * 256) | 0);
    const raw = nvdbBloscPayload(noise);
    expect(raw[8 + 2]! & 0x02).toBe(0x02);
    expect(same(raw.subarray(8), compressBlosc(noise, { destSize: noise.length + 16 })!)).toBe(true);
    expect(() => nvdbBloscPayload(bytes, 0)).toThrow(expect.objectContaining({ code: "VDB_BAD_OPTION" }));
    expect(() => nvdbBloscPayload(bytes, 2 ** 30 + 1)).toThrow(expect.objectContaining({ code: "VDB_BAD_OPTION" }));
  });
});
