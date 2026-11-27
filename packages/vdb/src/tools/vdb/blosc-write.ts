/**
 * blosc-write.ts — a Blosc1 frame writer with the LZ4 codec (K7.C1), for `writeNvdb({ codec: 'blosc' })`.
 * The reader is `decompressBlosc` in `blosc.ts`.
 *
 * Ported from the compression half of the Blosc OpenVDB v14.0.0 vendors
 * (`ext/blosc/openvdb/ext/blosc/vdb_blosc.c`, c-blosc 1.21.6 cut down to LZ4: `blosc_compress_ctx`,
 * `compute_blocksize`, `write_compression_header`, `serial_blosc`, `blosc_c`, `shuffle_block`):
 *
 *   Blosc - Blocked Shuffling and Compression Library
 *   Copyright (c) 2009-2018 Francesc Alted <francesc@blosc.org>
 *   Copyright (c) 2019-present Blosc Development Team <blosc@blosc.org>
 *   License: BSD 3-Clause (see THIRD_PARTY_NOTICES.md)
 *
 * Same block size rule (a forced block size is enlarged by `typesize` for split codecs, so NanoVDB's
 * requested 256 KiB is a 1 MiB block of four 256 KiB byte planes), same splits, same per-split output
 * budget (`maxout`), same fallbacks (a split LZ4 can't shrink is stored raw; a frame that doesn't fit
 * `destSize` becomes a memcpy'd frame). With `compressLz4Block` matching native LZ4 byte for byte, a frame
 * equals native's for the same input and options (DIVERGES V164). One thread (as NanoVDB asks for).
 */

import { compressLz4Block } from "./lz4.js";

const HEADER_SIZE = 16;
/** `BLOSC_VERSION_FORMAT`. */
const VERSION_FORMAT = 2;
/** `BLOSC_LZ4_VERSION_FORMAT`. */
const LZ4_VERSION_FORMAT = 1;
const FLAG_SHUFFLE = 0x01;
const FLAG_MEMCPYED = 0x02;
const LZ4_FORMAT = 1;
const MIN_BUFFERSIZE = 128;
const MAX_SPLITS = 16;
const L1 = 32 * 1024;
const MAX_TYPESIZE = 255;
/** `NANOVDB_BLOSC_MAX_BUFFERSIZE` (INT_MAX - 16). */
export const BLOSC_MAX_BUFFERSIZE = 0x7fffffff - HEADER_SIZE;

/** Options of {@link compressBlosc}; the defaults are NanoVDB's `io::Internal::write`. */
export interface BloscCompressOptions {
  /** 0–9; 0 stores a memcpy'd frame. Default 9 (LZ4 acceleration `10 − clevel`). */
  clevel?: number;
  /** Byte shuffle. Default true. */
  shuffle?: boolean;
  /** Element size for the shuffle and the splits, 1–255 (larger is treated as 1). Default 4. */
  typesize?: number;
  /** Requested block size (0: Blosc chooses from `clevel`). Default 2^18, NanoVDB's. */
  blocksize?: number;
  /** Capacity of the output, as the `destsize` passed to Blosc. Default `src.length + 16` (NanoVDB's). */
  destSize?: number;
}

function splitBlock(typesize: number, blocksize: number): boolean {
  return typesize <= MAX_SPLITS && Math.floor(blocksize / typesize) >= MIN_BUFFERSIZE;
}

/** `compute_blocksize`. */
function computeBlocksize(clevel: number, typesize: number, nbytes: number, forced: number): number {
  if (nbytes < typesize) return 1;
  let blocksize = nbytes;
  if (forced) {
    blocksize = Math.max(forced, MIN_BUFFERSIZE);
  } else if (nbytes >= L1) {
    blocksize = L1;
    if (clevel === 0) blocksize /= 4;
    else if (clevel === 1) blocksize /= 2;
    else if (clevel === 3) blocksize *= 2;
    else if (clevel === 4 || clevel === 5) blocksize *= 4;
    else if (clevel >= 6) blocksize *= 8;
  }
  if (clevel > 0 && splitBlock(typesize, blocksize)) {
    if (blocksize > 1 << 18) blocksize = 1 << 18;
    blocksize *= typesize;
    if (blocksize < 1 << 16) blocksize = 1 << 16;
    if (blocksize > 1024 * 1024) blocksize = 1024 * 1024;
  }
  if (blocksize > nbytes) blocksize = nbytes;
  if (blocksize > typesize) blocksize = Math.floor(blocksize / typesize) * typesize;
  return blocksize;
}

/** `shuffle_block`: byte `j` of element `i` goes to `j * elements + i`; a partial element's bytes stay at the end. */
export function shuffleBytes(src: Uint8Array, dest: Uint8Array, typesize: number): void {
  const n = src.length;
  const elements = Math.floor(n / typesize);
  for (let j = 0; j < typesize; j++) {
    const row = j * elements;
    for (let i = 0; i < elements; i++) dest[row + i] = src[i * typesize + j]!;
  }
  const tail = elements * typesize;
  dest.set(src.subarray(tail, n), tail);
}

/**
 * Compresses `src` into one Blosc1 frame (LZ4 codec) as `blosc_compress_ctx(clevel, shuffle, typesize,
 * src.length, src, dest, destSize, "lz4", blocksize, 1)` from OpenVDB v14's Blosc does, and returns the
 * frame (exactly its `cbytes` long), or `null` when Blosc returns 0 (the frame can't fit `destSize`).
 * Throws a `RangeError` on options Blosc rejects (a negative return there).
 */
export function compressBlosc(src: Uint8Array, opts: BloscCompressOptions = {}): Uint8Array | null {
  const clevel = opts.clevel ?? 9;
  const doShuffle = opts.shuffle ?? true;
  let typesize = opts.typesize ?? 4;
  const forced = opts.blocksize ?? 1 << 18;
  const nbytes = src.length;
  const destSize = opts.destSize ?? nbytes + HEADER_SIZE;
  if (!Number.isInteger(clevel) || clevel < 0 || clevel > 9) throw new RangeError(`blosc clevel must be 0-9, got ${clevel}`);
  if (!Number.isInteger(typesize) || typesize < 1) throw new RangeError(`blosc typesize must be a positive integer, got ${typesize}`);
  if (!Number.isInteger(forced) || forced < 0) throw new RangeError(`blosc blocksize must be a non-negative integer, got ${forced}`);
  if (!Number.isInteger(destSize) || destSize < 0) throw new RangeError(`blosc destSize must be a non-negative integer`);
  // initialize_context_compression: size limits return 0 (no frame).
  if (nbytes > BLOSC_MAX_BUFFERSIZE || destSize < HEADER_SIZE) return null;
  if (typesize > MAX_TYPESIZE) typesize = 1;

  const blocksize = computeBlocksize(clevel, typesize, nbytes, forced);
  const leftover = nbytes % blocksize;
  const nblocks = Math.floor(nbytes / blocksize) + (leftover > 0 ? 1 : 0);

  // write_compression_header.
  // Room for the block-start table even when `destSize` is smaller (Blosc writes it before it finds the frame
  // can't fit, past its buffer; here it lands in scratch and the call returns null).
  const dest = new Uint8Array(
    Math.max(HEADER_SIZE + 4 * nblocks, Math.min(destSize, nbytes + HEADER_SIZE + 4 * nblocks + 4 * nblocks * MAX_SPLITS + 64)),
  );
  const view = new DataView(dest.buffer);
  let flags = 0;
  let outBytes = HEADER_SIZE + 4 * nblocks;
  if (clevel === 0 || nbytes < MIN_BUFFERSIZE) {
    flags |= FLAG_MEMCPYED;
    outBytes = HEADER_SIZE;
  }
  if (doShuffle) flags |= FLAG_SHUFFLE;
  if (!splitBlock(typesize, blocksize)) flags |= 0x10;
  flags |= LZ4_FORMAT << 5;

  // blosc_compress_context.
  if (flags & FLAG_MEMCPYED && nbytes + HEADER_SIZE > destSize) return null;
  let ntbytes = flags & FLAG_MEMCPYED ? -1 : compressBlocks(src, dest, view, flags, typesize, blocksize, nblocks, leftover, clevel, outBytes, destSize);
  if (ntbytes === 0 && nbytes + HEADER_SIZE <= destSize) flags |= FLAG_MEMCPYED;
  if (flags & FLAG_MEMCPYED && ntbytes <= 0) {
    if (nbytes + HEADER_SIZE > dest.length) return null;
    dest.set(src, HEADER_SIZE);
    ntbytes = HEADER_SIZE + nbytes;
  }
  if (ntbytes <= 0) return null;

  dest[0] = VERSION_FORMAT;
  dest[1] = LZ4_VERSION_FORMAT;
  dest[2] = flags;
  dest[3] = typesize;
  view.setInt32(4, nbytes, true);
  view.setInt32(8, blocksize, true);
  view.setInt32(12, ntbytes, true);
  return dest.slice(0, ntbytes);
}

/** `serial_blosc` (compression, not memcpy'd): the bytes written, or 0 when the frame overruns `maxbytes`. */
function compressBlocks(
  src: Uint8Array,
  dest: Uint8Array,
  view: DataView,
  flags: number,
  typesize: number,
  blocksize: number,
  nblocks: number,
  leftover: number,
  clevel: number,
  start: number,
  maxbytes: number,
): number {
  const tmp = new Uint8Array(blocksize);
  const doShuffle = (flags & FLAG_SHUFFLE) !== 0 && typesize > 1;
  const dontSplit = (flags & 0x10) !== 0;
  const accel = 10 - clevel;
  let ntbytes = start;
  for (let j = 0; j < nblocks; j++) {
    view.setInt32(HEADER_SIZE + 4 * j, ntbytes, true);
    const isLeftover = j === nblocks - 1 && leftover > 0;
    const bsize = isLeftover ? leftover : blocksize;
    const block = src.subarray(j * blocksize, j * blocksize + bsize);
    let data = block;
    if (doShuffle) {
      shuffleBytes(block, tmp.subarray(0, bsize), typesize);
      data = tmp.subarray(0, bsize);
    }
    // blosc_c.
    const nsplits = !dontSplit && !isLeftover ? typesize : 1;
    const neblock = Math.floor(bsize / nsplits);
    for (let s = 0; s < nsplits; s++) {
      const sizeAt = ntbytes;
      ntbytes += 4;
      let maxout = neblock;
      if (ntbytes + maxout > maxbytes) {
        maxout = maxbytes - ntbytes;
        if (maxout <= 0) return 0;
      }
      const part = data.subarray(s * neblock, (s + 1) * neblock);
      let cbytes = compressLz4Block(part, dest.subarray(ntbytes, ntbytes + maxout), accel);
      if (cbytes === 0 || cbytes === neblock) {
        if (ntbytes + neblock > maxbytes) return 0;
        dest.set(part, ntbytes);
        cbytes = neblock;
      }
      view.setInt32(sizeAt, cbytes, true);
      ntbytes += cbytes;
    }
  }
  return ntbytes;
}
