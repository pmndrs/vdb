/**
 * lz4.ts — an LZ4 block compressor (K7.C1), so `writeNvdb({ codec: 'blosc' })` writes the LZ4 Blosc frames
 * NanoVDB's `io::Internal::write` writes. The decoder is `decompressLz4Block` in `blosc.ts`.
 *
 * Ported from `LZ4_compress_generic_validated` of the LZ4 library OpenVDB v14.0.0 vendors
 * (`ext/blosc/openvdb/ext/blosc/lz4/lz4.c`, LZ4 1.9.4), for the one case Blosc calls it with: one
 * independent block (no dictionary, no stream state), `LZ4_compress_fast` on a 64-bit host:
 *
 *   LZ4 Library
 *   Copyright (c) 2011-2020, Yann Collet
 *   License: BSD 2-Clause (see THIRD_PARTY_NOTICES.md)
 *
 * The match finder, its hash functions (`LZ4_hash4` for inputs under 64 KiB + 11, the 64-bit
 * `LZ4_hash5` otherwise), the 2^14-byte table (LZ4_MEMORY_USAGE 14), the skip schedule
 * (`acceleration << 6`), the backward catch-up, and the conservative `limitedOutput` budget checks are
 * the C code's, so a block compresses to the same bytes as native LZ4 (DIVERGES V164): Blosc then stores
 * a split raw exactly where native does. Pointers become indices into `src`; a 32-bit host's `byPtr`
 * table (and its `LZ4_hash4` on large inputs) is not ported.
 */

/** `LZ4_MAX_INPUT_SIZE`. */
export const LZ4_MAX_INPUT_SIZE = 0x7e000000;
const MINMATCH = 4;
const LASTLITERALS = 5;
const MFLIMIT = 12;
const LZ4_MIN_LENGTH = MFLIMIT + 1;
const ML_MASK = 15;
const RUN_MASK = 15;
/** `LZ4_64Klimit`: inputs below it use the 16-bit table and `LZ4_hash4`. */
const LZ4_64K_LIMIT = 65536 + (MFLIMIT - 1);
const LZ4_SKIP_TRIGGER = 6;
const LZ4_DISTANCE_MAX = 65535;
const LZ4_ACCELERATION_MAX = 65537;
/** LZ4_MEMORY_USAGE 14: 2^12 u32 slots, or 2^13 u16 slots for small inputs. */
const LZ4_HASHLOG = 12;

const PRIME4 = 2654435761 | 0;
/** `prime5bytes` = 889523592379 = 0xCF_1BBCDCBB, split in 32-bit halves. */
const PRIME5_LO = 0x1bbcdcbb;
const PRIME5_HI = 0xcf;

let table32: Uint32Array | null = null;
let table16: Uint16Array | null = null;

/** `LZ4_compressBound`: the largest output of a block of `n` bytes (0 past `LZ4_MAX_INPUT_SIZE`). */
export function lz4CompressBound(n: number): number {
  return n > LZ4_MAX_INPUT_SIZE || n < 0 ? 0 : n + Math.floor(n / 255) + 16;
}

/**
 * Compresses `src` into one raw LZ4 block (no frame) at the start of `dest`, as `LZ4_compress_fast(src,
 * dest, src.length, dest.length, acceleration)` does, and returns the block's length, or 0 when it does
 * not fit in `dest` (LZ4's limited-output checks, which give up a few bytes before the exact end, apply
 * whenever `dest` is shorter than {@link lz4CompressBound}). An empty input is the one-byte block `0`.
 */
export function compressLz4Block(src: Uint8Array, dest: Uint8Array, acceleration = 1): number {
  const inputSize = src.length;
  if (inputSize > LZ4_MAX_INPUT_SIZE) return 0;
  let accel = Math.floor(acceleration);
  if (!(accel >= 1)) accel = 1;
  if (accel > LZ4_ACCELERATION_MAX) accel = LZ4_ACCELERATION_MAX;
  const limited = dest.length < lz4CompressBound(inputSize);
  const olimit = dest.length;
  if (inputSize === 0) {
    if (limited && olimit <= 0) return 0;
    dest[0] = 0;
    return 1;
  }

  const small = inputSize < LZ4_64K_LIMIT;
  let t16: Uint16Array | null = null;
  let t32: Uint32Array | null = null;
  if (small) {
    t16 = table16 ??= new Uint16Array(1 << (LZ4_HASHLOG + 1));
    t16.fill(0);
  } else {
    t32 = table32 ??= new Uint32Array(1 << LZ4_HASHLOG);
    t32.fill(0);
  }
  const hash = small
    ? (p: number): number =>
        Math.imul(src[p]! | (src[p + 1]! << 8) | (src[p + 2]! << 16) | (src[p + 3]! << 24), PRIME4) >>> (32 - (LZ4_HASHLOG + 1))
    : (p: number): number => {
        // ((read64(p) << 24) * prime5bytes) >> (64 - 12): only bytes p..p+4 reach the product.
        // x = hi:lo with lo = b0 << 24, hi = b1 | b2 << 8 | b3 << 16 | b4 << 24.
        const b0 = src[p]!;
        const hi = src[p + 1]! | (src[p + 2]! << 8) | (src[p + 3]! << 16) | (src[p + 4]! << 24);
        const carry = Math.floor((b0 * PRIME5_LO) / 256); // high word of (b0 << 24) * PRIME5_LO (< 2^37 / 2^8)
        const top = (Math.imul(hi, PRIME5_LO) + Math.imul(b0 << 24, PRIME5_HI) + carry) | 0;
        return top >>> (32 - LZ4_HASHLOG);
      };
  const get = small ? (h: number): number => t16![h]! : (h: number): number => t32![h]!;
  const put = small
    ? (idx: number, h: number): void => {
        t16![h] = idx;
      }
    : (idx: number, h: number): void => {
        t32![h] = idx;
      };
  const read32 = (p: number): number => src[p]! | (src[p + 1]! << 8) | (src[p + 2]! << 16) | (src[p + 3]! << 24);

  const iend = inputSize;
  const mflimitPlusOne = iend - MFLIMIT + 1;
  const matchlimit = iend - LASTLITERALS;
  let anchor = 0;
  let op = 0;
  let ip = 0;

  if (inputSize >= LZ4_MIN_LENGTH) {
    put(0, hash(0));
    ip = 1;
    let forwardH = hash(ip);

    mainLoop: for (;;) {
      let match = 0;
      // Find a match.
      {
        let forwardIp = ip;
        let step = 1;
        let searchMatchNb = accel << LZ4_SKIP_TRIGGER;
        for (;;) {
          const h = forwardH;
          const current = forwardIp;
          const matchIndex = get(h);
          ip = forwardIp;
          forwardIp += step;
          step = searchMatchNb++ >> LZ4_SKIP_TRIGGER;
          if (forwardIp > mflimitPlusOne) break mainLoop;
          match = matchIndex;
          forwardH = hash(forwardIp);
          put(current, h);
          if (!small && matchIndex + LZ4_DISTANCE_MAX < current) continue; // too far
          if (read32(match) === read32(ip)) break;
        }
      }

      // Catch up.
      while (ip > anchor && match > 0 && src[ip - 1] === src[match - 1]) {
        ip--;
        match--;
      }

      // Encode literals.
      let token: number;
      {
        const litLength = ip - anchor;
        token = op++;
        if (limited && op + litLength + (2 + 1 + LASTLITERALS) + Math.floor(litLength / 255) > olimit) return 0;
        if (litLength >= RUN_MASK) {
          let len = litLength - RUN_MASK;
          dest[token] = RUN_MASK << 4;
          for (; len >= 255; len -= 255) dest[op++] = 255;
          dest[op++] = len;
        } else {
          dest[token] = litLength << 4;
        }
        dest.set(src.subarray(anchor, ip), op);
        op += litLength;
      }

      // _next_match (re-entered with a zero-literal token while the next position matches).
      for (;;) {
        const offset = ip - match;
        dest[op] = offset & 0xff;
        dest[op + 1] = offset >>> 8;
        op += 2;

        let matchCode = 0;
        {
          let a = ip + MINMATCH;
          let b = match + MINMATCH;
          while (a < matchlimit && src[a] === src[b]) {
            a++;
            b++;
          }
          matchCode = a - (ip + MINMATCH);
        }
        ip += matchCode + MINMATCH;
        if (limited && op + (1 + LASTLITERALS) + Math.floor((matchCode + 240) / 255) > olimit) return 0;
        if (matchCode >= ML_MASK) {
          dest[token] = dest[token]! + ML_MASK;
          matchCode -= ML_MASK;
          for (; matchCode >= 255; matchCode -= 255) dest[op++] = 255;
          dest[op++] = matchCode;
        } else {
          dest[token] = dest[token]! + matchCode;
        }

        anchor = ip;
        if (ip >= mflimitPlusOne) break mainLoop;

        // Fill table.
        put(ip - 2, hash(ip - 2));

        // Test next position.
        const h = hash(ip);
        const current = ip;
        const matchIndex = get(h);
        match = matchIndex;
        put(current, h);
        if ((small || matchIndex + LZ4_DISTANCE_MAX >= current) && read32(match) === read32(ip)) {
          token = op++;
          dest[token] = 0;
          continue;
        }
        break;
      }

      // Prepare next loop.
      forwardH = hash(++ip);
    }
  }

  // Last literals.
  const lastRun = iend - anchor;
  if (limited && op + lastRun + 1 + Math.floor((lastRun + 255 - RUN_MASK) / 255) > olimit) return 0;
  if (lastRun >= RUN_MASK) {
    let acc = lastRun - RUN_MASK;
    dest[op++] = RUN_MASK << 4;
    for (; acc >= 255; acc -= 255) dest[op++] = 255;
    dest[op++] = acc;
  } else {
    dest[op++] = lastRun << 4;
  }
  dest.set(src.subarray(anchor, iend), op);
  op += lastRun;
  return op;
}
