#!/usr/bin/env node
// K7.C1: `writeNvdb({ codec: 'blosc' })` against native NanoVDB v14 (OpenVDB v14.0.0 tools, built as
// packages/vdb/test/tools/fixtures/native-v14/README.md says; no Docker).
//
//   pnpm build && NANOVDB_BIN=<dir with nanovdb_convert, nanovdb_print, nanovdb_validate> \
//     node scripts/check-nvdb-blosc-native.mjs [out-dir]
//
// 1. Native writes: `nanovdb_convert -b` turns each `.vdb` (fixtures/parity/*.normalized.vdb, the
//    primitives' sources when present) into a BLOSC `.nvdb`; we read its grid images and write them again
//    with `codec: 'blosc'`. The files must be byte-equal except FileHeader/FileMetaData version (we write
//    our ABI version) and FileMetaData.nameKey (we write 0, as for NONE and ZIP).
// 2. Native reads: `nanovdb_validate -v` and `nanovdb_print -l` on files we write from our own images
//    (TS-built Float and Fp8 grids, Half through vdbQuantize), one grid and several per file.
// 3. Sizes: NONE vs ZIP (level 9) vs BLOSC for every grid.
// Exit code 1 on any mismatch or native failure.

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { NvdbFile, vdbQuantize, writeNvdb } from "../packages/vdb/dist/tools/index.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const BIN = process.env.NANOVDB_BIN;
if (!BIN || !existsSync(path.join(BIN, "nanovdb_convert"))) {
  console.error("✗ set NANOVDB_BIN to a directory holding the v14 nanovdb_convert, nanovdb_print and nanovdb_validate");
  process.exit(1);
}
const OUT = path.resolve(process.argv[2] ?? path.join(ROOT, "fixtures", "parity", "blosc"));
mkdirSync(OUT, { recursive: true });

let failed = false;
const fail = (msg) => {
  failed = true;
  console.log(`  ✗ ${msg}`);
};
const run = (tool, args) => spawnSync(path.join(BIN, tool), args, { encoding: "utf8" });
const KiB = (n) => `${(n / 1024).toFixed(1)} KiB`;
const ab = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const imagesOf = (file) => file.grids.map((_, i) => file.gridImage(i));

/** Byte offsets allowed to differ: FileHeader version (8..12); per grid FileMetaData nameKey (16..24), version (172..176). */
function allowedOffsets(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const allowed = new Set([8, 9, 10, 11]);
  let off = 16;
  for (let g = 0; g < dv.getUint16(12, true); g++) {
    for (let k = 16; k < 24; k++) allowed.add(off + k);
    for (let k = 172; k < 176; k++) allowed.add(off + k);
    off += 176 + dv.getUint32(off + 136, true);
  }
  return allowed;
}

// 1. Native writes, ours again.
const sources = [];
const parity = path.join(ROOT, "fixtures", "parity");
if (existsSync(parity)) for (const f of readdirSync(parity)) if (f.endsWith(".normalized.vdb")) sources.push(path.join(parity, f));
if (sources.length === 0) console.log("(no fixtures/parity/*.normalized.vdb: run `pnpm parity:native` once, or pass .vdb files)");
console.log("1. native BLOSC writes vs ours");
for (const src of sources) {
  const name = path.basename(src, ".normalized.vdb");
  const nativePath = path.join(OUT, `${name}.native.blosc.nvdb`);
  const conv = run("nanovdb_convert", ["-f", "-b", src, nativePath]);
  if (conv.status !== 0) {
    fail(`${name}: nanovdb_convert -b failed: ${conv.stderr}`);
    continue;
  }
  const native = readFileSync(nativePath);
  const images = imagesOf(NvdbFile.fromArrayBuffer(ab(native)));
  const ours = new Uint8Array(writeNvdb(images, { codec: "blosc" }));
  writeFileSync(path.join(OUT, `${name}.ours.blosc.nvdb`), ours);
  const allowed = allowedOffsets(ours);
  let diffs = 0;
  let first = -1;
  for (let k = 0; k < Math.max(ours.length, native.length); k++) {
    if (ours[k] !== native[k] && !allowed.has(k)) {
      diffs++;
      if (first < 0) first = k;
    }
  }
  const imageBytes = images.reduce((s, im) => s + im.byteLength, 0);
  if (ours.length !== native.length || diffs) fail(`${name}: ${diffs} bytes differ (first at ${first}); sizes ${ours.length} vs ${native.length}`);
  else console.log(`  ✓ ${name}: ${images.length} grid(s), image ${KiB(imageBytes)} -> ${KiB(ours.length)}, byte-equal to native (version, nameKey aside)`);
}

// 2. Native reads files we write from our own images.
console.log("2. native reads of our BLOSC files");
const ownSources = [];
for (const f of existsSync(parity) ? readdirSync(parity) : []) if (/\.typescript\.(float|fp8)\.nvdb$/.test(f)) ownSources.push(path.join(parity, f));
const prims = path.join(ROOT, "fixtures", "primitives");
for (const f of existsSync(prims) ? readdirSync(prims) : []) if (/_(float|fp8|fpn)\.nvdb$/.test(f)) ownSources.push(path.join(prims, f));
const sizes = [];
const all = [];
for (const src of ownSources) {
  const images = imagesOf(NvdbFile.fromArrayBuffer(ab(readFileSync(src))));
  const half = /float\.nvdb$/.test(src) ? [vdbQuantize(images[0], "half")] : [];
  for (const [label, imgs] of [[path.basename(src, ".nvdb"), images], ...(half.length ? [[`${path.basename(src, ".nvdb")}->half`, half]] : [])]) {
    const file = path.join(OUT, `${label}.ours.blosc.nvdb`);
    const bytes = new Uint8Array(writeNvdb(imgs, { codec: "blosc" }));
    writeFileSync(file, bytes);
    all.push(...imgs);
    const v = run("nanovdb_validate", ["-v", file]);
    const p = run("nanovdb_print", ["-l", file]);
    const ok = v.status === 0 && !/fail|error/i.test(v.stdout + v.stderr) && p.status === 0 && /BLOSC/.test(p.stdout);
    // v14's validator walks leaves with `n0++`, which can't step over FpN's variable-size leaves: it fails
    // native's own FpN grids the same way. Accept that one message when the source fails identically.
    const fpnQuirk = /Leaf node is not stored breadth-first/.test(v.stderr + v.stdout) &&
      /Leaf node is not stored breadth-first/.test((({ stdout, stderr }) => stdout + stderr)(run("nanovdb_validate", ["-v", src]))) &&
      p.status === 0 && /BLOSC/.test(p.stdout);
    if (fpnQuirk) console.log(`  ~ ${label}: nanovdb_print -l reads it; nanovdb_validate fails it as it fails native's own FpN grid (upstream validator)`);
    else if (!ok) fail(`${label}: native read failed\n${v.stdout}${v.stderr}${p.stdout}${p.stderr}`);
    else console.log(`  ✓ ${label}: nanovdb_validate -v and nanovdb_print -l read it (codec BLOSC)`);
    const none = writeNvdb(imgs).byteLength;
    const zip = writeNvdb(imgs, { codec: "zip" }).byteLength;
    sizes.push([label, none, zip, bytes.length]);
  }
}
if (all.length > 1) {
  const file = path.join(OUT, "multi.ours.blosc.nvdb");
  writeFileSync(file, new Uint8Array(writeNvdb(all.slice(0, 8), { codec: "blosc" })));
  const v = run("nanovdb_validate", ["-v", file]);
  const p = run("nanovdb_print", ["-l", file]);
  if (v.status !== 0 || p.status !== 0 || /fail|error/i.test(v.stdout + v.stderr)) fail(`multi-grid file: ${v.stdout}${v.stderr}`);
  else console.log(`  ✓ ${Math.min(8, all.length)} grids in one segment: native reads every grid\n${p.stdout.trim().split("\n").map((l) => `      ${l}`).join("\n")}`);
}

// 3. Sizes.
console.log("3. file sizes (NONE / ZIP level 9 / BLOSC)");
for (const [label, none, zip, blosc] of sizes) {
  console.log(`  ${label.padEnd(36)} ${KiB(none).padStart(12)} ${KiB(zip).padStart(12)} ${KiB(blosc).padStart(12)}  (BLOSC ${(blosc / zip).toFixed(2)}x ZIP)`);
}
process.exit(failed ? 1 : 0);
