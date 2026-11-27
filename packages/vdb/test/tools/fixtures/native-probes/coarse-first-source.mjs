// K4.17: the procedural coarse-first `.nvdb` that native NanoVDB v14 reads in `pnpm parity:native`
// and `native-probes.test.ts` rebuilds byte for byte. MIT (ours).
//
// `tools` is the `@pmndrs/vdb/tools` namespace (src in tests, dist in scripts), so both build the
// same bytes from the same code.

/** A 40³ fog blob (smooth radial falloff, value 0 outside radius 17) at index origin (-8, 4, -20). */
export function coarseFirstSourceImage(tools) {
  const n = 40;
  const values = new Float32Array(n * n * n);
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++)
      for (let z = 0; z < n; z++) {
        const r = Math.hypot(x - 19.5, (y - 19.5) * 1.25, z - 19.5) / 17;
        values[(x * n + y) * n + z] = r < 1 ? Math.fround((1 - r * r) * (0.75 + (((x * 7 + z * 3) % 11) / 40))) : 0;
      }
  return tools.vdbBuildFromDense(values, [n, n, n], { origin: [-8, 4, -20], voxelSize: 0.5, gridName: "density", gridClass: "FogVolume" });
}

/** The coarse-first file: previews at resolutions 8 and 16 (FpN), then the full Float grid. */
export function coarseFirstFile(tools, codec) {
  return tools.vdbBakeCoarseFirst(coarseFirstSourceImage(tools), { previewResolution: [8, 16], codec });
}

/** Index coordinates nvdb_probe reads in every grid (inside, edges, outside). */
export const COARSE_FIRST_PROBES = [
  [11, 23, 0],
  [12, 24, 0],
  [0, 10, -10],
  [-8, 4, -20],
  [31, 43, 19],
  [5, 30, -2],
  [2, 2, 2],
  [1, 6, -5],
  [3, 7, -1],
  [100, 0, 0],
];
