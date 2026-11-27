/**
 * P1.1: the point-sample contract between @pmndrs/vdb and pmndrs/pathtracer.
 *
 * The pathtracer samples a dense grid with voxel centres at integer index coordinates and zero outside
 * `[0, n - 1]` (`gridDensity`). Our `vdbSample` with `outside: 'zero'` on a `vdbBuildFromDense` image of the
 * same array (transposed: the pathtracer's arrays are x fastest, `vdbBuildFromDense` reads x-major) must equal
 * it: same centres, same zero outside. For a fog grid with inactive 0 (a dense build with background 0) the
 * index bbox `[min, max]` of the image lies inside `[0, n - 1]` and every voxel outside it is 0 in the dense
 * array, so zero outside the bbox is zero outside the grid. (A native grid whose inactive values are not 0
 * would differ: the native-background trap.) GPU parity of the same points: e2e/p1-parity.spec.ts.
 */
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { parseVdb, vdbBuildFromDense, vdbSampleTrilinearCpu } from "../../src/tools/index.js";
import { indexBBox, vdbReadValueF32Cpu } from "../../src/tools/internal.js";
import { contractPoints, denseTrilinear, smokeDense, toXMajor } from "../../../../examples/src/harness/p1-smoke.js";

const SMOKE = fileURLToPath(new URL("../../../../fixtures/vdb-samples/smoke.vdb", import.meta.url));
const present = existsSync(SMOKE);
/** The pathtracer's generated `vdb-smoke.f32` (git-ignored there), when this machine has it. */
const PT_F32 = process.env.VDB_SMOKE_F32 ?? join(homedir(), "Developer", "pathtracer", "bench", "assets", "generated", "volumes", "vdb-smoke.f32");


describe.skipIf(!present)("P1.1 point-sample contract (vdb-smoke)", () => {
  const buf = present ? readFileSync(SMOKE) : Buffer.alloc(0);
  const dense = present ? smokeDense(parseVdb(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))) : null;

  it("the rebuilt dense grid is the pathtracer's vdb-smoke (60 x 115 x 60, peak 5.38671875)", () => {
    expect(dense!.resolution).toEqual([60, 115, 60]);
    expect(dense!.peak).toBe(5.38671875);
  });

  it.skipIf(!existsSync(PT_F32))("equals its generated vdb-smoke.f32 bit for bit", () => {
    const b = readFileSync(PT_F32);
    const theirs = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    expect(theirs.length).toBe(dense!.density.length);
    let diff = 0;
    for (let i = 0; i < theirs.length; i++) if (!Object.is(theirs[i], dense!.density[i])) diff++;
    expect(diff).toBe(0);
  });

  it("vdbSample (outside 'zero') on vdbBuildFromDense equals its dense trilinear at 10k index points (≤ 1e-6)", () => {
    const { resolution: res, density } = dense!;
    const image = vdbBuildFromDense(toXMajor(density, res), res);
    const { min, max } = indexBBox(image);
    expect(min.every((v) => v >= 0) && max.every((v, a) => v <= res[a]! - 1)).toBe(true);
    let worst = 0;
    let nonzero = 0;
    for (const p of contractPoints(res)) {
      const ours = vdbSampleTrilinearCpu(image, p, { outside: "zero" });
      const theirs = denseTrilinear(density, res, p);
      worst = Math.max(worst, Math.abs(ours - theirs));
      if (theirs > 0) nonzero++;
    }
    expect(worst).toBeLessThanOrEqual(1e-6);
    expect(nonzero).toBeGreaterThan(2000); // the points reach the smoke, not just empty space
  });

  it("P1.8 (our side): every voxel centre of the image is the dense value, bit for bit", () => {
    const { resolution: res, density } = dense!;
    const image = vdbBuildFromDense(toXMajor(density, res), res);
    let diff = 0;
    for (let k = 0; k < res[2]; k++)
      for (let j = 0; j < res[1]; j++)
        for (let i = 0; i < res[0]; i++) if (!Object.is(vdbReadValueF32Cpu(image, [i, j, k]), density[i + res[0] * (j + res[1] * k)]!)) diff++;
    expect(diff).toBe(0);
  });

  it("fails before: half-voxel-shifted centres or an untransposed array are caught", () => {
    const { resolution: res, density } = dense!;
    const image = vdbBuildFromDense(toXMajor(density, res), res);
    const untransposed = vdbBuildFromDense(density, [res[2], res[1], res[0]]);
    let shifted = 0;
    let swapped = 0;
    for (const p of contractPoints(res, 2000, 3)) {
      const theirs = denseTrilinear(density, res, p);
      shifted = Math.max(shifted, Math.abs(vdbSampleTrilinearCpu(image, [p[0] - 0.5, p[1] - 0.5, p[2] - 0.5], { outside: "zero" }) - theirs));
      swapped = Math.max(swapped, Math.abs(vdbSampleTrilinearCpu(untransposed, p, { outside: "zero" }) - theirs));
    }
    expect(shifted).toBeGreaterThan(1e-2);
    expect(swapped).toBeGreaterThan(1e-2);
  });
});
