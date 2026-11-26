// K6.N: how good is the training target? On random scenes of the dataset's generator, the scattered fluence by his
// light-depth grid (next-event transmittance by quarter-voxel quadrature, interpolated between voxel centres), the same
// quadrature from the collision itself ('march') and unbiased ratio tracking: Σ grid / Σ ratio − 1 and Σ march / Σ ratio − 1,
// and the share of samples whose grid and ratio means differ by more than 3 joint standard errors. Our code (MIT).
// Run: node scripts/k6/neural/target-check.mjs [scenes=24] [points=40] [paths=512]
import { createRandom, trainingPhaseG } from './features.mjs';
import { makeSceneSetup } from './dataset.mjs';
import { createTraceScene, estimateScatteredFluence } from './trace.mjs';
import { mediumKinds } from './media.mjs';

const scenes = Number(process.argv[2] ?? 24);
const points = Number(process.argv[3] ?? 40);
const paths = Number(process.argv[4] ?? 512);
const byKind = mediumKinds.map(() => ({ grid: 0, ratio: 0, march: 0, n: 0, far: 0 }));

for (let s = 0; s < scenes; s++) {
  const sceneIndex = 5000 + s;
  const { random, kind, albedo, grid, lightDirection, data, points: picked } = makeSceneSetup(sceneIndex, 7);
  const scene = createTraceScene(grid, data, lightDirection, albedo);
  const r = grid.resolution;
  const rng = createRandom(sceneIndex);

  for (let i = 0; i < Math.min(points, picked.length); i++) {
    const index = picked[Math.floor(random() * picked.length)];
    const point = [(index % r) + 0.5, (Math.floor(index / r) % r) + 0.5, Math.floor(index / (r * r)) + 0.5];
    const a = estimateScatteredFluence(scene, point, paths, rng, trainingPhaseG, { nee: 'grid' });
    const b = estimateScatteredFluence(scene, point, paths, rng, trainingPhaseG, { nee: 'ratio' });
    const c = estimateScatteredFluence(scene, point, paths, rng, trainingPhaseG, { nee: 'march' });
    const row = byKind[kind];

    row.grid += a.mean;
    row.ratio += b.mean;
    row.march += c.mean;
    row.n++;
    row.far += Math.abs(a.mean - b.mean) > 3 * Math.sqrt(a.variance + b.variance) + 1e-9 ? 1 : 0;
  }
}

let total = { grid: 0, ratio: 0, march: 0, n: 0, far: 0 };

mediumKinds.forEach((name, k) => {
  const row = byKind[k];

  if (row.n) {
    console.log(`${name.padEnd(8)} n ${String(row.n).padStart(4)}  grid / ratio - 1 = ${(100 * (row.grid / row.ratio - 1)).toFixed(2).padStart(7)} %   march / ratio - 1 = ${(100 * (row.march / row.ratio - 1)).toFixed(2).padStart(7)} %   grid vs ratio > 3 sigma ${row.far}`);
    total = { grid: total.grid + row.grid, ratio: total.ratio + row.ratio, march: total.march + row.march, n: total.n + row.n, far: total.far + row.far };
  }
});
console.log(`all      n ${String(total.n).padStart(4)}  grid / ratio - 1 = ${(100 * (total.grid / total.ratio - 1)).toFixed(2).padStart(7)} %   march / ratio - 1 = ${(100 * (total.march / total.ratio - 1)).toFixed(2).padStart(7)} %   grid vs ratio > 3 sigma ${total.far} (${((100 * total.far) / total.n).toFixed(1)} %)`);
