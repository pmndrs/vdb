// K6.N: compares the 'neural' networks on the held-out scenes of a dataset (scene index % 10 = 0, or any scenes a
// dataset from another seed holds): ours, his (copied unchanged) and the diffusion / octave modes, against the
// path-traced scattered light. Our code (MIT).
// Run: node scripts/k6/neural/eval.mjs [dataset] [--weights=<ts file>] [--all]   (--all: every scene, e.g. a fresh seed)
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createNetwork, neuralHidden, neuralInputs, runNetwork } from './features.mjs';
import { mediumKinds } from './media.mjs';

const positional = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const option = (name, fallback) => process.argv.find((a) => a.startsWith(`--${name}=`))?.split('=')[1] ?? fallback;
const root = fileURLToPath(new URL('../../../', import.meta.url));
const datasetPath = positional[0] ?? `${root}artifacts/k6-neural/dataset.bin`;
const oursPath = option('weights', `${root}packages/vdb/src/tools/frame/neural-weights-procedural.ts`);
const hisPath = `${root}packages/vdb/src/tools/frame/neural-weights.ts`;
const all = process.argv.includes('--all');
const floats = neuralInputs + 7;

// The numbers of the `new Float32Array([...])` of a weights module, in his vec4 layout (scaling folded in).
const readWeights = (path) => {
  const text = readFileSync(path, 'utf8');
  const body = text.slice(text.indexOf('new Float32Array(') + 17);
  const numbers = body.slice(body.indexOf('[') + 1, body.indexOf(']')).split(',').map((v) => v.trim()).filter(Boolean).map(Number);
  const network = createNetwork();
  let offset = 0;

  for (const key of ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']) {
    network[key].set(numbers.slice(offset, offset + network[key].length));
    offset += network[key].length;
  }

  return network;
};

const buffer = readFileSync(datasetPath);
const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
const count = data.length / floats;
const networks = { ours: readWeights(oursPath), his: readWeights(hisPath) };
const rows = [...mediumKinds, 'all'].map(() => ({ count: 0, target: 0, noise: 0, err: {}, bias: {}, rel: [] }));
const modes = ['ours', 'his', 'diffusion', 'octaves'];
const inputs = new Float32Array(neuralInputs);
const cache = { h1: new Float32Array(neuralHidden), h2: new Float32Array(neuralHidden) };

for (let i = 0; i < count; i++) {
  const offset = i * floats;

  if (!all && data[offset + neuralInputs + 5] % 10 !== 0) {
    continue;
  }

  inputs.set(data.subarray(offset, offset + neuralInputs));

  const target = data[offset + neuralInputs];
  const values = {
    ours: Math.exp(Math.min(runNetwork(networks.ours, inputs, cache), 4)),
    his: Math.exp(Math.min(runNetwork(networks.his, inputs, cache), 4)),
    diffusion: data[offset + neuralInputs + 2],
    octaves: data[offset + neuralInputs + 3],
  };

  for (const row of [rows[data[offset + neuralInputs + 6]], rows[rows.length - 1]]) {
    row.count++;
    row.target += target;
    row.noise += Math.sqrt(data[offset + neuralInputs + 1]) * Math.sqrt(2 / Math.PI);

    for (const mode of modes) {
      row.err[mode] = (row.err[mode] ?? 0) + Math.abs(values[mode] - target);
      row.bias[mode] = (row.bias[mode] ?? 0) + values[mode] - target;
    }
  }
}

const cell = (row, mode) => `${((100 * row.err[mode]) / row.target).toFixed(0).padStart(4)}% ${((100 * row.bias[mode]) / row.target).toFixed(0).padStart(5)}%`;

console.log(`${all ? 'all scenes' : 'held-out scenes (index % 10 = 0)'} of ${datasetPath}: relative mean abs error, relative bias`);
[...mediumKinds, 'all'].forEach((name, k) => {
  const row = rows[k];

  if (row.count) {
    console.log(
      `${name.padEnd(8)} ${String(row.count).padStart(6)}  ${modes.map((m) => `${m} ${cell(row, m)}`).join('   ')}   noise ${((100 * row.noise) / row.target).toFixed(0)}%`,
    );
  }
});
