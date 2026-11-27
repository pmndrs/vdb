// K6.Na (pmndrs/vdb#3): writes the default 'neural' network as a `.tza` v2 blob, the weight container the pmndrs neural
// tooling reads (pmndrs/denoiser `parseTZA`, its `tools/onnx-convert/tza.py`): `layers.<i>.weight` [outputs, inputs] and
// `layers.<i>.bias` [outputs], hidden layers ReLU, the last linear, output exp(min(o, 4)), levels (inputs − 2) / 3.
//
//   pnpm build && node scripts/k6/neural/to-tza.mjs [out.tza] [--type=f|h]
//
// Reads packages/vdb/dist (run `pnpm build` first: a stale dist writes old weights). Our data (MIT).
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { vdbFrameMlpFromNeural, vdbFrameMlpFromTza, vdbFrameMlpToTza, vdbFrameNeuralWeights } from '../../../packages/vdb/dist/tools/internal.js';

const positional = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const type = process.argv.find((a) => a.startsWith('--type='))?.split('=')[1] ?? 'f';
if (type !== 'f' && type !== 'h') throw new Error(`--type must be f or h (got ${type})`);
const out = path.resolve(positional[0] ?? fileURLToPath(new URL(`../../../artifacts/k6-neural/neural-weights${type === 'h' ? '-f16' : ''}.tza`, import.meta.url)));

const mlp = vdbFrameMlpFromNeural(vdbFrameNeuralWeights);
const blob = vdbFrameMlpToTza(mlp, { type });
const back = vdbFrameMlpFromTza(blob);
const exact = back.weights.every((w, i) => w === mlp.weights[i]);
if (type === 'f' && !exact) throw new Error('the f32 blob does not read back to the same weights');
mkdirSync(path.dirname(out), { recursive: true });
writeFileSync(out, blob);
console.log(`${out}: ${blob.byteLength} bytes, ${mlp.layers.map((l) => `${l.inputs}→${l.outputs}`).join(', ')}, ${type === 'f' ? 'f32 (exact)' : 'f16 (rounded)'}`);
