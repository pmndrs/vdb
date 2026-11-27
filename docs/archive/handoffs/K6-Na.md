# K6.Na: 'neural' multiple scattering and the shared pmndrs neural tooling ([pmndrs/vdb#3](https://github.com/pmndrs/vdb/issues/3))

Branch `worktree-agent-a825e66794383219b`, from main `64f1ebc`; commits `56cd4e8` (code) and the docs commit after it. The sibling tooling was read at pmndrs/denoiser `3ad5e98`
(`/Users/dex/Developer/Denoiser`, read-only; MIT, Copyright (c) 2024 Dennis Smolek). Nothing in it was changed.

## 1. Fit study

### What we run

`multipleScattering: 'neural'` is a 20 → 32 → 32 → 1 MLP (ReLU, ReLU, linear, then `exp(min(o, 4))`), about 1.7k
multiply-adds, evaluated **inline in the transmittance kernel**: one thread per shadow-grid cell (64³ by default) and light
slot, inputs sampled from the cell's 6 box-mip levels in that thread, the output added to the slot's transmittance in that
thread. It runs only when the shadow grid redraws. The kernel is the core's own compute pass on the host's encoder, with one
storage buffer (the weights, D10); the shape is a uniform and the feature levels a 3D texture.

### What pmndrs/denoiser offers (read at `3ad5e98`)

| piece | what it is | fits our MLP? |
|---|---|---|
| `@pmndrs/denoiser-core` `NetworkRuntime` / `NetworkSession` / `NetworkBinding` (`src/runtime.ts`) | `load(model)` with `model = { name, channels: 3/6/9, precision }` (an OIDN model name), `bind({ batch, tileW, tileH })` (tiles of multiples of 16) gives NCHW `[B, C, H, W]` input and `[B, 3, H, W]` output **buffers**; `run(): Promise<void>` | No. The contract is an image network with 3 output channels; ours has 20 inputs per cell and 1 output, computed and consumed inside one kernel. |
| `denoiser-wgsl` `WgslRuntime` (`src/runtime.ts`, `graph.ts`, `conv.ts`, `mma.ts`) | the OIDN U-Net as ~16 fused **3 × 3 conv** dispatches; `buildGraph` needs OIDN's layer names (`enc_conv0` …) and throws on any kernel but 3 × 3; ReLU is `clamp(x, 0, 6)` (relu6); `run()` makes **its own command encoder and submits** it | No. No dense layer; a dense layer as a 3 × 3 conv would cost 9 × and still needs the U-Net names; relu6 changes our network (below); and it submits on its own instead of recording into the host's encoder (D34, D10 rules). |
| `denoiser-webnn` | the same U-Net as a WebNN graph (`navigator.ml`, Chrome flag only); IO through exported `MLTensor`s, its own submits | No: WebNN cannot run inside our compute pass, is behind a flag, and has its own context and queue. |
| `denoiser-core` `parseTZA` (`src/tza.ts`, 60 lines; Python twin `tools/onnx-convert/tza.py`) | the OIDN `.tza` v2 container: named tensors, any shape, f32 or f16, plus a layout string | **Yes.** It is generic. Our network maps onto it as PyTorch `nn.Linear` names (`layers.<i>.weight` [out, in], `layers.<i>.bias` [out]). |
| `denoiser-wgsl` `packWeights` / `packBias` (`src/conv.ts`) and its `mat4x4 * vec4` layer math | weights as a mat4x4 per (output group, input chunk, tap), activations as vec4 | **Yes, as a layout.** A dense layer is its one-tap case. The kernel around it (workgroup tiles, image addressing) does not apply. |
| training | none: `tools/onnx-convert` converts OIDN weights; nothing trains | Nothing to share. Our trainer (`scripts/k6/neural/`) stays. |
| packaging | `@pmndrs/denoiser-core` / `-wgsl` / `-webnn` are `private` (unpublished workspaces); npm has only `denoiser` 2.1.0, whose `denoiser/core` re-exports denoiser-core (the whole Denoiser facade and tiled engine) | A runtime dependency would bring a denoiser into `@pmndrs/vdb` for a 60-line parser. |

relu6 against our network (`scratchpad/neural-3/relu6.mjs`, 20 000 random feature vectors in [0, 4]; the input scaling is
folded into w1, so hidden values are not normalised): 21 % of hidden activations exceed 6 (max 74), and clamping them changes
98 % of the outputs. Our network needs plain ReLU, which no denoiser kernel has.

### The three options

| | (a) depend on `@pmndrs/denoiser-core` at runtime | (b) share the weight format and loader | (c) share WGSL layer code (vendor or generate) |
|---|---|---|---|
| what it would take | denoiser changes first: a non-image model descriptor, a dense op, plain ReLU, a `record(encoder)` that does not submit, and inputs that a host kernel writes; then our transmittance pass split in three (features to a buffer, the network, the combine) | nothing upstream; a `.tza` reader / writer for our network | nothing upstream for the layout; a shared generator would need denoiser to grow a dense-layer emitter that returns a WGSL **function** (per thread), not a dispatch |
| D20 (engine-free core, no new deps) | adds a dependency on unpublished workspaces (or on `denoiser` 2.1.0 whole); owner approval needed | none (vendored, MIT, credited) | none if vendored; a shared generator would be a dependency again |
| D10 / D34 (one storage buffer per kernel; record into the host encoder) | broken today (its `run()` submits; its kernels bind 4–6 storage buffers) | unaffected | kept: the packed weights stay the kernel's one storage buffer |
| memory | features materialised: 64³ cells × slots × 20 f32 ≈ 21 MB per slot, plus activations | none | none |
| what we gain | one runtime for every pmndrs network (WebNN later) | weights loadable by URL in the format the sibling's tools read; f16 storage; the denoiser's converter could turn ours into ONNX | the sibling's layer layout (vec4 / mat4x4 math, the shape its f16 and subgroup-matrix paths are built on) |
| risk | a large change for a 1.7k-MAC network run once per grid redraw | little | a second evaluator to keep correct (twin + mutation cover it) |

**Conclusion: the network does not run through denoiser's runtime as it is.** (a) needs five denoiser changes and breaks two
of our rules until they land. (b) fits now. (c) fits as a layout (vendored), not as shared code: there is no dense-layer code
to share. So this task stops at the study plus a minimal prototype of (b) and (c), behind an internal setting, with the
current path the default (the brief: "stop after the study and a minimal prototype, and say so").

**Recommendation:** keep the generic evaluator as the default; adopt (b) now (the `.tza` reader / writer, our weights
exportable); keep (c) behind `mlpRuntime: 'packed'` until K6.T can time it, because its only claim over the generic loop is
speed. Propose to pmndrs/denoiser the pieces that would make (a) possible later (§6). The test that settles (c): the K6.T
timing A/B of `mlpRuntime` generic vs packed (transmittance pass GPU time on a 64³ and a 128³ grid, Metal and one non-Apple
GPU); both already pass parity, so speed alone decides.

## 2. What was built

| piece | where | what |
|---|---|---|
| packed layout (option (c), vendored) | `tools/frame/mlp.ts`: `vdbFrameMlpPack`, `vdbFrameMlpPackedShape`, `vdbFrameMlpPackedOffsets`, `vdbFrameMlpPackedLayerVectors`, `vdbFrameMlpForwardPacked` (f64) | denoiser-wgsl's `packWeights` / `packBias` in the one-tap case: per layer a mat4x4 per (output group, input chunk), column j = input 4k + j, then the bias groups; padded lanes zero. Our network: 465 vec4 (7.4 KB). |
| packed evaluator | `core/frame/wgsl/lighting.ts` (`MLP_PACKED_*`), kernel key `'mlp-packed'` (`VDBFrameScatteringKernel`, `vdbFrameScatteringIsMlp`); `scattering.ts` uploads the packed weights and vec4 offsets; `kernels.ts`, `renderer.ts` take either key | `s += mat4x4f(4 vec4 columns) * a[k]` per chunk, vec4 activations, plain ReLU, `exp(min(o, 4))`. Same group 2 as the generic one (weights the one storage buffer, D10; shape uniform; feature mips). The feature code is shared text. The generic kernel's WGSL is byte-identical (size snapshot unchanged; one new entry, `transmittance (mlp-packed)` 15 581 B min). |
| the setting | `VDBFrameInternalSettings.mlpRuntime: 'generic' \| 'packed'`, default `'generic'` (`core/frame/settings.ts`) | internal (`core/internal.ts` type), so no public API change (D32 snapshots unchanged). Switching it re-uploads the weights and redraws the transmittance. |
| `.tza` (option (b), vendored) | `tools/frame/mlp-tza.ts`: `vdbTzaRead`, `vdbTzaWrite`, `vdbFrameMlpToTza`, `vdbFrameMlpFromTza` (tools internals) | denoiser-core's `parseTZA` format with a writer; f32 exact, f16 rounded to nearest (`vdbFloatToHalf`). `renderer.mlp = vdbFrameMlpFromTza(await (await fetch(url)).arrayBuffer())` loads a network by URL. `scripts/k6/neural/to-tza.mjs` writes ours (`pnpm build` first): 7 313 B in f32, 3 791 B in f16. |
| harness | `?mlpRuntime=packed` in k6-core, k3-frame, k6-scene; k6-core mutation `wgsl-mlp-packed` (each input chunk's lanes turned) | e2e: two new `K63_PROBES` rows (`e2e/k6-core.spec.ts`). |
| tests | `test/tools/frame-mlp-packed.test.ts` (7), `frame.test.ts` (the new spec in the WGSL name / size checks) | packing lossless vs the f64 generic forward on our net and odd widths (fail-before: a transposed pack fails it), transposed / shifted-bias mutations move the output, `.tza` round trips and errors, and **denoiser's own `parseTZA` reads our blob to the same tensors** (f32 and f16; skipped without a denoiser checkout). Also by hand: denoiser's Python `tools/onnx-convert/tza.py` lists our blob's 6 tensors with the right shapes. |

Not built: a runtime dependency on any denoiser package (option (a)); anything in the denoiser repo; any change to the
features, the weights or the trainer. The default path is unchanged.

## 3. Evidence: the kernels against their twin (SwiftShader)

`WEBVDB_E2E_PORT=5348 npx playwright test e2e/k6-core.spec.ts -g "mlp|neural|compile"`: 8 passed (both compile probes
include the packed variant). The packed rows match the same f64 twin as the generic ones, to the same counts:

| probe | generic transmittance / frame | packed transmittance / frame | mutations (values over the bar) |
|---|---|---|---|
| `?probe=frame&ms=mlp` | max 0.50 / 0.65 f16 ulp, 0 over | max 0.50 / 0.65 f16 ulp, 0 over | packed: `mlp-weights` 2614 / 16384, `wgsl-mlp-packed` 2599 / 16384 |
| `?probe=frame&ms=neural&source=sparse&groups=2` | max 0.50 / 1.80, 0 over | max 0.50 / 1.80, 0 over | packed: `wgsl-mlp-packed` 1752 / 16384 |

The WGSL mutation patches only the packed kernel's text (the run fails if it matches no module), so the packed kernel is
what ran.

## 4. Evidence: frames (`e2e/k6-frames.spec.ts`, SwiftShader)

Before set from `git archive 64f1ebc` (main), its own install and port (5349), warmed with one frame: 98 PNGs.

- This branch, defaults: **98 / 98 identical** (max 0/255).
- This branch, `K6_FRAMES_QUERY=mlpRuntime=packed` (every page runs the packed evaluator): 97 / 98 identical; the one other
  is a 'neural' frame, `step6-cast-shadow-two-bunnies-dir-isolated`: max 1/255, 0.0000 % of pixels over 1/255. Every
  non-'neural' frame is identical, and so are the other 15 'neural' frames.

## 5. Evidence: bias against the path-traced reference (hardware)

`scripts/k6/reference-report.mjs` from a copy of this branch (`scratchpad/neural-3/refrun`, own install, port 5350), installed
Chrome headless (adapter `apple metal-3`, the owner's Apple M5 Pro; loaded machine, no timings taken), 1280 x 720 frames 8x down
to 160 x 90, ambientOcclusion on, the K6.N groups. No cached reference from earlier runs matched this tree's queries, so the
first run rendered all 32 references fresh at the K6.N settings (`--target=0.04 --budget=200`) and the second reused them, so
both evaluators are judged against the same references:

```
G=frame,mie,mie-rgb,env,wdas,wdas-ambient,smoke,smoke-ambient,bunny-absorbing,bunny-absorbing-ambient,bunny-thin,bunny-thin-ambient,bunny-point,smoke-hemi,wdas-probe,bunny-absorbing-hemi
node scripts/k6/reference-report.mjs --headless --port=5350 --groups=$G --rows='two-sided|neural' --query=ao=1 --reuse --target=0.04 --budget=200
node scripts/k6/reference-report.mjs --headless --port=5350 --groups=$G --rows='neural' --query='ao=1&mlpRuntime=packed' --reuse --target=0.04 --budget=200
```

Cells: luminance bias / median relative error (%), as K6-N.md section 5.

| group | K6.N ours (bias / median) | two-sided now | neural generic now | neural packed now |
|---|---|---|---|---|
| frame | -5.9 / 24.6 | +1.9 / 28.8 | -5.9 / 24.6 | -5.9 / 24.6 |
| mie | +3.0 / 23.9 | -2.2 / 31.9 | +3.0 / 23.9 | +3.0 / 23.9 |
| mie-rgb | +3.4 / 23.6 | -1.8 / 32.1 | +3.3 / 23.7 | +3.3 / 23.7 |
| env | +14.2 / 60.9 | +14.2 / 60.9 | +14.2 / 60.9 | +14.2 / 60.9 |
| wdas | -31.5 / 33.0 | -32.0 / 31.8 | -31.5 / 33.0 | -31.5 / 33.0 |
| wdas-ambient | -25.4 / 27.5 | -25.8 / 25.7 | -25.4 / 27.5 | -25.4 / 27.5 |
| smoke | -2.0 / 18.7 | +29.8 / 27.2 | -2.0 / 18.4 | -2.0 / 18.4 |
| smoke-ambient | -1.2 / 12.9 | +21.6 / 18.0 | -1.2 / 12.9 | -1.2 / 12.9 |
| bunny-absorbing | -5.2 / 12.6 | -30.7 / 32.7 | -5.2 / 12.6 | -5.2 / 12.6 |
| bunny-absorbing-ambient | -2.5 / 11.5 | -23.3 / 25.2 | -2.5 / 11.5 | -2.5 / 11.5 |
| bunny-thin | +9.1 / 11.0 | +8.1 / 9.3 | +9.1 / 11.0 | +9.1 / 11.0 |
| bunny-thin-ambient | +5.4 / 7.0 | +4.8 / 6.1 | +5.4 / 7.0 | +5.4 / 7.0 |
| bunny-point | +23.6 / 32.7 | +30.2 / 28.2 | +23.6 / 32.7 | +23.6 / 32.7 |
| smoke-hemi | -1.2 / 15.4 | +25.2 / 21.4 | -1.2 / 15.4 | -1.2 / 15.4 |
| wdas-probe | -23.3 / 25.9 | -23.7 / 23.9 | -23.3 / 25.9 | -23.3 / 25.9 |
| bunny-absorbing-hemi | -3.2 / 11.6 | -25.9 / 27.4 | -3.2 / 11.6 | -3.2 / 11.6 |

mean |bias| over 16 groups: K6.N ours 10.0, two-sided 18.8, generic 10.0, packed 10.0

- **Packed = generic.** Against the same references the two evaluators agree to the printed digit in all 16 groups. 11 groups
  are bit-identical in every metric; 5 (mie, mie-rgb and the three absorbing-bunny groups) differ by at most 4.4e-7 in the
  bias fraction (f32 summation order), which also shows the packed kernel did run on Metal.
- **No worse than K6.N.** The 'neural' mean |bias| is 10.0 % (K6.N: 10.0 %; two-sided 18.8 %). Two cells moved by 0.1 to 0.3
  points against K6-N.md (mie-rgb +3.4 to +3.3, smoke median 18.7 to 18.4): the references are new renders, so that is
  reference noise, and it is the same for both evaluators.

## 5a. Questions for the owner

1. **Keep `mlpRuntime: 'packed'` or delete it?** Suggestion: keep it internal until K6.T times it, then delete the loser.
   Pro: both pass the same parity and frame gates, so speed alone decides, and the packed layout is the shape denoiser's f16 and
   subgroup-matrix paths build on. Con: two evaluators to keep in step (the twin and the `wgsl-mlp-packed` mutation cover the
   packed one). Settles it: the K6.T A/B (transmittance-pass GPU time, 64³ and 128³, Metal plus one non-Apple GPU).
2. **Make the `.tza` helpers public** (`vdbFrameMlpFromTza` / `ToTza` in `@pmndrs/vdb/tools`) so hosts can load a network by
   URL? Suggestion: yes once a second network exists (an anisotropy or resolution-independent retrain, K6.Nb). Pro: weights as
   data in the sibling's format, no TS module needed. Con: a public format commitment (the convention in V181 has no
   metadata for activations / output). Settles it: a host or the pathtracer asking for it; or denoiser agreeing a metadata
   convention (§6 item 3).
3. **Depend on denoiser later (option (a))?** Suggestion: no for 0.x. Pro: one shared runtime. Con: five denoiser changes (§6)
   and an unpublished dependency for a 1.7k-MAC network. Settles it: denoiser shipping a published dense-layer function emitter
   and a `record(encoder)` API; then re-run this study.
4. **Ship f16 weights?** Not evaluated: a `.tza` with `type: 'h'` rounds every weight (a different network). If wanted, it is
   one reference-report run (`renderer.mlp = vdbFrameMlpFromTza(f16 blob)` behind a harness option).

## 6. Proposals for pmndrs/denoiser (not made: read-only)

What would let `@pmndrs/vdb` run its network *through* the shared tooling instead of beside it (option (a)), in order of
value. None blocks us; each is a denoiser issue the owner can file.

1. **A dense-layer WGSL emitter that returns a function, not a dispatch.** E.g. `denseWgsl({ prefix, layers, activation:
   'relu' | 'relu6' | 'linear', weights: { binding, type: 'vec4' } })` returning `fn <prefix>Forward(x: array<vec4f, N>) ->
   array<vec4f, M>` over the `packWeights` layout. Hosts (our transmittance kernel, a denoiser kernel-prediction head) inline
   it; it reads one storage buffer. Our `MLP_PACKED_FORWARD_WGSL` (`core/frame/wgsl/lighting.ts`) is a working draft of it.
2. **`packDense(w, outputs, inputs)` exported** beside `packWeights` (the one-tap case, no 9-tap stride): our
   `vdbFrameMlpPack` is that function.
3. **A non-OIDN model convention for `.tza`**: tensor names for dense layers (`layers.<i>.weight` / `.bias`, as we write) and
   a way to carry activations and an output transform (a small `meta` tensor or a JSON sidecar), so a `.tza` is
   self-describing for networks that are not the U-Net.
4. **`NetworkBinding.record(encoder)`** beside `run()`: encode into the caller's encoder without submitting (the WGSL runtime
   already encodes into one encoder; only the submit is in the way). This is what any library that shares the host's frame
   needs (D34's rule here).
5. **Plain ReLU** in the conv epilogue as an option (`relu6` today): networks trained with ReLU cannot run on it.
6. **Training**: denoiser has no trainer. If pmndrs wants one shared trainer, our `scripts/k6/neural/train.mjs` (dependency
   free, Adam, CPU, ~400 lines, MIT, derived from mjurczyk's) is a candidate to move there with a `.tza` writer.

## 7. Grid-resolution dependence (K6.Q1b finding, not addressed here)

K6.Q1b (main `136a298`) found that 'neural' moves with the shadow grid's side: the features are `log(1 + σ h 2^k)` with h the
grid voxel, so a finer grid feeds the network shorter reaches and smaller optical depths per level (absorbing bunny −5.3 % →
+8.2 % at the adaptive side, smoke −2 % → −10 % at side 128). Neither prototype here touches the features: `mlpRuntime`
changes only how the same network is evaluated, and the `.tza` work only how its weights are stored, so both inherit the
dependence unchanged. Two ways to remove it, for the next K6.Na step (neither needs the sibling tooling):

1. **Features in world units.** Fix the reaches to `h₀ 2^k` with h₀ a world length tied to the volume (e.g. the box side / 64),
   and sample the mips at the level whose texel matches the reach (trilinear between levels). The network then sees the same
   inputs at any grid side; the 64³ case is unchanged, so our weights stay valid there. Cost: the feature code in
   `lighting.ts` / `mlp.ts` and its twin; a retrain is not needed if h₀ equals today's h at 64³.
2. **Train across sides.** Add the grid side (or h / box side) as an input and draw it in `dataset.mjs` (32–256). Needs a
   new input (the core's 3 L + 2 check) and a retrain.

The test that settles it: the K6.Q1b reference groups (absorbing bunny, smoke) at sides 64, 128 and the adaptive side; a
resolution-independent network keeps its bias within the reference noise (≈ 4 %) across the three.
