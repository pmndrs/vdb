/**
 * K6.10a / K6.10b fail-before mutations on the GPU side: `?mutate=wgsl-*` replaces a text in every WGSL module the
 * device compiles (a `createShaderModule` wrapper), so a probe proves the KERNEL computes the term, not only that its
 * comparator would see a twin change. A patch that matches no module is an error: a stale patch never passes silently.
 * (k6-core.ts keeps its own copy, `WGSL_PATCHES`, from K6.10a.)
 */

/** A mutation: the WGSL text to find and its replacement. */
export type VDBWgslPatches = Readonly<Record<string, readonly [string, string]>>;

/**
 * Installs `patches[mutate]` on `device` when `mutate` names one; returns the check that restores `createShaderModule`
 * and throws when the patch matched no module. Throws at once for an unknown `wgsl-*` name.
 */
export function installWgslPatch(device: GPUDevice, patches: VDBWgslPatches, mutate: string): () => void {
  const patch = patches[mutate];
  if (!patch) {
    if (mutate.startsWith("wgsl-")) throw new Error(`unknown WGSL mutation ${mutate}`);
    return () => {};
  }
  const d = device as unknown as { createShaderModule: (desc: GPUShaderModuleDescriptor) => GPUShaderModule };
  const saved = d.createShaderModule;
  let matched = 0;
  d.createShaderModule = (desc) => {
    if (!desc.code.includes(patch[0])) return saved.call(device, desc);
    matched++;
    return saved.call(device, { ...desc, code: desc.code.replaceAll(patch[0], patch[1]) });
  };
  return () => {
    d.createShaderModule = saved;
    if (matched === 0) throw new Error(`WGSL mutation ${mutate} matched no shader module`);
  };
}
