/**
 * K3.2c / K6.8: GPU times of the froxel renderer. The core update's kernels are bracketed by timestamp writes
 * (`VDBFrameRenderer.passHook.timestampWrites`: a `'timestamp'` query set, written before the first and after the last
 * kernel the update records); three's own render passes (`WebGPURenderer({ trackTimestamp: true })`) give the rest of
 * the frame (the scene and the volume composite). Both need the `timestamp-query` feature; D25: timings are provisional.
 */
import type { WebGPURenderer } from "three/webgpu";
import type { VDBFrameRenderer } from "@pmndrs/vdb/three";

export interface PassTimes {
  /** GPU ms of the last core update's kernels (null: no update since the last collect). */
  core: number | null;
  /** Core updates recorded since the last collect (`passHook.begin` / `end`). */
  updates: number;
  /** GPU ms of three's render passes since the last collect (0 without `trackTimestamp`). */
  render: number;
}

export interface PassTimer {
  collect(): Promise<PassTimes>;
  /** Removes the hook and frees the query set. */
  dispose(): void;
}

/** Installs the timestamp bracket on `frameRenderer` (the device needs `timestamp-query`). */
export function installPassTimer(renderer: WebGPURenderer, frameRenderer: VDBFrameRenderer, device: GPUDevice): PassTimer {
  const querySet = device.createQuerySet({ type: "timestamp", count: 2, label: "k3 pass timer" });
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC, label: "k3 pass timer resolve" });
  const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label: "k3 pass timer read" });
  let begun = 0;
  let ended = 0;
  const previous = frameRenderer.passHook;
  frameRenderer.passHook = {
    timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    begin: () => {
      begun++;
    },
    end: () => {
      ended++;
    },
  };
  return {
    async collect() {
      const updates = ended;
      let core: number | null = null;
      if (updates > 0) {
        const encoder = device.createCommandEncoder({ label: "k3 pass timer" });
        encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, read, 0, 16);
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const t = new BigUint64Array(read.getMappedRange().slice(0));
        read.unmap();
        core = Number(t[1]! - t[0]!) / 1e6;
      }
      begun = ended = 0;
      const ms = await renderer.resolveTimestampsAsync("render").catch(() => 0);
      return { core, updates, render: typeof ms === "number" ? ms : 0 };
    },
    dispose() {
      frameRenderer.passHook = previous;
      querySet.destroy();
      resolve.destroy();
      read.destroy();
      void begun;
    },
  };
}
