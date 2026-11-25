/**
 * Resources of the core frame renderer (K6.1, design §4.1–§4.2): the retire queue and the froxel texture
 * allocation. The renderer owns every texture and buffer it makes; a replaced one is retired and destroyed
 * two updates later, in a later task (K6.10a, see {@link VDBRetireQueue}). `dispose()` drains the queue.
 */

/** Something with a `destroy()` (a GPUTexture, GPUBuffer or GPUQuerySet). */
export interface VDBDestroyable {
  destroy(): void;
}

/**
 * Destroys retired objects two updates after they were retired, and never in the task that made them due (K6.10a).
 *
 * WebGPU accepts `destroy()` on an object that a SUBMITTED command buffer uses (the free waits for the GPU), but a
 * submit that names an object destroyed before it fails. The queue cannot see the host's submit, so it waits for two
 * things, either of which makes the destroy safe:
 * - two updates (the K6.1 contract: one update per submit, so the encoders that used the object were submitted), and
 * - a task boundary after the update that made it due (`setTimeout(0)`): a host that records several updates (or
 *   pass records) into one encoder and submits it in the same task has submitted it by then.
 *
 * Before K6.10a the destroy ran inside the update, so three updates recorded into one encoder before one submit
 * destroyed a texture the first update's passes still used. The remaining unsafe pattern is three or more updates
 * recorded without a submit AND across tasks (an `await` between record and submit); the renderer's docs say to submit
 * each update's encoder in the task that recorded it.
 */
export class VDBRetireQueue {
  #items: { tick: number; object: VDBDestroyable }[] = [];
  /** Due objects whose destroy waits for the next task (removed when destroyed, by the timer or by `drain`). */
  readonly #scheduled = new Set<VDBDestroyable>();
  #tick = 0;

  /** Objects waiting to be destroyed (retired, or due and waiting for the next task). */
  get pending(): number {
    return this.#items.length + this.#scheduled.size;
  }

  /** Keeps `object` until two more {@link advance} calls, then destroys it in a later task. */
  retire(object: VDBDestroyable | null | undefined): void {
    if (object) this.#items.push({ tick: this.#tick, object });
  }

  /**
   * Destroys `object` in a later task, without waiting for updates: for an owner that has no update of its own (a
   * source replaced between the host's record and submit, K6.10a). Safe when the host submits in the task that
   * recorded; the owner's users rebind at their next update.
   */
  retireAfterTask(object: VDBDestroyable | null | undefined): void {
    if (object) this.#schedule([object]);
  }

  /** One update: what was retired two or more updates ago is destroyed in a later task. */
  advance(): void {
    this.#tick++;
    const keep: { tick: number; object: VDBDestroyable }[] = [];
    const due: VDBDestroyable[] = [];
    for (const item of this.#items) {
      if (this.#tick - item.tick >= 2) due.push(item.object);
      else keep.push(item);
    }
    this.#items = keep;
    if (due.length > 0) this.#schedule(due);
  }

  #schedule(objects: VDBDestroyable[]): void {
    for (const o of objects) this.#scheduled.add(o);
    setTimeout(() => {
      for (const o of objects) if (this.#scheduled.delete(o)) o.destroy();
    }, 0);
  }

  /** Destroys everything now (`dispose()`: the host has no more work naming these objects). */
  drain(): void {
    for (const item of this.#items) item.object.destroy();
    this.#items = [];
    for (const o of this.#scheduled) o.destroy();
    this.#scheduled.clear();
  }
}

/** `GPUTextureUsage` values, restated (the globals do not exist in Node). */
export const TEXTURE_USAGE = { COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04, STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10 } as const;
/** `GPUBufferUsage` values. */
export const BUFFER_USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;
/** `GPUShaderStage` values. */
export const STAGE = { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 } as const;

/** A froxel-sized 3D storage texture (write by compute, read by sampling or `textureLoad`, copy for readbacks). */
export function createFroxelTexture(device: GPUDevice, label: string, size: readonly [number, number, number], format: GPUTextureFormat, sampled = true): GPUTexture {
  return device.createTexture({
    label,
    size: [size[0], size[1], size[2]],
    dimension: "3d",
    format,
    usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.COPY_SRC | (sampled ? TEXTURE_USAGE.TEXTURE_BINDING : 0),
  });
}
