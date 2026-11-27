/**
 * Demo 14's decode worker: downloads an OpenVDB point data grid, parses it with our parser (`parseVdb`) and turns it
 * into render data with `vdbPointRenderData` (prefix order, `Cd` and `v`, progress), so a 157 MB file never blocks
 * the page. The arrays go back as a transfer list (`vdbPointRenderDataTransfer`). Public API only.
 *
 * The points come back in prefix order (`order: 'prefix'`): every 10th point first, then the rest of the even
 * points, then the odd ones. A prefix of 10% (or 50%) of the arrays is then an even subsample of the whole cloud, and
 * the page changes the density by changing `VDBPoints.pointCount`, without touching the buffers.
 */
import { parseVdb, vdbPointRenderData, vdbPointRenderDataTransfer, type VDBPointRenderData, type VdbPointDataGrid } from "@pmndrs/vdb/tools";

export interface DecodeRequest {
  url: string;
}

export type DecodeMessage =
  | { type: "progress"; stage: "download"; received: number; total: number }
  | { type: "progress"; stage: "parse" }
  | { type: "progress"; stage: "decode" | "copy"; done: number; total: number }
  | { type: "done"; result: DecodedPoints }
  | { type: "error"; message: string };

export interface DecodedPoints {
  /** `vdbPointRenderData(grid, { order: 'prefix', attributes: ['v'] })`: positions, Cd, v, prefix counts, bounds, spacing. */
  data: VDBPointRenderData;
  /** `|v|` in the same order, or null. */
  speed: Float32Array | null;
  /** 2nd and 98th percentile of `|v|` (colour ramp range). */
  speedRange: [number, number];
  /** Attribute names the file has, and the ones the decoder skipped. */
  attributes: string[];
  skipped: string[];
  fileBytes: number;
  activeVoxels: number;
  ms: { download: number; parse: number; decode: number };
}

const post = (m: DecodeMessage, transfer: Transferable[] = []): void => (self as unknown as Worker).postMessage(m, transfer);

self.onmessage = async (e: MessageEvent<DecodeRequest>) => {
  try {
    const t0 = performance.now();
    const r = await fetch(e.data.url);
    if (!r.ok || !r.body) throw new Error(`fetch ${e.data.url}: ${r.status}`);
    const total = Number(r.headers.get("content-length") ?? 0);
    const reader = r.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    let lastPost = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      const now = performance.now();
      if (now - lastPost > 50) {
        lastPost = now;
        post({ type: "progress", stage: "download", received, total });
      }
    }
    const bytes = new Uint8Array(received);
    let o = 0;
    for (const c of chunks) {
      bytes.set(c, o);
      o += c.byteLength;
    }
    chunks.length = 0;
    const t1 = performance.now();

    post({ type: "progress", stage: "parse" });
    const file = parseVdb(bytes.buffer);
    const grid = file.grids.find((g): g is VdbPointDataGrid => g.valueType === "ptdataidx32");
    if (!grid) throw new Error(`${e.data.url} has no point data grid`);
    const t2 = performance.now();

    // Every point (no stride), prefix order, Cd and v; progress at most every 50 ms.
    lastPost = 0;
    const data = vdbPointRenderData(grid, {
      maxPoints: Infinity,
      order: "prefix",
      attributes: ["v"],
      onProgress: ({ stage, done, total: all }) => {
        const now = performance.now();
        if (done < all && now - lastPost < 50) return;
        lastPost = now;
        post({ type: "progress", stage, done, total: all });
      },
    });
    const n = data.count;
    const v = data.attributes.v;
    let speed: Float32Array | null = null;
    let speedRange: [number, number] = [0, 1];
    if (v && v.components === 3 && v.values instanceof Float32Array) {
      speed = new Float32Array(n);
      const vv = v.values;
      for (let i = 0; i < n; i++) speed[i] = Math.hypot(vv[i * 3]!, vv[i * 3 + 1]!, vv[i * 3 + 2]!);
      // Percentiles of |v| from an even sample (the 10% prefix is one).
      const sample = speed.slice(0, Math.min(n, 200_000)).sort();
      if (sample.length) speedRange = [sample[Math.floor(sample.length * 0.02)]!, sample[Math.floor(sample.length * 0.98)]!];
      if (!(speedRange[1] > speedRange[0])) speedRange = [speedRange[0], speedRange[0] + 1];
    }
    // The page draws |v|, not v: drop the vector before the transfer.
    data.attributes = {};
    const t3 = performance.now();

    const list = grid.points();
    const result: DecodedPoints = {
      data,
      speed,
      speedRange,
      attributes: Object.keys(list.attributes),
      skipped: list.skipped.map((s) => s.name),
      fileBytes: received,
      activeVoxels: Number(grid.activeVoxelCount),
      ms: { download: t1 - t0, parse: t2 - t1, decode: t3 - t2 },
    };
    const transfer: Transferable[] = vdbPointRenderDataTransfer(data);
    if (speed) transfer.push(speed.buffer);
    post({ type: "done", result }, transfer);
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err) });
  }
};
