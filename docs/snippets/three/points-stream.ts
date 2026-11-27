// #region points-render-data
import { parseVdb, vdbPointRenderData, vdbPointRenderDataTransfer, type VdbPointDataGrid } from '@pmndrs/vdb/tools';
// #endregion
// #region points-wvdb
import { WvdbStreamReader, type WvdbPointsGridHeader, type WvdbPointsLevel } from '@pmndrs/vdb/tools';
import { VDBPoints } from '@pmndrs/vdb/three';
// #endregion

import type { Scene } from 'three/webgpu';

/** In a worker: decode, order for subsampling, and hand the arrays to the page without a copy. */
export function decodeInWorker(bytes: ArrayBuffer): void {
    // #region points-render-data
    const grid = parseVdb(bytes).grids.find((g): g is VdbPointDataGrid => g.valueType === 'ptdataidx32')!;
    const data = vdbPointRenderData(grid, {
        order: 'prefix', // data.prefix[k]: the first k tenths are an even subsample (prefix[1] = every 10th point)
        attributes: ['v'], // data.attributes.v.values, in the same order as data.positions
        onProgress: ({ stage, done, total }) => postMessage({ stage, done, total }),
    });
    postMessage(data, { transfer: vdbPointRenderDataTransfer(data) });
    // In the page: new VDBPoints(data); later points.pointCount = data.prefix![1] draws 10 % of them.
    // #endregion
}

/** Stream a `.wvdb` points grid into a VDBPoints, coarse LODs first. */
export async function streamPoints(url: string, scene: Scene): Promise<VDBPoints> {
    // #region points-wvdb
    const reader = new WvdbStreamReader({ partialLods: true });
    const body = (await fetch(url)).body!.getReader();
    let points: VDBPoints | null = null;
    for (;;) {
        const { done, value } = await body.read();
        if (done) break;
        for (const event of reader.push(value)) {
            if (!points) {
                // Sized for the whole grid; draws nothing until the first level arrives.
                points = VDBPoints.fromWvdbGrid(reader.header!.grids[0] as WvdbPointsGridHeader);
                scene.add(points);
            }
            // Uploads the new range, draws every point that has arrived, widens the sprites while it is sparse.
            points.addWvdbLevel(event.levels[0] as WvdbPointsLevel);
        }
    }
    reader.end();
    // #endregion
    return points!;
}
