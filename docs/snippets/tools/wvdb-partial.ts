// #region wvdb-partial
import { WvdbStreamReader, wvdbDensePartialVolume } from '@pmndrs/vdb/tools';
import { vdbDenseTextures, type VDBThreeDenseSource } from '@pmndrs/vdb/three';
// #endregion

export async function wvdbPartial(url: string, show: (source: VDBThreeDenseSource) => void): Promise<void> {
    // #region wvdb-partial
    // A partial event after every page; the levels are live, so build the volume before the next push.
    const reader = new WvdbStreamReader({ partialLods: true });
    let source: VDBThreeDenseSource | null = null;
    const response = await fetch(url);
    const stream = response.body!.getReader();
    for (;;) {
        const { done, value } = await stream.read();
        if (done) break;
        const event = reader.push(value).at(-1);
        if (!event) continue;
        // Bricks not decoded yet are upsampled from the coarser LOD (on the GPU, by the frame renderer's dense source).
        const volume = wvdbDensePartialVolume(reader.file!, event, { fill: 'upsample' });
        if (source) source.replace(volume);
        else show((source = vdbDenseTextures(volume))); // e.g. new VDBFrameVolume(source, { extinction: 100 })
    }
    reader.end();
    // #endregion
}
