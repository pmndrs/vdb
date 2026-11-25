// #region migrate-0-2
import { VDB_FRAME_LIGHT_BITS, type VDBFrameRenderer, type VDBFrameSettings, type VDBFrameVolume } from '@pmndrs/vdb/three';
// #endregion

export function migrate(frameRenderer: VDBFrameRenderer, volume: VDBFrameVolume): VDBFrameSettings {
    // #region migrate-0-2
    // 0.1: frameRenderer.neuralNetwork = weights;   0.2: the generic MLP data (null: 1D diffusion)
    frameRenderer.mlp = null;
    // 0.1: frameRenderer.holdFroxels = true;   0.2: false keeps the last froxels
    frameRenderer.froxels = false;
    // 0.1: vdbFrameBlackbodyEmission('planck', 1.5);   0.2: plain data
    volume.emissionColor = { spectrum: 'planck', exponent: 1.5 };
    // 0.1: vdbFrameLights.directional | vdbFrameLights.ambient
    volume.lights = VDB_FRAME_LIGHT_BITS.directional | VDB_FRAME_LIGHT_BITS.ambient;
    // 0.1: VDBFrameRendererSettings;   0.2: the core's VDBFrameSettings
    const settings: VDBFrameSettings = frameRenderer.settings;
    // the core renderer itself, after the first render (its outputs: shadow maps, emission probes)
    console.log(frameRenderer.core?.outputs);
    // #endregion
    return settings;
}
