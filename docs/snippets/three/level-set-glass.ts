// #region level-set-glass
import { Mesh } from 'three/webgpu';
import { VDBGridNode, VDBLevelSetPhysicalMaterial, vdbProxyGeometry } from '@pmndrs/vdb/three';
// #endregion

import type { Scene, WebGPURenderer } from 'three/webgpu';
import type { VDBGridPool } from '@pmndrs/vdb';

export function levelSetGlass(renderer: WebGPURenderer, pool: VDBGridPool, id: number, scene: Scene): VDBLevelSetPhysicalMaterial {
    // #region level-set-glass
    const grids = new VDBGridNode(renderer, pool, { profile: 'march', helpers: ['levelSet'] });
    // Three's physical material on the traced surface: transmission refracts from the hit, thickness in world units.
    const glass = new VDBLevelSetPhysicalMaterial(renderer, grids, id, {
        color: 0xffffff,
        roughness: 0.04,
        transmission: 1,
        ior: 1.5,
        thickness: 0.2,
        attenuationColor: 0xbfe3ff,
        attenuationDistance: 0.8,
    });
    scene.add(new Mesh(vdbProxyGeometry(pool.info(id), { pad: 1 }), glass));
    // Lacquer instead: { clearcoat: 1, clearcoatRoughness: 0.05 } (the clearcoat uses the traced normal).
    // #endregion
    return glass;
}
