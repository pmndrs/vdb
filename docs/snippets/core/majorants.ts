// #region majorant-host
import { VDBGridPool, vdbModuleEntries, vdbModuleWgsl, type VDBSlotModuleOptions } from '@pmndrs/vdb';
import { vdbBrickMajorant } from '@pmndrs/vdb/tools';
// #endregion

export async function majorantHost(device: GPUDevice, smoke: Uint32Array, cloud: Uint32Array, fire: Uint32Array) {
    // #region majorant-host
    const pool = await VDBGridPool.create(device);
    // Bounds over 8³ bricks for delta / ratio tracking, with minima for a residual tracker.
    // outside: 'zero' matches the module below (a tap outside the grid's bbox reads 0).
    const id = pool.add(smoke, { majorant: vdbBrickMajorant(smoke, { minima: true, outside: 'zero' }) });

    const slot = { group: 2, firstBinding: 0 };
    const options: VDBSlotModuleOptions = {
        slot,
        types: [...pool.types],
        profile: 'majorant', // sample + vdbMajorantInit / vdbMajorantNext
        majorant: 'brick',
        minima: true,
        outside: 'zero',
        reader: 'lean', // a fifth of the vendored accessor's size, same values
    };
    const module = vdbModuleWgsl(options);
    const layout = device.createBindGroupLayout({ entries: vdbModuleEntries(slot) });
    // #endregion

    // #region multi-grid
    // Several grids, one binding: one module and one pool serve every medium; the shader picks the grid by id.
    const media = [id, pool.add(cloud), pool.add(fire)];
    pool.setTransform(media[1]!, [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 10, 0, -4, 1]); // instance: scale 2, then move
    pool.setTransform(media[2]!, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -6, 1, 0, 1]);
    // A host keeps its own per-medium data (sigma, albedo) in its own buffer, indexed the same way.
    // #endregion
    return { pool, module, layout, media };
}
