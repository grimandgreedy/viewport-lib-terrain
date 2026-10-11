# viewport-lib-terrain

An `ItemTypePlugin` for heightmap and splatmap terrain in [viewport-lib](https://github.com/grimandgreedy/viewport-lib).

![A textured Mount Rainier terrain rendered by viewport-lib-terrain](https://raw.githubusercontent.com/grimandgreedy/viewport-lib-terrain/master/assets/terrain.jpg)

## Why not a mesh

You can load a triangulated landscape into viewport-lib as an ordinary mesh item, and it will draw. For a large terrain this crate is cheaper in a few ways:

- The terrain is cut into patches, and patches outside the view are not drawn.
- A 2049x2049 heightmap is about 8 MB as `u16`. The same surface as a triangle mesh with normals and UVs is well over 100 MB.
- Each patch is baked at several levels of detail, and the plugin picks one per patch every frame. Skirts hide the cracks between neighbours at different levels. A static mesh draws every triangle every frame.
- Up to eight surface layers are blended on the GPU from two splatmaps. On a mesh you would need a custom shader or a baked texture, and the separate layers would be lost.

If your terrain is small and uses one texture, a mesh is fine.

## Using it

Register the plugin once, then build a `TerrainItem` and submit it each frame.

```rust
renderer.with_item_type_plugin(device, Box::new(TerrainPlugin::new()));

let mut terrain = TerrainItem::new(heightmap, [513, 513]);
terrain.world_size = [256.0, 256.0];
terrain.height_range = [0.0, 40.0];
terrain.splatmaps = [
    SplatmapData::new(splat_a, [1024, 1024]),
    SplatmapData::new(splat_b, [1024, 1024]),
];

frame.scene.submit_plugin_items(
    TYPE_NAME,
    TerrainCollection { items: vec![terrain.clone()] },
);
```

The heightmap is a `Vec<u16>`. `TerrainItem::from_u16_le_bytes` builds one from a raw `.r16` file. The two splatmaps hold four layer weights each, so a terrain has up to eight surface layers. A layer is flat colour with metallic and roughness values until you give the item texture arrays, and then it samples diffuse, normal and mask maps. `examples/common/assets.rs` shows how the examples load them.

Grass and trees can be scattered over the surface with `scatter_terrain_details`. The scattered meshes are ordinary opaque scene items. Distant trees switch to billboards, which only draw if the sprite plugin from `viewport-lib-plugins` is registered.

## Examples

```sh
cargo run --release --example eframe-terrain
cargo run --release --example eframe-terrain-lod
cargo run --release --example eframe-terrain-detail
cargo run --release --example eframe-terrain-stress
cargo run --release --example eframe-plugin-jobs
```

- `eframe-terrain` is the basic case: a real heightmap with eight textured layers.
- `eframe-terrain-lod` shows the patch LOD, with a readout of how many patches draw at each level.
- `eframe-terrain-detail` scatters grass and trees.
- `eframe-terrain-stress` draws a grid of terrains.
- `eframe-plugin-jobs` generates heightmaps on a worker thread.

The textures, heightmap and tree models are in `examples/assets`, and their sources are listed in `examples/assets/CREDITS.md`. The source in [`examples/`](examples/) shows plugin registration, heightmap loading, splatmap packing and detail scatter.
