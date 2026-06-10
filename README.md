# viewport-lib-terrain

An `ItemTypePlugin` providing heightmap and splatmap terrain rendering for [viewport-lib](https://github.com/grimandgreedy/viewport-lib).

## Why not just a mesh?

You can put a triangulated landscape into viewport-lib as a regular mesh item. It will render. What you give up:

- **Frustum culling at patch granularity.** Off-screen patches skip the draw entirely.
- **Memory.** A 2049x2049 heightmap is roughly 8 MB ( with `u16`). The same surface as a triangle mesh with per-vertex normals and UVs is hundreds of MB before you've drawn the first frame.
- **LOD.** This crate splits the terrain into a quadtree of patches and picks a decimation level per patch per frame, with skirts to hide cracks between neighbouring patches. A static mesh draws every triangle every frame.
- **Material.** Eight surface layers (albedo, metallic, roughness) blended on the GPU through two splatmap textures with height-aware weighting. Doing this on a mesh means an eight-pass material, a custom shader, or baking it down and losing the layers.

If your terrain is small and single-textured, a mesh is fine. Past that, this crate is the cheaper path.

## What you can attach

- A `u16` heightmap, loaded directly from raw `.r16` style blobs or built in memory.
- Two RGBA splatmap textures driving up to eight surface layers. Per-layer weight maps can be packed into the splatmap channels for you.
- Eight `TerrainLayer` slots, each carrying albedo, metallic, roughness, and a height bias for the height-aware blend.
- World-space origin, footprint, and height range so the same heightmap can be placed and scaled anywhere in the scene.
- Detail scatter (grass, trees, props) over the surface, with tree impostor LOD for distant instances.

## Examples

```sh
cargo run --release --example eframe-terrain
cargo run --release --example eframe-terrain-lod
cargo run --release --example eframe-terrain-stress
cargo run --release --example eframe-plugin-jobs
```

See [`examples/`](examples/) for the source of each demo, including the plugin registration, heightmap loading, splatmap packing, and detail scatter setup.
