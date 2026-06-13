# Changelog

## [1.0.0]

Initial release.

- `TerrainPlugin` registers as a viewport-lib `ItemTypePlugin` and draws inside the standard scene pass.
- `u16` heightmaps with world-space placement, footprint, and height range.
- Eight surface layers (albedo, metallic, roughness, height bias) blended through two RGBA splatmap textures with height-aware weighting.
- Quadtree-style patch LOD with skirts to hide cracks between neighbouring patches.
- Per-patch frustum culling.
- CPU ray-picking against the source heightmap.
- Detail scatter for grass billboards and tree meshes, with tree impostor LOD.
