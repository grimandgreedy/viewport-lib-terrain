# Changelog

## [1.3.0]

### Added

- World-space height and surface-normal queries on `TerrainItem`. `height_at(x, y)` returns the bilinear-filtered terrain height and `normal_at(x, y)` the finite-difference surface normal, both in world space with points outside the footprint clamping to the nearest edge. These read the same `u16` heightmap the renderer draws, so gameplay, camera, navigation, and foliage-placement code can sample the surface without keeping a separate copy or casting a pick ray.
- Physics-ready heightfield export. `TerrainItem::heightfield()` derives a `TerrainHeightfield` — a physics-engine-neutral, centred height grid plus the world-space `body_translation()` / `body_rotation()` that place it under the rendered surface — directly from the item's heightmap. A collision backend can register it as a static collider so characters walk on exactly the surface being drawn, with no second heightmap loaded or synthesised. The sample buffer is emitted in the centred, column-major, row-reversed layout that heightfield colliders expect, so the "thin spot" fall-through that a naive row-major buffer causes is handled once, here.

## [1.2.0]

### Added

- Terrain answers GPU object picking. `TerrainPlugin` renders its pick ids into viewport-lib's pick pass, so `pick_object(PickBackend::Gpu, ...)` returns terrain items without a CPU heightmap ray-cast. The CPU `pick` path is unchanged and remains the way to get an on-surface hit position.

## [1.1.0]

### Added

- Surface layers can sample a diffuse texture instead of a flat colour. `TerrainItem` holds an optional `LayerTextures` array with one RGBA texture per layer, and each `TerrainLayer` gains a `textured` flag plus `uv_scale` (world metres per tile) and `uv_offset`. Textured layers sample their diffuse through a wrapping sampler and blend by splat weight; untextured layers keep shading as their flat `albedo`, so existing terrains render unchanged.

## [1.0.0]

Initial release.

- `TerrainPlugin` registers as a viewport-lib `ItemTypePlugin` and draws inside the standard scene pass.
- `u16` heightmaps with world-space placement, footprint, and height range.
- Eight surface layers (albedo, metallic, roughness, height bias) blended through two RGBA splatmap textures with height-aware weighting.
- Quadtree-style patch LOD with skirts to hide cracks between neighbouring patches.
- Per-patch frustum culling.
- CPU ray-picking against the source heightmap.
- Detail scatter for grass billboards and tree meshes, with tree impostor LOD.
