//! Host-side scatter helpers for grass, foliage, and tree LOD.
//!
//! Detail meshes and billboards are not part of the terrain item itself.
//! The host configures a list of [`DetailLayer`]s, calls
//! [`scatter_terrain_details`](crate::scatter_terrain_details) once per frame, and submits the returned
//! `SpriteItem` batches and opaque `SceneRenderItem`s via the regular
//! sprite and surface-item paths.
//!
//! Positions are generated from a Halton(2,3) low-discrepancy sequence
//! over the terrain bounds and accepted with a probability proportional
//! to the per-layer surface weight at that point (read from the
//! splatmap channel that owns the layer). This gives stable,
//! frame-coherent placement without retaining any state, so grass
//! tufts do not pop or swim as the camera moves.
//!
//! All sampling is clamped to a circular footprint around
//! [`DetailScatterParams::camera_xy`]; samples outside the footprint
//! are skipped without per-sample work.

use viewport_lib_types::colour::Colour;
use viewport_lib_types::ids::{MaterialId, MeshId, TextureId};

/// What a [`DetailLayer`] emits per scatter point.
#[derive(Clone)]
pub enum DetailKind {
    /// Camera-facing billboard. One [`SpriteItem`] per layer, world-space sized.
    Billboard {
        /// Optional texture handle from `ViewportGpuResources::upload_texture`.
        /// `None` renders a solid-coloured quad.
        texture_id: Option<TextureId>,
        /// RGBA tint applied to every instance.
        colour: Colour,
    },
    /// Mesh. One opaque scene item per placed instance.
    Mesh {
        mesh_id: MeshId,
        /// Material from `upload_material`. Every instance shares it.
        material: MaterialId,
        /// Replaces the material's base colour on each instance.
        colour: Colour,
    },
    /// Tree-style LOD: full mesh under [`TreeLod::switch_distance`],
    /// camera-facing impostor billboard beyond. One mesh batch + one
    /// sprite batch per layer.
    TreeLod {
        mesh_id: MeshId,
        /// Material from `upload_material` for the full mesh.
        mesh_material: MaterialId,
        impostor_texture_id: Option<TextureId>,
        colour: Colour,
        /// Distance from camera at which the renderer switches the
        /// individual tree from mesh to impostor.
        switch_distance: f32,
    },
}

/// A single scatter recipe over the terrain surface.
#[derive(Clone)]
pub struct DetailLayer {
    pub kind: DetailKind,
    /// Index `0..LAYER_COUNT` into the terrain's surface layers; the
    /// matching splatmap channel weights placement (stronger weight =
    /// more accepted samples here).
    pub surface_layer: usize,
    /// Acceptance threshold against the splatmap channel value (0..1).
    /// Lower = scatter spreads further; higher = denser clusters in the
    /// painted region.
    pub weight_threshold: f32,
    /// World-space size of each instance (metres). For billboards this
    /// is the quad edge length; for meshes this is the uniform scale.
    pub size: f32,
    /// Random size jitter as a fraction of [`size`](Self::size).
    pub size_jitter: f32,
    /// Maximum distance from the camera at which this layer is
    /// scattered. Beyond this radius no samples are emitted.
    pub max_distance: f32,
    /// Per-square-metre target density.
    pub density: f32,
}

/// Per-frame scatter inputs read from the host's camera state.
pub struct DetailScatterParams {
    /// Camera position projected onto the terrain plane (XY in world
    /// space). The scatter footprint is the circle of radius
    /// `max(layer.max_distance)` centred here.
    pub camera_xy: glam::Vec2,
    /// Camera height above the world. Used only to break the trees'
    /// mesh/impostor split into per-tree distance comparisons.
    pub camera_z: f32,
}
