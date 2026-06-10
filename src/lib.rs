//! Heightmap and splatmap terrain rendering for viewport-lib.
//!
//! The plugin registers as a viewport-lib
//! [`ItemTypePlugin`](viewport_lib::plugin_api::ItemTypePlugin) and
//! draws terrain inside the standard scene pass, picking up shadows,
//! clipping, selection outline, picking, and frustum culling from the
//! lib automatically.
//!
//! ```no_run
//! # use viewport_lib_terrain::*;
//! # use viewport_lib::ItemSettings;
//! # let mut renderer: viewport_lib::renderer::ViewportRenderer = unimplemented!();
//! # let device: &wgpu::Device = unimplemented!();
//! renderer.with_item_type_plugin(device, Box::new(TerrainPlugin::new()));
//!
//! // Each frame:
//! # let mut frame: viewport_lib::FrameData = Default::default();
//! frame.scene.submit_plugin_items(
//!     TYPE_NAME,
//!     TerrainCollection { items: vec![] },
//! );
//! ```
//!
//! See `examples/eframe_terrain.rs` for a runnable demo.

mod detail;
mod item;
mod mesh;
mod pick;
mod plugin;

pub use detail::{
    DetailKind, DetailLayer, DetailScatterOutput, DetailScatterParams, scatter_terrain_details,
};
pub use item::{LAYER_COUNT, SplatmapData, TerrainCollection, TerrainItem, TerrainLayer};
pub use plugin::TerrainPlugin;

/// Stable name used as the
/// [`SceneFrame::submit_plugin_items`](viewport_lib::renderer::SceneFrame::submit_plugin_items)
/// key. Match this against
/// [`TerrainPlugin::type_name`](viewport_lib::plugin_api::ItemTypePlugin::type_name)
/// when wiring the host side.
pub const TYPE_NAME: &str = "viewport-lib-terrain";
