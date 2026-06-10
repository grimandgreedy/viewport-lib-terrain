//! Terrain item types submitted on `SceneFrame::plugin_items`.

use std::any::Any;
use std::sync::Arc;

use viewport_lib::ItemSettings;
use viewport_lib::plugin_api::PluginItemCollection;
use viewport_lib::renderer::PickId;

/// Number of surface layers in a [`TerrainItem`]. Two splatmaps carry
/// four channels each; channels map to layers in order.
pub const LAYER_COUNT: usize = 8;

/// One surface layer in a splatmap-blended terrain.
#[derive(Copy, Clone, Debug)]
pub struct TerrainLayer {
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub height_bias: f32,
}

impl Default for TerrainLayer {
    fn default() -> Self {
        Self {
            albedo: [0.5, 0.5, 0.5],
            metallic: 0.0,
            roughness: 0.9,
            height_bias: 0.0,
        }
    }
}

/// Per-channel layer weights painted across the terrain.
///
/// `rgba` is held behind an [`Arc`] so consumers can clone the
/// `SplatmapData` per frame without copying the underlying bytes.
/// Edit by building a new `Vec<u8>` and calling
/// [`SplatmapData::replace`], or by `Arc::make_mut`ing in place; in
/// either case bump [`version`](Self::version) so the renderer
/// detects the change and re-uploads the texture.
#[derive(Clone)]
pub struct SplatmapData {
    rgba: Arc<[u8]>,
    /// Splatmap resolution (width, height).
    pub dims: [u32; 2],
    /// Bumped by the consumer when `rgba` changes. The renderer
    /// compares the stored version against the last upload to decide
    /// whether to push new texture data; no byte-level comparison or
    /// hashing happens on the hot path.
    pub version: u64,
}

impl SplatmapData {
    /// Construct from an owned byte vector. The vector is moved into
    /// an `Arc<[u8]>` so subsequent clones of the `SplatmapData`
    /// share the bytes.
    pub fn new(rgba: Vec<u8>, dims: [u32; 2]) -> Self {
        Self {
            rgba: Arc::from(rgba.into_boxed_slice()),
            dims,
            version: 0,
        }
    }

    /// Borrow the RGBA bytes.
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Replace the bytes with a new buffer. Bumps `version` so the
    /// renderer re-uploads on the next frame.
    pub fn replace(&mut self, rgba: Vec<u8>, dims: [u32; 2]) {
        self.rgba = Arc::from(rgba.into_boxed_slice());
        self.dims = dims;
        self.version = self.version.wrapping_add(1);
    }

    /// A 1x1 splatmap with all-zero channels. Use this for the second
    /// splatmap when a terrain only uses layers 0..4.
    pub fn empty() -> Self {
        Self::new(vec![0, 0, 0, 0], [1, 1])
    }

    /// A 1x1 splatmap that selects layer 0 (channel R) everywhere.
    pub fn solid_layer0() -> Self {
        Self::new(vec![255, 0, 0, 0], [1, 1])
    }

    /// Pack per-layer single-channel weight maps into the two-splatmap
    /// DRAKE layout.
    ///
    /// `per_layer` is one byte buffer per layer, each of length
    /// `dims[0] * dims[1]`, holding `u8` weights in `[0, 255]`. Up to
    /// the first eight layers are packed: layers 0..4 land in splatmap
    /// A (channels R, G, B, A in that order); layers 4..8 land in
    /// splatmap B. Missing trailing layers are zero-filled.
    ///
    /// Returns `None` if any provided buffer has the wrong length or
    /// `per_layer` is empty.
    pub fn pack_layers(per_layer: &[Vec<u8>], dims: [u32; 2]) -> Option<[Self; 2]> {
        let pixel_count = (dims[0] as usize) * (dims[1] as usize);
        if per_layer.is_empty() {
            return None;
        }
        for buf in per_layer {
            if buf.len() != pixel_count {
                return None;
            }
        }
        let pack = |range: std::ops::Range<usize>| -> Vec<u8> {
            let mut out = vec![0u8; pixel_count * 4];
            for (slot, layer_idx) in range.enumerate() {
                if let Some(src) = per_layer.get(layer_idx) {
                    for (px, value) in src.iter().enumerate() {
                        out[px * 4 + slot] = *value;
                    }
                }
            }
            out
        };
        Some([Self::new(pack(0..4), dims), Self::new(pack(4..8), dims)])
    }
}

/// One terrain submission for the current frame.
///
/// `heightmap` is held behind an [`Arc`] so per-frame submissions
/// clone cheaply. Bump [`heightmap_version`](Self::heightmap_version)
/// whenever you replace the heightmap so the renderer re-bakes the
/// patch meshes.
#[derive(Clone)]
pub struct TerrainItem {
    /// 16-bit heightmap samples in row-major order, shared via `Arc`.
    pub heightmap: Arc<[u16]>,
    /// Bumped when [`heightmap`](Self::heightmap) bytes change.
    pub heightmap_version: u64,
    /// Grid resolution (width, height).
    pub dims: [u32; 2],
    /// World-space extents along X and Y. Z-up.
    pub world_size: [f32; 2],
    /// World-space height range; samples are remapped from `u16` to
    /// `[height_range[0], height_range[1]]`.
    pub height_range: [f32; 2],
    /// World-space origin (lower-left corner in XY).
    pub origin: glam::Vec3,
    /// Eight surface layers.
    pub surface_layers: [TerrainLayer; LAYER_COUNT],
    /// Two splatmaps. Channels: A.r -> 0, A.g -> 1, ..., B.a -> 7.
    pub splatmaps: [SplatmapData; 2],
    /// Sharpness of the height-blend pass. 0 = pure weight blend.
    pub height_blend_strength: f32,
    /// Scale of the procedural noise feeding the height-blend.
    pub height_blend_noise_scale: f32,
    pub settings: ItemSettings,
}

impl TerrainItem {
    /// Construct with `heightmap_version` set to 0. Subsequent
    /// edits to the heightmap should call [`replace_heightmap`].
    pub fn new(heightmap: Vec<u16>, dims: [u32; 2]) -> Self {
        Self {
            heightmap: Arc::from(heightmap.into_boxed_slice()),
            heightmap_version: 0,
            dims,
            world_size: [1.0, 1.0],
            height_range: [0.0, 1.0],
            origin: glam::Vec3::ZERO,
            surface_layers: [TerrainLayer::default(); LAYER_COUNT],
            splatmaps: [SplatmapData::solid_layer0(), SplatmapData::empty()],
            height_blend_strength: 0.0,
            height_blend_noise_scale: 64.0,
            settings: ItemSettings::default(),
        }
    }

    /// Replace the heightmap and bump [`heightmap_version`].
    pub fn replace_heightmap(&mut self, heightmap: Vec<u16>, dims: [u32; 2]) {
        self.heightmap = Arc::from(heightmap.into_boxed_slice());
        self.dims = dims;
        self.heightmap_version = self.heightmap_version.wrapping_add(1);
    }

    /// Construct a fully-specified `TerrainItem` from a raw little-endian
    /// `u16` heightmap blob.
    ///
    /// Convenience for loaders that produce on-disk heightmap files (raw
    /// `.r16`, Unity `TerrainData` exports, Terragen, etc.): owns the
    /// `u16` LE decode once so callers do not redo the byte handling.
    ///
    /// `bytes.len()` must equal `dims[0] * dims[1] * 2`; otherwise this
    /// returns `None`.
    pub fn from_u16_le_bytes(
        bytes: &[u8],
        dims: [u32; 2],
        world_size: [f32; 2],
        height_range: [f32; 2],
        origin: glam::Vec3,
    ) -> Option<Self> {
        let expected = (dims[0] as usize) * (dims[1] as usize) * 2;
        if bytes.len() != expected {
            return None;
        }
        let mut heights: Vec<u16> = Vec::with_capacity(expected / 2);
        for chunk in bytes.chunks_exact(2) {
            heights.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        let mut item = Self::new(heights, dims);
        item.world_size = world_size;
        item.height_range = height_range;
        item.origin = origin;
        Some(item)
    }

    /// Replace the eight surface layers from a normalised descriptor list.
    ///
    /// Slots `0..descriptors.len().min(LAYER_COUNT)` are overwritten;
    /// the remainder fall back to [`TerrainLayer::default`]. Useful for
    /// imported terrains where the source authored fewer than eight
    /// layers and the rest should stay neutral.
    pub fn set_layers_from_descriptors(&mut self, descriptors: &[TerrainLayer]) {
        let mut layers = [TerrainLayer::default(); LAYER_COUNT];
        for (slot, src) in layers.iter_mut().zip(descriptors.iter().take(LAYER_COUNT)) {
            *slot = *src;
        }
        self.surface_layers = layers;
    }
}

/// Per-frame collection of terrains.
pub struct TerrainCollection {
    pub items: Vec<TerrainItem>,
}

impl PluginItemCollection for TerrainCollection {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn item_settings(&self, index: usize) -> &ItemSettings {
        &self.items[index].settings
    }

    fn pick_id(&self, index: usize) -> PickId {
        self.items[index].settings.pick_id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_u16_le_bytes_round_trips_a_ramp() {
        let dims = [4u32, 3u32];
        let heights: Vec<u16> = (0u16..12u16).map(|i| i * 1000).collect();
        let mut bytes = Vec::with_capacity(heights.len() * 2);
        for h in &heights {
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        let item = TerrainItem::from_u16_le_bytes(
            &bytes,
            dims,
            [10.0, 20.0],
            [-5.0, 100.0],
            glam::Vec3::new(1.0, 2.0, 3.0),
        )
        .expect("decode succeeds");
        assert_eq!(item.dims, dims);
        assert_eq!(item.world_size, [10.0, 20.0]);
        assert_eq!(item.height_range, [-5.0, 100.0]);
        assert_eq!(item.origin, glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(item.heightmap.as_ref(), heights.as_slice());
    }

    #[test]
    fn from_u16_le_bytes_rejects_wrong_length() {
        let bytes = vec![0u8; 7];
        assert!(
            TerrainItem::from_u16_le_bytes(
                &bytes,
                [2, 2],
                [1.0, 1.0],
                [0.0, 1.0],
                glam::Vec3::ZERO,
            )
            .is_none()
        );
    }

    #[test]
    fn pack_layers_routes_channels_correctly() {
        let dims = [2u32, 2u32];
        let pixel_count = 4;
        let mut per_layer = Vec::new();
        for layer in 0..6 {
            per_layer.push(vec![(layer * 10) as u8; pixel_count]);
        }
        let [a, b] = SplatmapData::pack_layers(&per_layer, dims).expect("pack");
        for px in 0..pixel_count {
            assert_eq!(a.rgba()[px * 4], 0);
            assert_eq!(a.rgba()[px * 4 + 1], 10);
            assert_eq!(a.rgba()[px * 4 + 2], 20);
            assert_eq!(a.rgba()[px * 4 + 3], 30);
            assert_eq!(b.rgba()[px * 4], 40);
            assert_eq!(b.rgba()[px * 4 + 1], 50);
            assert_eq!(b.rgba()[px * 4 + 2], 0);
            assert_eq!(b.rgba()[px * 4 + 3], 0);
        }
    }

    #[test]
    fn pack_layers_rejects_short_buffers() {
        let dims = [2u32, 2u32];
        let per_layer = vec![vec![0u8; 3]];
        assert!(SplatmapData::pack_layers(&per_layer, dims).is_none());
    }

    #[test]
    fn set_layers_from_descriptors_pads_with_default() {
        let mut item = TerrainItem::new(vec![0; 1], [1, 1]);
        let custom = TerrainLayer {
            albedo: [0.1, 0.2, 0.3],
            metallic: 0.5,
            roughness: 0.4,
            height_bias: 0.7,
        };
        item.set_layers_from_descriptors(&[custom]);
        assert_eq!(item.surface_layers[0].albedo, [0.1, 0.2, 0.3]);
        let default = TerrainLayer::default();
        assert_eq!(item.surface_layers[1].albedo, default.albedo);
        assert_eq!(item.surface_layers[7].albedo, default.albedo);
    }
}
