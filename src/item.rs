//! Terrain item types submitted on `SceneFrame::plugin_items`.

use std::sync::Arc;

use viewport_lib_types::colour::{Colour, ColourSpace};
use viewport_lib_types::data::texture::{TextureData, TexturePayload};
use viewport_lib_types::material::ItemSettings;

use crate::heightfield::TerrainHeightfield;

/// Number of surface layers in a [`TerrainItem`]. Two splatmaps carry
/// four channels each; channels map to layers in order.
pub const LAYER_COUNT: usize = 8;

/// One surface layer in a splatmap-blended terrain.
///
/// When `textured` is false the layer shades as a flat `albedo` colour
/// (alpha is ignored).
/// When true it samples slot `i` of the terrain's
/// [`LayerTextures`](TerrainItem::layer_textures) array, tiled across the
/// surface by `uv_scale` (world metres per tile) and shifted by
/// `uv_offset`. `albedo` still acts as a tint fallback if the terrain has
/// no texture array bound.
#[derive(Copy, Clone, Debug)]
pub struct TerrainLayer {
    pub albedo: Colour,
    pub metallic: f32,
    pub roughness: f32,
    pub height_bias: f32,
    /// Whether this layer samples the diffuse texture array.
    pub textured: bool,
    /// Whether this layer samples the normal-map array. When false the
    /// layer uses the flat geometric surface normal.
    pub normal_mapped: bool,
    /// Strength of the tangent-space normal perturbation. 1.0 is the
    /// authored strength; 0.0 flattens it.
    pub normal_scale: f32,
    /// Whether this layer samples the mask-map array for per-pixel
    /// metallic, ambient occlusion, and smoothness. When false the layer
    /// uses its flat `metallic` / `roughness` scalars.
    pub mask_mapped: bool,
    /// Per-channel remap minimums applied to the mask sample
    /// (R metallic, G AO, B detail, A smoothness).
    pub mask_remap_min: [f32; 4],
    /// Per-channel remap maximums applied to the mask sample.
    pub mask_remap_max: [f32; 4],
    /// World metres covered by one tile of the layer's textures, along X
    /// and Y. Values at or below zero are clamped to keep the tiling
    /// finite. Shared by the diffuse, normal, and mask maps.
    pub uv_scale: [f32; 2],
    /// Constant shift added to the tiled coordinate, in tile units.
    pub uv_offset: [f32; 2],
}

impl Default for TerrainLayer {
    fn default() -> Self {
        Self {
            albedo: Colour::linear_rgb(0.5, 0.5, 0.5),
            metallic: 0.0,
            roughness: 0.9,
            height_bias: 0.0,
            textured: false,
            normal_mapped: false,
            normal_scale: 1.0,
            mask_mapped: false,
            mask_remap_min: [0.0, 0.0, 0.0, 0.0],
            mask_remap_max: [1.0, 1.0, 1.0, 1.0],
            uv_scale: [1.0, 1.0],
            uv_offset: [0.0, 0.0],
        }
    }
}

/// A per-layer texture array (diffuse, normal or mask), packed so every
/// layer samples from a single binding.
///
/// Built from [`TextureData`] slots, one per layer, which must all be
/// RGBA8, the same size and in the same colour space. The colour space
/// decides how the array is uploaded: sRGB slots (diffuse colour, from
/// [`TextureData::srgb`]) are decoded to linear on sampling, linear slots
/// (normal and mask maps, from [`TextureData::normal_map`] or
/// [`TextureData::linear`]) are read as-is. Slots without a source are
/// transparent black and should be paired with a [`TerrainLayer`] whose
/// matching flag (`textured` / `normal_mapped` / `mask_mapped`) is false.
///
/// The pixels sit behind an [`Arc`] so per-frame clones of the owning
/// [`TerrainItem`] do not copy them. Bump [`version`](Self::version)
/// when the bytes change so the renderer re-uploads.
#[derive(Clone)]
pub struct LayerTextures {
    rgba: Arc<[u8]>,
    dims: [u32; 2],
    colour_space: ColourSpace,
    /// Bumped by the consumer when the pixels change.
    pub version: u64,
}

/// Why [`LayerTextures::new`] rejected its sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerTexturesError {
    /// No sources were given.
    Empty,
    /// More than [`LAYER_COUNT`] sources were given.
    TooMany(usize),
    /// The slot is not an 8-bit RGBA image.
    NotRgba8 { slot: usize },
    /// The slot's size differs from slot 0's, or its payload length does
    /// not match its size.
    SizeMismatch { slot: usize },
    /// The slot's colour space differs from slot 0's.
    ColourSpaceMismatch { slot: usize },
}

impl std::fmt::Display for LayerTexturesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "no layer textures given"),
            Self::TooMany(n) => write!(f, "{n} layer textures given, at most {LAYER_COUNT} allowed"),
            Self::NotRgba8 { slot } => write!(f, "layer texture {slot} is not RGBA8"),
            Self::SizeMismatch { slot } => {
                write!(f, "layer texture {slot} is not the same size as layer texture 0")
            }
            Self::ColourSpaceMismatch { slot } => {
                write!(f, "layer texture {slot} is not in the same colour space as layer texture 0")
            }
        }
    }
}

impl std::error::Error for LayerTexturesError {}

impl LayerTextures {
    /// Assemble from up to [`LAYER_COUNT`] textures in layer order.
    /// Trailing slots are filled with transparent black.
    pub fn new(sources: &[TextureData]) -> Result<Self, LayerTexturesError> {
        let first = sources.first().ok_or(LayerTexturesError::Empty)?;
        if sources.len() > LAYER_COUNT {
            return Err(LayerTexturesError::TooMany(sources.len()));
        }
        let dims = [first.width(), first.height()];
        let colour_space = first.colour_space();
        let slot_len = (dims[0] as usize) * (dims[1] as usize) * 4;
        if slot_len == 0 {
            return Err(LayerTexturesError::SizeMismatch { slot: 0 });
        }
        let mut rgba = vec![0u8; slot_len * LAYER_COUNT];
        for (slot, src) in sources.iter().enumerate() {
            let TexturePayload::Rgba8(bytes) = src.payload() else {
                return Err(LayerTexturesError::NotRgba8 { slot });
            };
            if [src.width(), src.height()] != dims || bytes.len() != slot_len {
                return Err(LayerTexturesError::SizeMismatch { slot });
            }
            if src.colour_space() != colour_space {
                return Err(LayerTexturesError::ColourSpaceMismatch { slot });
            }
            let start = slot * slot_len;
            rgba[start..start + slot_len].copy_from_slice(bytes);
        }
        Ok(Self {
            rgba: Arc::from(rgba.into_boxed_slice()),
            dims,
            colour_space,
            version: 0,
        })
    }

    /// Borrow the packed RGBA bytes: [`LAYER_COUNT`] slots of
    /// `dims[0] * dims[1] * 4` each.
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Per-slot texture resolution (width, height).
    pub fn dims(&self) -> [u32; 2] {
        self.dims
    }

    /// The colour space every slot is in.
    pub fn colour_space(&self) -> ColourSpace {
        self.colour_space
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
    /// Diffuse textures for the surface layers. `None` shades every
    /// layer as its flat `albedo`; `Some` lets layers with `textured`
    /// set sample the array.
    pub layer_textures: Option<LayerTextures>,
    /// Normal-map textures for the surface layers. `None` uses the flat
    /// geometric normal; `Some` lets layers with `normal_mapped` set
    /// perturb the surface. Shares the `uv_scale` / `uv_offset` tiling
    /// with the diffuse array.
    pub normal_textures: Option<LayerTextures>,
    /// Mask-map textures for the surface layers (R metallic, G AO,
    /// B detail, A smoothness). `None` uses the flat per-layer scalars;
    /// `Some` lets layers with `mask_mapped` set drive metallic / AO /
    /// smoothness per pixel. Shares the diffuse array's tiling.
    pub mask_textures: Option<LayerTextures>,
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
            layer_textures: None,
            normal_textures: None,
            mask_textures: None,
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

    /// Bilinear-filtered world-space height at `(x, y)`.
    ///
    /// Treats the heightmap as a uniform grid stretched across
    /// [`world_size`](Self::world_size) from [`origin`](Self::origin) (lower-left
    /// in `xy`) and remaps samples through [`height_range`](Self::height_range).
    /// Points outside the footprint clamp to the nearest edge — the same value
    /// the renderer shows there. Cheap enough to call per query for physics,
    /// camera, navigation, or foliage placement.
    pub fn height_at(&self, x: f32, y: f32) -> f32 {
        let cols = self.dims[0].max(1) as i32;
        let rows = self.dims[1].max(1) as i32;
        let u = ((x - self.origin.x) / self.world_size[0]).clamp(0.0, 1.0);
        let v = ((y - self.origin.y) / self.world_size[1]).clamp(0.0, 1.0);
        let fx = u * (cols - 1) as f32;
        let fy = v * (rows - 1) as f32;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let x1 = (x0 + 1).min(cols - 1);
        let y1 = (y0 + 1).min(rows - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let span = self.height_range[1] - self.height_range[0];
        let sample = |sx: i32, sy: i32| -> f32 {
            let raw = self.heightmap[(sy * cols + sx) as usize] as f32 / u16::MAX as f32;
            self.height_range[0] + raw * span
        };
        let h00 = sample(x0, y0);
        let h10 = sample(x1, y0);
        let h01 = sample(x0, y1);
        let h11 = sample(x1, y1);
        let h0 = h00 * (1.0 - tx) + h10 * tx;
        let h1 = h01 * (1.0 - tx) + h11 * tx;
        h0 * (1.0 - ty) + h1 * ty
    }

    /// World-space surface normal at `(x, y)`, from finite differences over the
    /// heightmap. `z`-up convention; the returned vector is normalised.
    pub fn normal_at(&self, x: f32, y: f32) -> glam::Vec3 {
        let cols = self.dims[0].max(2) as f32;
        let rows = self.dims[1].max(2) as f32;
        let dx = self.world_size[0] / (cols - 1.0);
        let dy = self.world_size[1] / (rows - 1.0);
        let hl = self.height_at(x - dx, y);
        let hr = self.height_at(x + dx, y);
        let hd = self.height_at(x, y - dy);
        let hu = self.height_at(x, y + dy);
        let dzdx = (hr - hl) / (2.0 * dx);
        let dzdy = (hu - hd) / (2.0 * dy);
        glam::Vec3::new(-dzdx, -dzdy, 1.0).normalize()
    }

    /// Derive a physics-ready [`TerrainHeightfield`] from this item's heightmap.
    ///
    /// The height samples come from [`heightmap`](Self::heightmap), so the
    /// collider is always consistent with the rendered surface — there is no
    /// second copy to keep in sync. See [`TerrainHeightfield`] for the buffer
    /// layout and body-placement conventions. Register the result at
    /// [`body_translation`](TerrainHeightfield::body_translation) with
    /// [`body_rotation`](TerrainHeightfield::body_rotation).
    pub fn heightfield(&self) -> TerrainHeightfield {
        let cols = self.dims[0] as usize;
        let rows = self.dims[1] as usize;
        let mut heights = vec![0.0f32; rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                // Reverse rows so source row 0 (world min-y) lands at the grid
                // row that the body rotation maps back to world min-y, and pack
                // column-major (`i + j * rows`).
                let src_row = rows - 1 - i;
                let raw = self.heightmap[src_row * cols + j] as f32 / u16::MAX as f32;
                heights[i + j * rows] = raw;
            }
        }
        let span = self.height_range[1] - self.height_range[0];
        TerrainHeightfield {
            rows: self.dims[1],
            cols: self.dims[0],
            heights,
            scale: glam::Vec3::new(self.world_size[0], span, self.world_size[1]),
            world_extent: glam::Vec2::new(self.world_size[0], self.world_size[1]),
            origin: self.origin,
            height_min: self.height_range[0],
        }
    }
}

/// Per-frame collection of terrains.
pub struct TerrainCollection {
    pub items: Vec<TerrainItem>,
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
            albedo: Colour::linear_rgb(0.1, 0.2, 0.3),
            metallic: 0.5,
            roughness: 0.4,
            height_bias: 0.7,
            textured: true,
            normal_mapped: false,
            normal_scale: 1.0,
            mask_mapped: false,
            mask_remap_min: [0.0, 0.0, 0.0, 0.0],
            mask_remap_max: [1.0, 1.0, 1.0, 1.0],
            uv_scale: [4.0, 4.0],
            uv_offset: [0.0, 0.0],
        };
        item.set_layers_from_descriptors(&[custom]);
        assert_eq!(item.surface_layers[0].albedo, Colour::linear_rgb(0.1, 0.2, 0.3));
        assert!(item.surface_layers[0].textured);
        let default = TerrainLayer::default();
        assert_eq!(item.surface_layers[1].albedo, default.albedo);
        assert!(!item.surface_layers[1].textured);
        assert_eq!(item.surface_layers[7].albedo, default.albedo);
    }

    #[test]
    fn layer_textures_pack_slots_in_order() {
        let slot_len = 2 * 2 * 4;
        let red = vec![255u8; slot_len];
        let green: Vec<u8> = (0..slot_len)
            .map(|i| if i % 4 == 1 { 255 } else { 0 })
            .collect();
        let tex = LayerTextures::new(&[
            TextureData::srgb(2, 2, red.clone()),
            TextureData::srgb(2, 2, green.clone()),
        ])
        .expect("pack");
        assert_eq!(tex.dims(), [2, 2]);
        assert_eq!(tex.colour_space(), ColourSpace::Srgb);
        let packed = tex.rgba();
        assert_eq!(packed.len(), slot_len * LAYER_COUNT);
        assert_eq!(&packed[..slot_len], red.as_slice());
        assert_eq!(&packed[slot_len..slot_len * 2], green.as_slice());
        // Trailing slots are zero-filled.
        assert!(packed[slot_len * 2..].iter().all(|&b| b == 0));
    }

    #[test]
    fn layer_textures_reject_bad_input() {
        let ok = || TextureData::srgb(2, 2, vec![0u8; 16]);
        assert_eq!(LayerTextures::new(&[]).err(), Some(LayerTexturesError::Empty));
        assert_eq!(
            LayerTextures::new(&[ok(), TextureData::srgb(2, 2, vec![0u8; 3])]).err(),
            Some(LayerTexturesError::SizeMismatch { slot: 1 })
        );
        assert_eq!(
            LayerTextures::new(&[ok(), TextureData::srgb(1, 1, vec![0u8; 4])]).err(),
            Some(LayerTexturesError::SizeMismatch { slot: 1 })
        );
        assert_eq!(
            LayerTextures::new(&[ok(), TextureData::linear(2, 2, vec![0u8; 16])]).err(),
            Some(LayerTexturesError::ColourSpaceMismatch { slot: 1 })
        );
        let too_many: Vec<_> = (0..=LAYER_COUNT).map(|_| ok()).collect();
        assert_eq!(
            LayerTextures::new(&too_many).err(),
            Some(LayerTexturesError::TooMany(LAYER_COUNT + 1))
        );
    }
}
