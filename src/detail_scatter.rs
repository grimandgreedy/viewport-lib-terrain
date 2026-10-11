//! Scattering detail layers into this frame's sprites and mesh items. Needs
//! the `gpu` feature, since the output is the renderer's item types.

use viewport_lib::{Colour, Material, MeshId, SceneRenderItem, SpriteBlend, TextureId};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteSizeMode};

use crate::TerrainItem;
use crate::detail::{DetailKind, DetailLayer, DetailScatterParams};

/// Output batches built by [`scatter_terrain_details`].
#[derive(Default)]
pub struct DetailScatterOutput {
    pub sprites: Vec<SpriteItem>,
    /// One opaque scene item per placed mesh. They go in the frame's surface
    /// items, so they write depth and are occluded by the terrain like any
    /// other opaque geometry. The renderer batches items that share a mesh.
    pub meshes: Vec<SceneRenderItem>,
}

impl DetailScatterOutput {
    pub fn clear(&mut self) {
        self.sprites.clear();
        self.meshes.clear();
    }
}

/// Halton sequence value at index `i` with base `b`.
fn halton(mut i: u32, b: u32) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while i > 0 {
        f /= b as f32;
        r += f * (i % b) as f32;
        i /= b;
    }
    r
}

/// Cheap hash from a 2D integer cell to a 0..1 float.
fn hash01(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E3779B1);
    h ^= (y as u32).wrapping_mul(0x85EBCA77);
    h ^= seed.wrapping_mul(0xC2B2AE3D);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846CA68B);
    h ^= h >> 16;
    (h & 0xFFFFFF) as f32 / (1 << 24) as f32
}

fn sample_heightmap(item: &TerrainItem, u: f32, v: f32) -> f32 {
    let (w, h) = (item.dims[0] as f32, item.dims[1] as f32);
    let fx = (u.clamp(0.0, 1.0) * (w - 1.0)).clamp(0.0, w - 1.0001);
    let fy = (v.clamp(0.0, 1.0) * (h - 1.0)).clamp(0.0, h - 1.0001);
    let x0 = fx as usize;
    let y0 = fy as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let stride = item.dims[0] as usize;
    let s = |x: usize, y: usize| -> f32 {
        item.heightmap[y * stride + x] as f32 * (1.0 / 65535.0)
    };
    let h00 = s(x0, y0);
    let h10 = s(x1, y0);
    let h01 = s(x0, y1);
    let h11 = s(x1, y1);
    let hx0 = h00 * (1.0 - tx) + h10 * tx;
    let hx1 = h01 * (1.0 - tx) + h11 * tx;
    let n = hx0 * (1.0 - ty) + hx1 * ty;
    item.height_range[0] + n * (item.height_range[1] - item.height_range[0])
}

fn sample_splat_weight(item: &TerrainItem, u: f32, v: f32, layer: usize) -> f32 {
    if layer >= 8 {
        return 0.0;
    }
    let splat_idx = layer / 4;
    let chan = layer % 4;
    let map = &item.splatmaps[splat_idx];
    let (w, h) = (map.dims[0] as usize, map.dims[1] as usize);
    if w == 0 || h == 0 {
        return 0.0;
    }
    let fx = (u.clamp(0.0, 1.0) * (w as f32 - 1.0)).max(0.0);
    let fy = (v.clamp(0.0, 1.0) * (h as f32 - 1.0)).max(0.0);
    let x = (fx as usize).min(w - 1);
    let y = (fy as usize).min(h - 1);
    let rgba = map.rgba();
    let p = (y * w + x) * 4 + chan;
    if p >= rgba.len() { 0.0 } else { rgba[p] as f32 / 255.0 }
}

/// Build per-frame sprite/mesh batches for the given detail layers.
///
/// One layer can contribute up to two batches: an instanced mesh and a
/// billboard set (the [`DetailKind::TreeLod`] case). The output is
/// reused across frames if the caller passes the same vector back in
/// and calls [`DetailScatterOutput::clear`] each frame.
pub fn scatter_terrain_details(
    item: &TerrainItem,
    layers: &[DetailLayer],
    params: &DetailScatterParams,
    out: &mut DetailScatterOutput,
) {
    out.clear();
    if layers.is_empty() {
        return;
    }
    let origin = item.origin;
    let size = glam::Vec2::new(item.world_size[0], item.world_size[1]);

    for (layer_idx, layer) in layers.iter().enumerate() {
        let area = std::f32::consts::PI * layer.max_distance * layer.max_distance;
        let target = (layer.density * area).clamp(0.0, 200_000.0) as u32;
        if target == 0 {
            continue;
        }

        let mut mesh_transforms: Vec<[[f32; 4]; 4]> = Vec::new();
        let mut sprite_positions: Vec<[f32; 3]> = Vec::new();
        let mut sprite_sizes: Vec<f32> = Vec::new();

        let seed = (layer_idx as u32).wrapping_mul(2654435761);
        let r_max = layer.max_distance;
        let r_max_sq = r_max * r_max;

        for i in 0..target {
            // Halton(2,3) over the camera footprint disc via concentric
            // mapping: radius = sqrt(h2) * r_max, angle = h3 * 2pi.
            let h2 = halton(i + 1, 2);
            let h3 = halton(i + 1, 3);
            let r = r_max * h2.sqrt();
            let theta = h3 * std::f32::consts::TAU;
            let wx = params.camera_xy.x + r * theta.cos();
            let wy = params.camera_xy.y + r * theta.sin();

            // Clamp into terrain bounds.
            let local_x = wx - origin.x;
            let local_y = wy - origin.y;
            if local_x < 0.0 || local_y < 0.0 || local_x > size.x || local_y > size.y {
                continue;
            }
            let u = local_x / size.x;
            let v = local_y / size.y;

            let w_layer = sample_splat_weight(item, u, v, layer.surface_layer);
            if w_layer < layer.weight_threshold {
                continue;
            }
            // Stochastic acceptance proportional to weight above threshold.
            let accept_rng = hash01(i as i32, layer_idx as i32, seed ^ 0xA341316C);
            let denom = (1.0 - layer.weight_threshold).max(1e-4);
            let accept_p = ((w_layer - layer.weight_threshold) / denom).clamp(0.0, 1.0);
            if accept_rng > accept_p {
                continue;
            }

            let h = sample_heightmap(item, u, v);
            let pos = glam::Vec3::new(wx, wy, origin.z + h);

            let s_rng = hash01(i as i32, layer_idx as i32, seed ^ 0xB5297A4D);
            let scale = layer.size * (1.0 + (s_rng * 2.0 - 1.0) * layer.size_jitter);

            match &layer.kind {
                DetailKind::Billboard { .. } => {
                    // Lift the billboard so its base sits on the ground.
                    sprite_positions.push([pos.x, pos.y, pos.z + scale * 0.5]);
                    sprite_sizes.push(scale);
                }
                DetailKind::Mesh { .. } => {
                    let rot_rng = hash01(i as i32, layer_idx as i32, seed ^ 0x68E31DA4);
                    let yaw = rot_rng * std::f32::consts::TAU;
                    let m = glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(scale),
                        glam::Quat::from_rotation_z(yaw),
                        pos,
                    );
                    mesh_transforms.push(m.to_cols_array_2d());
                }
                DetailKind::TreeLod { switch_distance, .. } => {
                    let dx = pos.x - params.camera_xy.x;
                    let dy = pos.y - params.camera_xy.y;
                    let dz = pos.z - params.camera_z;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < switch_distance * switch_distance {
                        let rot_rng = hash01(i as i32, layer_idx as i32, seed ^ 0x68E31DA4);
                        let yaw = rot_rng * std::f32::consts::TAU;
                        let m = glam::Mat4::from_scale_rotation_translation(
                            glam::Vec3::splat(scale),
                            glam::Quat::from_rotation_z(yaw),
                            pos,
                        );
                        mesh_transforms.push(m.to_cols_array_2d());
                    } else {
                        sprite_positions.push([pos.x, pos.y, pos.z + scale * 0.5]);
                        sprite_sizes.push(scale);
                    }
                }
            }

            let _ = r_max_sq;
        }

        let make_sprite = |tex: Option<TextureId>,
                           colour: Colour,
                           positions: Vec<[f32; 3]>,
                           sizes: Vec<f32>|
         -> SpriteItem {
            let mut s = SpriteItem::default();
            s.texture_id = tex;
            s.positions = positions;
            s.sizes = sizes;
            s.default_colour = colour;
            s.default_size = layer.size;
            s.size_mode = SpriteSizeMode::WorldSpace;
            // A textured sprite has transparent texels, which must not write depth.
            s.depth_write = tex.is_none();
            s.blend = SpriteBlend::AlphaBlend;
            s
        };
        let make_meshes = |mesh_id: MeshId,
                           tex: Option<TextureId>,
                           transforms: Vec<[[f32; 4]; 4]>,
                           colour: Colour|
         -> Vec<SceneRenderItem> {
            let mut material = Material::from_colour(colour);
            material.texture_id = tex;
            material.metallic = 0.0;
            material.roughness = 0.9;
            transforms
                .into_iter()
                .map(|model| {
                    let mut item = SceneRenderItem::default();
                    item.mesh_id = mesh_id;
                    item.model = model;
                    item.material = material.clone();
                    item
                })
                .collect()
        };

        match &layer.kind {
            DetailKind::Billboard { texture_id, colour } => {
                if !sprite_positions.is_empty() {
                    out.sprites.push(make_sprite(
                        *texture_id,
                        *colour,
                        sprite_positions,
                        sprite_sizes,
                    ));
                }
            }
            DetailKind::Mesh { mesh_id, texture_id, colour } => {
                if !mesh_transforms.is_empty() {
                    out.meshes.extend(make_meshes(
                        *mesh_id,
                        *texture_id,
                        mesh_transforms,
                        *colour,
                    ));
                }
            }
            DetailKind::TreeLod {
                mesh_id,
                mesh_texture_id,
                impostor_texture_id,
                colour,
                ..
            } => {
                if !mesh_transforms.is_empty() {
                    out.meshes.extend(make_meshes(
                        *mesh_id,
                        *mesh_texture_id,
                        mesh_transforms,
                        *colour,
                    ));
                }
                if !sprite_positions.is_empty() {
                    out.sprites.push(make_sprite(
                        *impostor_texture_id,
                        *colour,
                        sprite_positions,
                        sprite_sizes,
                    ));
                }
            }
        }
    }
}
