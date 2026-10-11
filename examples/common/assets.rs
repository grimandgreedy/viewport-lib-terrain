//! Loads the example assets under `examples/assets`: ground materials, the
//! Mount Rainier heightmap and the Kenney nature models.
//!
//! The ground materials are 1K PBR sets from ambientCG and Poly Haven. Every
//! layer of a terrain shares one texture size, so all eight must be the same
//! resolution. The heightmap is built from USGS 3DEP elevation tiles; its
//! credits are in `examples/assets/CREDITS.md`.

use std::path::{Path, PathBuf};

use viewport_lib::{MeshData, TextureData};
use viewport_lib_terrain::{LAYER_COUNT, LayerTextures, TerrainItem};

/// Root of the asset directory.
pub fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets")
}

// ---------------------------------------------------------------------------
// Heightmap
// ---------------------------------------------------------------------------

/// Metres of real terrain covered by one side of the Rainier heightmap.
pub const RAINIER_SPAN_M: f32 = 26_700.0;
/// Lowest and highest real elevation in the heightmap, in metres.
pub const RAINIER_ELEVATION_M: [f32; 2] = [497.9, 4388.1];

/// The Rainier heightmap at `dim` x `dim` samples. `dim` is 513 or 257; the
/// smaller size is decimated from the 513 file.
///
/// Samples are normalised to the full `u16` range, so the world height of a
/// sample is `height_range[0] + raw / 65535 * (height_range[1] - height_range[0])`.
pub fn rainier_heightmap(dim: u32) -> Vec<u16> {
    const FILE_DIM: usize = 513;
    assert!(dim == 513 || dim == 257, "no Rainier heightmap at {dim}x{dim}; use 513 or 257");
    let path = dir().join("heightmaps/rainier_513.r16");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let full: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(full.len(), FILE_DIM * FILE_DIM, "unexpected size for rainier_513.r16");
    if dim as usize == FILE_DIM {
        return full;
    }
    let step = (FILE_DIM - 1) / (dim as usize - 1);
    let mut out = Vec::with_capacity((dim * dim) as usize);
    for y in 0..dim as usize {
        for x in 0..dim as usize {
            out.push(full[y * step * FILE_DIM + x * step]);
        }
    }
    out
}

/// Height in world units for a heightmap spanning `world_size` metres, with the
/// vertical scale of the real terrain times `exaggeration`.
pub fn rainier_height_range(world_size: f32, exaggeration: f32) -> [f32; 2] {
    let scale = world_size / RAINIER_SPAN_M * exaggeration;
    [0.0, (RAINIER_ELEVATION_M[1] - RAINIER_ELEVATION_M[0]) * scale]
}

// ---------------------------------------------------------------------------
// Ground materials
// ---------------------------------------------------------------------------

enum Source {
    AmbientCg,
    PolyHaven,
}

/// One surface layer's material.
pub struct Ground {
    dir: &'static str,
    source: Source,
    /// World metres covered by one tile.
    pub tile_metres: f32,
    /// How strongly the normal map perturbs the surface.
    pub normal_scale: f32,
}

/// The eight layers, in splatmap channel order: A.r, A.g, A.b, A.a, B.r, B.g,
/// B.b, B.a.
pub const GROUND: [Ground; LAYER_COUNT] = [
    // 0 grass
    Ground { dir: "Grass004", source: Source::AmbientCg, tile_metres: 3.0, normal_scale: 1.0 },
    // 1 dirt
    Ground { dir: "Ground103", source: Source::AmbientCg, tile_metres: 4.0, normal_scale: 1.0 },
    // 2 rock
    Ground { dir: "Rock051", source: Source::AmbientCg, tile_metres: 6.0, normal_scale: 1.2 },
    // 3 sand
    Ground { dir: "coast_sand_01", source: Source::PolyHaven, tile_metres: 4.0, normal_scale: 0.8 },
    // 4 pebbles
    Ground { dir: "Gravel041", source: Source::AmbientCg, tile_metres: 2.0, normal_scale: 1.0 },
    // 5 moss
    Ground { dir: "Ground037", source: Source::AmbientCg, tile_metres: 4.0, normal_scale: 1.0 },
    // 6 snow
    Ground { dir: "Snow014", source: Source::AmbientCg, tile_metres: 4.0, normal_scale: 0.8 },
    // 7 dark rock
    Ground { dir: "Rock030", source: Source::AmbientCg, tile_metres: 6.0, normal_scale: 1.2 },
];

fn decode_jpeg(path: &Path) -> viewport_lib_io::RasterImageData {
    viewport_lib_io::loaders::jpeg::texture_from_path(path)
        .unwrap_or_else(|e| panic!("decode {}: {e:?}", path.display()))
}

/// The diffuse, normal and mask arrays for all eight layers.
pub struct GroundTextures {
    pub diffuse: LayerTextures,
    pub normal: LayerTextures,
    pub mask: LayerTextures,
}

/// Decode the eight ground materials.
///
/// The mask array packs R metallic, G ambient occlusion, B detail and A
/// smoothness, which is what the terrain shader reads. The sources carry
/// roughness, so smoothness is its inverse.
pub fn load_ground() -> GroundTextures {
    let root = dir().join("textures");
    let mut diffuse = Vec::new();
    let mut normal = Vec::new();
    let mut mask = Vec::new();
    let mut size = [0u32; 2];

    for g in GROUND.iter() {
        let d = root.join(g.dir);
        let (colour, nor, ao, rough, metal) = match g.source {
            Source::AmbientCg => {
                let f = |suffix: &str| d.join(format!("{}_1K-JPG_{suffix}.jpg", g.dir));
                (
                    decode_jpeg(&f("Color")),
                    decode_jpeg(&f("NormalGL")),
                    Some(decode_jpeg(&f("AmbientOcclusion"))),
                    decode_jpeg(&f("Roughness")),
                    None,
                )
            }
            Source::PolyHaven => {
                let f = |suffix: &str| d.join(format!("{}_{suffix}_1k.jpg", g.dir));
                // The packed map is R ambient occlusion, G roughness, B metallic.
                let arm = decode_jpeg(&f("arm"));
                (decode_jpeg(&f("diff")), decode_jpeg(&f("nor_gl")), None, arm.clone(), Some(arm))
            }
        };
        if size == [0, 0] {
            size = [colour.width, colour.height];
        }
        assert_eq!(
            [colour.width, colour.height],
            size,
            "{} is not the same size as the other layers",
            g.dir
        );

        let texels = (size[0] * size[1]) as usize;
        let mut m = vec![0u8; texels * 4];
        for i in 0..texels {
            let (occlusion, smoothness, metallic) = match g.source {
                Source::AmbientCg => (
                    ao.as_ref().map_or(255, |a| a.rgba[i * 4]),
                    255 - rough.rgba[i * 4],
                    0,
                ),
                Source::PolyHaven => {
                    let arm = metal.as_ref().unwrap();
                    (arm.rgba[i * 4], 255 - rough.rgba[i * 4 + 1], arm.rgba[i * 4 + 2])
                }
            };
            m[i * 4] = metallic;
            m[i * 4 + 1] = occlusion;
            m[i * 4 + 2] = 255;
            m[i * 4 + 3] = smoothness;
        }

        diffuse.push(colour.rgba);
        normal.push(nor.rgba);
        mask.push(m);
    }

    let pack = |v: &[Vec<u8>]| LayerTextures::new(v, size).expect("layer textures");
    GroundTextures { diffuse: pack(&diffuse), normal: pack(&normal), mask: pack(&mask) }
}

/// The decoded ground, shared by every terrain in the process. The arrays sit
/// behind `Arc`, so handing them to each item costs nothing.
pub fn ground() -> &'static GroundTextures {
    static GROUND_TEXTURES: std::sync::OnceLock<GroundTextures> = std::sync::OnceLock::new();
    GROUND_TEXTURES.get_or_init(load_ground)
}

/// Give `terrain` the textured ground: the three arrays, and every layer set to
/// sample them with its own tiling.
pub fn apply_ground(terrain: &mut TerrainItem) {
    let ground = ground();
    terrain.layer_textures = Some(ground.diffuse.clone());
    terrain.normal_textures = Some(ground.normal.clone());
    terrain.mask_textures = Some(ground.mask.clone());
    for (layer, g) in terrain.surface_layers.iter_mut().zip(GROUND.iter()) {
        layer.textured = true;
        layer.normal_mapped = true;
        layer.normal_scale = g.normal_scale;
        layer.mask_mapped = true;
        layer.uv_scale = [g.tile_metres, g.tile_metres];
    }
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// A model flattened into one mesh, with its base colour texture if it has one.
pub struct Model {
    pub mesh: MeshData,
    pub texture: Option<TextureData>,
}

/// The kit is stylised teal and orange, which clashes with photographic ground.
/// Wood and foliage materials get natural colours instead; anything else keeps
/// the colour it was authored with.
fn natural_colour(material: &str) -> Option<[f32; 3]> {
    let name = material.to_ascii_lowercase();
    let srgb = if ["wood", "bark", "trunk"].iter().any(|k| name.contains(k)) {
        [0.30, 0.21, 0.13]
    } else if ["leaf", "leaves", "grass", "plant", "bush"].iter().any(|k| name.contains(k)) {
        [0.20, 0.36, 0.13]
    } else {
        return None;
    };
    Some(viewport_lib::Colour::srgb_rgb(srgb[0], srgb[1], srgb[2]).to_linear_rgb())
}

/// Load a Kenney nature kit model, e.g. `"tree_pineTallA"`, and merge its parts
/// into one mesh.
pub fn load_kenney_model(name: &str) -> Model {
    let path = dir()
        .join("models/kenney_nature_kit/Models/GLTF format")
        .join(format!("{name}.glb"));
    let scene = viewport_lib_io::loaders::gltf::scene_from_path(&path)
        .unwrap_or_else(|e| panic!("load {}: {e:?}", path.display()));

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut colours: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut any_uvs = false;
    let mut any_colours = false;
    let mut texture = None;

    for part in &scene.meshes {
        let base = positions.len() as u32;
        let n = part.mesh.positions.len();
        let m = part.transform;
        for p in &part.mesh.positions {
            positions.push(m.transform_point3(glam::Vec3::from(*p)).to_array());
        }
        for v in &part.mesh.normals {
            normals.push(m.transform_vector3(glam::Vec3::from(*v)).normalize_or_zero().to_array());
        }
        match &part.mesh.uvs {
            Some(u) => {
                any_uvs = true;
                uvs.extend_from_slice(u);
            }
            None => uvs.extend(std::iter::repeat([0.0, 0.0]).take(n)),
        }
        match &part.mesh.colours {
            Some(c) => {
                any_colours = true;
                colours.extend_from_slice(c);
            }
            None => {
                // The Kenney models carry a flat colour per material and no
                // texture, so the colour goes into the vertices.
                let base = part
                    .material_index
                    .and_then(|i| scene.materials.get(i))
                    .map(|mat| natural_colour(&mat.name).unwrap_or(mat.base_color));
                if let Some(c) = base {
                    any_colours = true;
                    colours.extend(std::iter::repeat([c[0], c[1], c[2], 1.0]).take(n));
                } else {
                    colours.extend(std::iter::repeat([1.0; 4]).take(n));
                }
            }
        }
        indices.extend(part.mesh.indices.iter().map(|i| i + base));

        if texture.is_none() {
            let tex = part
                .material_index
                .and_then(|i| scene.materials.get(i))
                .and_then(|mat| mat.base_color_texture.as_ref());
            if let Some(viewport_lib_io::TextureSource::Decoded(img)) = tex {
                texture = Some(TextureData::srgb(img.width, img.height, img.rgba.clone()));
            }
        }
    }

    let mut mesh = MeshData::new(positions, normals, indices);
    if any_uvs {
        mesh.uvs = Some(uvs);
    }
    if any_colours {
        mesh.vertex_colours = Some(colours);
    }
    Model { mesh, texture }
}

/// Scale a model uniformly so it is `height` tall, with its lowest point on z = 0
/// and centred on the origin in x and y. A detail layer's `size` then reads as
/// the tree's height in metres.
pub fn scale_to_height(model: &mut Model, height: f32) {
    let aabb = model.mesh.compute_aabb();
    let (min, max) = (aabb.min, aabb.max);
    let scale = height / (max.z - min.z).max(1e-6);
    let cx = 0.5 * (min.x + max.x);
    let cy = 0.5 * (min.y + max.y);
    for p in &mut model.mesh.positions {
        p[0] = (p[0] - cx) * scale;
        p[1] = (p[1] - cy) * scale;
        p[2] = (p[2] - min.z) * scale;
    }
}

/// A pine silhouette for the distant impostor of a tree that is one unit tall
/// and about a quarter as wide: four stacked cones on a trunk, on a transparent
/// background. The quad is square, so the tree fills the height and the centre of
/// the width.
pub fn pine_impostor() -> TextureData {
    const N: usize = 256;
    let leaf = [51.0f32, 92.0, 33.0];
    let bark = [77.0f32, 54.0, 33.0];
    let mut rgba = vec![0u8; N * N * 4];
    let cover = |dist: f32| (dist + 0.5).clamp(0.0, 1.0);
    for y in 0..N {
        for x in 0..N {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let dx = fx - N as f32 * 0.5;
            let mut alpha = 0.0f32;
            let mut colour = leaf;

            // Trunk: the lowest 12 percent of the height.
            if fy > N as f32 * 0.88 {
                let a = cover(7.0 - dx.abs());
                if a > alpha {
                    alpha = a;
                    colour = bark;
                }
            }
            // Four tiers, the widest at the bottom. Each is a triangle that
            // overlaps the one above it.
            for tier in 0..4 {
                let base = N as f32 * (0.90 - 0.19 * tier as f32);
                let apex = base - N as f32 * 0.34;
                let half_base = 34.0 - 6.0 * tier as f32;
                if fy >= apex && fy <= base {
                    let half = half_base * (fy - apex) / (base - apex);
                    let a = cover(half - dx.abs()).min(cover(fy - apex)).min(cover(base - fy));
                    if a > alpha {
                        alpha = a;
                        // Light on the left, shade on the right, darker low in
                        // each tier.
                        let shade = 1.15 - 0.45 * ((dx / half_base) * 0.5 + 0.5)
                            - 0.15 * (fy - apex) / (base - apex);
                        colour = [leaf[0] * shade, leaf[1] * shade, leaf[2] * shade];
                    }
                }
            }

            // Sprites sample the texture with row 0 at the bottom of the quad, so
            // the image is written upside down.
            let i = ((N - 1 - y) * N + x) * 4;
            // Colour under transparent texels stays leaf green so filtering does
            // not pull a dark fringe in at the edges.
            let c = if alpha > 0.0 { colour } else { leaf };
            rgba[i] = c[0].clamp(0.0, 255.0) as u8;
            rgba[i + 1] = c[1].clamp(0.0, 255.0) as u8;
            rgba[i + 2] = c[2].clamp(0.0, 255.0) as u8;
            rgba[i + 3] = (alpha * 255.0) as u8;
        }
    }
    TextureData::srgb(N as u32, N as u32, rgba)
}
