//! Heightmap -> patch mesh builder.
//!
//! Each terrain is subdivided into a regular grid of patches; each
//! patch is baked at multiple LOD levels (decimation factors 1x, 2x,
//! 4x, ...) and the renderer picks the appropriate LOD per frame based
//! on camera distance.
//!
//! Patches include a skirt: a ring of duplicated perimeter vertices
//! dropped along -Z. The skirt fills cracks where a high-LOD patch
//! meets a low-LOD neighbour with a different vertex along their
//! shared edge.

use bytemuck::{Pod, Zeroable};

use crate::item::TerrainItem;

/// Interleaved vertex format: 8 floats = 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct TerrainVertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub uv:       [f32; 2],
}

/// Vertex buffer layout matching [`TerrainVertex`].
pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    static ATTRS: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 12,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 24,
            shader_location: 2,
        },
    ];
    wgpu::VertexBufferLayout {
        array_stride: 32,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// Number of vertices and indices in one patch at one LOD, used to
/// size buffers and report stats.
pub struct PatchMesh {
    pub vertices: Vec<TerrainVertex>,
    pub indices: Vec<u32>,
    pub aabb_min: glam::Vec3,
    pub aabb_max: glam::Vec3,
}

/// Build one LOD of one patch.
///
/// `patch_origin_cells` is the patch's lower-left corner in heightmap
/// cell coordinates (not world). `patch_cells` is the patch size at
/// LOD 0 in cells per side; the actual grid built at this LOD has
/// `patch_cells >> lod` cells per side. Adjacent patches overlap by
/// one row/column of heightmap samples so the base meshes are
/// watertight when neighbours pick the same LOD.
pub fn build_patch(
    item: &TerrainItem,
    patch_origin_cells: [u32; 2],
    patch_cells: u32,
    lod: u32,
    skirt_depth: f32,
) -> PatchMesh {
    let hw = item.dims[0] as i32;
    let hh = item.dims[1] as i32;
    debug_assert!(hw >= 2 && hh >= 2);

    let dx = item.world_size[0] / (item.dims[0] - 1) as f32;
    let dy = item.world_size[1] / (item.dims[1] - 1) as f32;
    let (h_min, h_max) = (item.height_range[0], item.height_range[1]);
    let h_span = h_max - h_min;

    let sample = |x: i32, y: i32| -> f32 {
        let cx = x.clamp(0, hw - 1) as usize;
        let cy = y.clamp(0, hh - 1) as usize;
        let raw = item.heightmap[cy * (hw as usize) + cx] as f32 / u16::MAX as f32;
        h_min + raw * h_span
    };

    let step = 1u32 << lod.max(0).min(8); // 1, 2, 4, 8, ...
    let cells = (patch_cells / step).max(1);
    let ox = patch_origin_cells[0] as i32;
    let oy = patch_origin_cells[1] as i32;

    let mut vertices = Vec::with_capacity(((cells + 1) * (cells + 1)) as usize + 64);
    let mut z_lo = f32::INFINITY;
    let mut z_hi = f32::NEG_INFINITY;
    let world_origin_x = item.origin.x;
    let world_origin_y = item.origin.y;
    let world_origin_z = item.origin.z;
    let world_w = item.world_size[0];
    let world_h = item.world_size[1];

    // Build the main grid vertices.
    for gy in 0..=cells {
        for gx in 0..=cells {
            let cell_x = ox + (gx * step) as i32;
            let cell_y = oy + (gy * step) as i32;
            let world_x = world_origin_x + cell_x as f32 * dx;
            let world_y = world_origin_y + cell_y as f32 * dy;
            let world_z = world_origin_z + sample(cell_x, cell_y);
            z_lo = z_lo.min(world_z);
            z_hi = z_hi.max(world_z);

            // Central differences in cell coordinates at this LOD's
            // step. Wider stencil at lower LOD smooths normals.
            let dzdx =
                (sample(cell_x + step as i32, cell_y) - sample(cell_x - step as i32, cell_y))
                    / (2.0 * step as f32 * dx);
            let dzdy =
                (sample(cell_x, cell_y + step as i32) - sample(cell_x, cell_y - step as i32))
                    / (2.0 * step as f32 * dy);
            let n = glam::Vec3::new(-dzdx, -dzdy, 1.0).normalize();

            let u = (world_x - world_origin_x) / world_w;
            let v = (world_y - world_origin_y) / world_h;
            vertices.push(TerrainVertex {
                position: [world_x, world_y, world_z],
                normal: n.to_array(),
                uv: [u, v],
            });
        }
    }

    let main_idx = |x: u32, y: u32| -> u32 { y * (cells + 1) + x };

    let mut indices = Vec::with_capacity((cells * cells * 6) as usize + 64 * 6);
    for y in 0..cells {
        for x in 0..cells {
            let i00 = main_idx(x, y);
            let i10 = main_idx(x + 1, y);
            let i01 = main_idx(x, y + 1);
            let i11 = main_idx(x + 1, y + 1);
            indices.push(i00);
            indices.push(i10);
            indices.push(i11);
            indices.push(i00);
            indices.push(i11);
            indices.push(i01);
        }
    }

    // Skirt: ring of perimeter vertices duplicated and dropped by
    // skirt_depth, triangulated as a strip back to the main perimeter.
    if skirt_depth > 0.0 {
        let perimeter: Vec<u32> = collect_perimeter(cells, &main_idx);
        let drop_base = vertices.len() as u32;
        for &top_idx in &perimeter {
            let top = vertices[top_idx as usize];
            let mut bot = top;
            bot.position[2] -= skirt_depth;
            // Keep the normal pointing the same way as the top so the
            // skirt does not look like an inverted wall when lit.
            vertices.push(bot);
        }
        // Build the skirt strip. perimeter[i] -> perimeter[i+1] (top
        // edge) closes with drop_base+i -> drop_base+i+1 (bottom edge).
        let m = perimeter.len() as u32;
        for i in 0..m {
            let i_next = (i + 1) % m;
            let t0 = perimeter[i as usize];
            let t1 = perimeter[i_next as usize];
            let b0 = drop_base + i;
            let b1 = drop_base + i_next;
            // Two triangles per quad. Orientation chosen so the
            // outward face points the same way as the patch top
            // (the two-sided fragment shader keeps both sides lit).
            indices.push(t0);
            indices.push(b0);
            indices.push(t1);
            indices.push(t1);
            indices.push(b0);
            indices.push(b1);
        }
        z_lo -= skirt_depth;
    }

    let aabb_min = glam::Vec3::new(
        world_origin_x + ox as f32 * dx,
        world_origin_y + oy as f32 * dy,
        z_lo,
    );
    let aabb_max = glam::Vec3::new(
        world_origin_x + (ox + (patch_cells as i32)) as f32 * dx,
        world_origin_y + (oy + (patch_cells as i32)) as f32 * dy,
        z_hi,
    );
    PatchMesh {
        vertices,
        indices,
        aabb_min,
        aabb_max,
    }
}

/// Walk the perimeter of the (cells+1) x (cells+1) grid clockwise
/// from (0, 0) and return the vertex indices.
fn collect_perimeter(cells: u32, idx: &impl Fn(u32, u32) -> u32) -> Vec<u32> {
    let mut out = Vec::with_capacity((cells * 4) as usize);
    for x in 0..cells {
        out.push(idx(x, 0));
    }
    for y in 0..cells {
        out.push(idx(cells, y));
    }
    for x in (1..=cells).rev() {
        out.push(idx(x, cells));
    }
    for y in (1..=cells).rev() {
        out.push(idx(0, y));
    }
    out
}
