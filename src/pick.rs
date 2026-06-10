//! CPU ray-against-heightmap intersection.
//!
//! Walks the 2D grid in XY using a digital differential analyser, then
//! tests the ray against the two triangles of each cell using
//! ray-triangle intersection. Returns the first hit along the ray.

use viewport_lib::interaction::picking::PickHit;
use viewport_lib::plugin_api::PickRay;
use viewport_lib::renderer::PickId;

use crate::plugin::CpuHeightmap;

pub fn pick_heightmap(
    ray: &PickRay,
    cpu: &CpuHeightmap,
    pick_id: PickId,
) -> Option<(f32, PickHit)> {
    if pick_id == PickId::NONE {
        return None;
    }
    let w = cpu.dims[0] as usize;
    let h = cpu.dims[1] as usize;
    if w < 2 || h < 2 {
        return None;
    }

    let dir = ray.direction.normalize();
    let t_entry = ray_aabb_entry(ray.origin, dir, cpu)?;
    let entry_world = ray.origin + dir * t_entry;

    let dx = cpu.world_size[0] / (w - 1) as f32;
    let dy = cpu.world_size[1] / (h - 1) as f32;

    let to_cell = |p: glam::Vec3| -> (i32, i32) {
        let cx = ((p.x - cpu.origin.x) / dx).floor() as i32;
        let cy = ((p.y - cpu.origin.y) / dy).floor() as i32;
        (cx, cy)
    };

    let (mut cx, mut cy) = to_cell(entry_world);

    let step_x = if dir.x > 0.0 { 1 } else { -1 };
    let step_y = if dir.y > 0.0 { 1 } else { -1 };

    let next_boundary = |cell: i32, step: i32, base: f32, cell_size: f32| -> f32 {
        let edge = if step > 0 { (cell + 1) as f32 } else { cell as f32 };
        base + edge * cell_size
    };
    let mut t_max_x = if dir.x.abs() < 1e-9 {
        f32::INFINITY
    } else {
        let edge_world = next_boundary(cx, step_x, cpu.origin.x, dx);
        (edge_world - ray.origin.x) / dir.x
    };
    let mut t_max_y = if dir.y.abs() < 1e-9 {
        f32::INFINITY
    } else {
        let edge_world = next_boundary(cy, step_y, cpu.origin.y, dy);
        (edge_world - ray.origin.y) / dir.y
    };
    let t_delta_x = if dir.x.abs() < 1e-9 {
        f32::INFINITY
    } else {
        dx / dir.x.abs()
    };
    let t_delta_y = if dir.y.abs() < 1e-9 {
        f32::INFINITY
    } else {
        dy / dir.y.abs()
    };

    let max_steps = (w + h) * 2;
    for _ in 0..max_steps {
        if cx >= 0 && cy >= 0 && (cx as usize) < w - 1 && (cy as usize) < h - 1 {
            if let Some(hit) = intersect_cell(
                ray.origin,
                dir,
                cx as usize,
                cy as usize,
                cpu,
                dx,
                dy,
            ) {
                let normal = cell_normal(cx as usize, cy as usize, cpu, dx, dy);
                let world_pos = ray.origin + dir * hit;
                return Some((
                    hit,
                    PickHit::object_hit(pick_id.0, world_pos, normal),
                ));
            }
        }
        if t_max_x < t_max_y {
            cx += step_x;
            t_max_x += t_delta_x;
            if cx < 0 || cx as usize >= w - 1 {
                break;
            }
        } else {
            cy += step_y;
            t_max_y += t_delta_y;
            if cy < 0 || cy as usize >= h - 1 {
                break;
            }
        }
    }
    None
}

/// Slab-test against the XY footprint extended over the full Z range.
/// Returns the entry `t` of the ray into the terrain's world AABB, or 0
/// when the origin is already inside.
fn ray_aabb_entry(origin: glam::Vec3, dir: glam::Vec3, cpu: &CpuHeightmap) -> Option<f32> {
    let lo = glam::Vec3::new(cpu.origin.x, cpu.origin.y, f32::NEG_INFINITY);
    let hi = glam::Vec3::new(
        cpu.origin.x + cpu.world_size[0],
        cpu.origin.y + cpu.world_size[1],
        f32::INFINITY,
    );
    let mut t_min: f32 = 0.0;
    let mut t_max: f32 = f32::INFINITY;
    for axis in 0..2 {
        let o = origin[axis];
        let d = dir[axis];
        if d.abs() < 1e-9 {
            if o < lo[axis] || o > hi[axis] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d;
        let mut t1 = (lo[axis] - o) * inv;
        let mut t2 = (hi[axis] - o) * inv;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        t_min = t_min.max(t1);
        t_max = t_max.min(t2);
        if t_min > t_max {
            return None;
        }
    }
    Some(t_min.max(0.0))
}

fn cell_corners(cx: usize, cy: usize, cpu: &CpuHeightmap, dx: f32, dy: f32) -> [glam::Vec3; 4] {
    let w = cpu.dims[0] as usize;
    let i00 = cy * w + cx;
    let i10 = cy * w + (cx + 1);
    let i01 = (cy + 1) * w + cx;
    let i11 = (cy + 1) * w + (cx + 1);
    let x0 = cpu.origin.x + cx as f32 * dx;
    let x1 = x0 + dx;
    let y0 = cpu.origin.y + cy as f32 * dy;
    let y1 = y0 + dy;
    [
        glam::Vec3::new(x0, y0, cpu.heights[i00]),
        glam::Vec3::new(x1, y0, cpu.heights[i10]),
        glam::Vec3::new(x0, y1, cpu.heights[i01]),
        glam::Vec3::new(x1, y1, cpu.heights[i11]),
    ]
}

fn intersect_cell(
    origin: glam::Vec3,
    dir: glam::Vec3,
    cx: usize,
    cy: usize,
    cpu: &CpuHeightmap,
    dx: f32,
    dy: f32,
) -> Option<f32> {
    let c = cell_corners(cx, cy, cpu, dx, dy);
    // Match the triangulation used in mesh.rs:
    //   tri A: (c00, c10, c11)
    //   tri B: (c00, c11, c01)
    let mut best: Option<f32> = None;
    if let Some(t) = ray_triangle(origin, dir, c[0], c[1], c[3]) {
        if t > 0.0 {
            best = Some(t);
        }
    }
    if let Some(t) = ray_triangle(origin, dir, c[0], c[3], c[2]) {
        if t > 0.0 {
            best = match best {
                Some(b) if b < t => Some(b),
                _ => Some(t),
            };
        }
    }
    best
}

fn cell_normal(cx: usize, cy: usize, cpu: &CpuHeightmap, dx: f32, dy: f32) -> glam::Vec3 {
    let c = cell_corners(cx, cy, cpu, dx, dy);
    let e1 = c[1] - c[0];
    let e2 = c[3] - c[0];
    e1.cross(e2).normalize_or_zero()
}

/// Moller-Trumbore ray-triangle intersection. Returns `t` along the
/// ray (so the world hit is `origin + dir * t`); positive only.
fn ray_triangle(
    origin: glam::Vec3,
    dir: glam::Vec3,
    a: glam::Vec3,
    b: glam::Vec3,
    c: glam::Vec3,
) -> Option<f32> {
    let e1 = b - a;
    let e2 = c - a;
    let h = dir.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let s = origin - a;
    let u = inv * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = inv * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = inv * e2.dot(q);
    if t > 0.0 { Some(t) } else { None }
}
