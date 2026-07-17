//! Physics-ready heightfield payload derived from a [`TerrainItem`].
//!
//! [`TerrainItem::heightfield`](crate::TerrainItem::heightfield) turns the
//! source heightmap into a [`TerrainHeightfield`] — a physics-engine-neutral
//! description of the terrain as a centred, sampled height grid plus the world
//! transform that places it under the rendered surface. This is the data a
//! collision backend registers so the same heightmap the renderer draws is the
//! one characters walk on, with no second copy loaded from disk.
//!
//! ## Layout
//!
//! [`heights`](TerrainHeightfield::heights) is **column-major** and
//! **row-reversed**, normalised to `[0, 1]`, matching the convention of a
//! centred height grid sampled on the local `xz` plane with height along `y`
//! (the layout `parry3d`'s `HeightField` and engines built on it expect).
//! Feeding a plain row-major buffer to such a reader transposes rows and
//! columns: for a square map the axis-aligned bounds are unchanged, so the
//! broad phase still finds the body, but every sampled height comes from the
//! swapped `xz` position and characters fall through "thin spots". Producing
//! the correct layout in one place keeps every consumer safe from that bug.
//!
//! ## Placement
//!
//! The grid lives on the local `xz` plane centred on the origin; heights run
//! along local `+y`. [`body_translation`](TerrainHeightfield::body_translation)
//! moves that centred grid onto the terrain's world footprint (corner anchor +
//! half-extent, with the height floor folded into `z`), and
//! [`body_rotation`](TerrainHeightfield::body_rotation) maps the grid's local
//! `y`-up frame into the world's `z`-up frame.

/// A physics-ready heightfield extracted from a [`TerrainItem`](crate::TerrainItem).
///
/// Build one with
/// [`TerrainItem::heightfield`](crate::TerrainItem::heightfield). The height
/// samples derive from the item's own heightmap, so the collider can never
/// drift from the rendered surface. See the [module docs](self) for the buffer
/// layout and placement conventions.
#[derive(Debug, Clone)]
pub struct TerrainHeightfield {
    /// Sample count along world `y` (the grid's local `z` / row axis).
    pub rows: u32,
    /// Sample count along world `x` (the grid's local `x` / column axis).
    pub cols: u32,
    /// Column-major, row-reversed heights, length `rows * cols`, normalised to
    /// `[0, 1]`. Multiply by `scale.y` and add the height floor
    /// (`height_range[0]`, already folded into
    /// [`body_translation`](Self::body_translation)) to recover world metres.
    pub heights: Vec<f32>,
    /// `(world_x_extent, height_span, world_y_extent)` on the grid's local
    /// `(x, y, z)`. `scale.y` is the height multiplier applied to the
    /// normalised `heights`.
    pub scale: glam::Vec3,
    /// World-space `xy` extent of the terrain footprint.
    pub world_extent: glam::Vec2,
    /// World-space corner anchor (lower-left in `xy`), matching the rendered
    /// terrain's placement.
    pub origin: glam::Vec3,
    /// World-space height floor (`height_range[0]`). Folded into the body
    /// translation so normalised heights of `0` land at the rendered floor.
    pub height_min: f32,
}

impl TerrainHeightfield {
    /// World-space translation for the heightfield body.
    ///
    /// Centres the grid's local `[-extent/2, +extent/2]` span on the terrain
    /// footprint and lifts it to the height floor so a normalised height of `0`
    /// sits at the rendered surface's minimum.
    pub fn body_translation(&self) -> glam::Vec3 {
        self.origin
            + glam::Vec3::new(
                self.world_extent.x * 0.5,
                self.world_extent.y * 0.5,
                self.height_min,
            )
    }

    /// World-space rotation for the heightfield body: maps the grid's local
    /// `y`-up frame into the world's `z`-up frame.
    pub fn body_rotation() -> glam::Quat {
        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
    }
}

#[cfg(test)]
mod tests {
    use crate::TerrainItem;

    /// A centred height grid sampled on the local `xz` plane stores its data
    /// column-major: `flat[i + j * nrows]` is the sample at (row `i`, col `j`).
    fn sample_at(heights: &[f32], nrows: usize, row: usize, col: usize) -> f32 {
        heights[row + col * nrows]
    }

    #[test]
    fn heightfield_normalises_and_reverses_rows() {
        // 4x4: source row 0 = zeros, source row 3 = max. After row reversal,
        // grid row 0 holds source row 3 (max) and grid row 3 holds source row
        // 0 (zeros).
        let n = 4u32;
        let mut heights = vec![0u16; (n * n) as usize];
        for col in 0..n as usize {
            heights[3 * n as usize + col] = u16::MAX;
        }
        let mut item = TerrainItem::new(heights, [n, n]);
        item.world_size = [3.0, 3.0];
        item.height_range = [0.0, 50.0];
        let hf = item.heightfield();

        assert_eq!(hf.rows, 4);
        assert_eq!(hf.cols, 4);
        assert_eq!(hf.heights.len(), 16);
        for col in 0..hf.cols as usize {
            assert!((sample_at(&hf.heights, hf.rows as usize, 0, col) - 1.0).abs() < 1e-6);
            assert_eq!(sample_at(&hf.heights, hf.rows as usize, 3, col), 0.0);
        }
    }

    #[test]
    fn heightfield_is_column_major() {
        // Single max spike at source (row 1, col 2). After reversal the grid
        // index is (row = n-1-1 = 2, col = 2). Column-major flat index =
        // row + col*nrows = 2 + 2*4 = 10.
        let n = 4u32;
        let mut heights = vec![0u16; (n * n) as usize];
        heights[1 * n as usize + 2] = u16::MAX;
        let mut item = TerrainItem::new(heights, [n, n]);
        item.world_size = [3.0, 3.0];
        item.height_range = [0.0, 1.0];
        let hf = item.heightfield();

        for k in 0..hf.heights.len() {
            let expected = if k == 10 { 1.0 } else { 0.0 };
            assert!(
                (hf.heights[k] - expected).abs() < 1e-6,
                "heights[{k}] = {} expected {expected}",
                hf.heights[k],
            );
        }
        assert!((hf.scale.x - 3.0).abs() < 1e-5);
        assert!((hf.scale.z - 3.0).abs() < 1e-5);
        assert!((hf.scale.y - 1.0).abs() < 1e-5);
        assert_eq!(hf.world_extent, glam::Vec2::new(3.0, 3.0));
    }

    #[test]
    fn body_translation_centres_and_folds_height_floor() {
        let n = 5u32;
        let mut item = TerrainItem::new(vec![0u16; (n * n) as usize], [n, n]);
        item.world_size = [8.0, 8.0];
        // A terrain whose heights start below zero (e.g. an island shelf): the
        // height floor must fold into the body translation's z so normalised 0
        // lands at the rendered minimum, not at world z = 0.
        item.height_range = [-8.0, 64.0];
        item.origin = glam::Vec3::new(-100.0, -200.0, 0.0);
        let hf = item.heightfield();

        let centre = hf.body_translation();
        assert!((centre.x - (-100.0 + 4.0)).abs() < 1e-5);
        assert!((centre.y - (-200.0 + 4.0)).abs() < 1e-5);
        assert!((centre.z - (-8.0)).abs() < 1e-5, "height floor folds into z");
        // Height span drives scale.y, not the raw maximum.
        assert!((hf.scale.y - 72.0).abs() < 1e-5);

        let rot = TerrainHeightfield::body_rotation();
        let up = rot * glam::Vec3::Y;
        assert!((up - glam::Vec3::Z).length() < 1e-5);
    }

    use super::TerrainHeightfield;
}
