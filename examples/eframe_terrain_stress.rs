//! Multi-item terrain stress test.
//!
//! Tiles a configurable NxN grid of `TerrainItem`s next to each other so the
//! plugin has to bake, frustum-cull, and draw hundreds to thousands of patches
//! per frame. Built to exercise the path that becomes interesting once the
//! plugin starts driving GPU cull through the public cull service: many small
//! batches, mixed LOD per patch, and a camera that can quickly bring most of
//! them on or off screen.
//!
//! Use the side panel to scale the grid size, the per-item patch resolution,
//! and the LOD distance. The HUD prints the resulting patch totals so the
//! cull cost can be eyeballed alongside the frame time.

use eframe::egui;
use std::sync::Arc;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, ItemSettings, LightingSettings,
    OrbitCameraController, PickId, SceneFrame, ScrollUnits, ViewportContext, ViewportEvent,
    ViewportRenderer,
};
use viewport_lib_terrain::{
    LAYER_COUNT, SplatmapData, TerrainCollection, TerrainItem, TerrainLayer, TerrainPlugin,
};

const ITEM_WORLD_SIZE: f32 = 64.0;
const HEIGHT_MIN: f32 = -2.0;
const HEIGHT_MAX: f32 = 14.0;

/// Available per-item heightmap resolutions. Larger values produce more
/// patches per item at the same `patch_cells` and dominate the total cull
/// workload faster than growing the grid.
const ITEM_DIM_CHOICES: &[u32] = &[129, 257, 513, 1025];
const DEFAULT_ITEM_DIM: u32 = 257;

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib-terrain : stress",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]),
            depth_buffer: 24,
            stencil_buffer: 8,
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let device = &rs.device;
            let format = rs.target_format;

            let mut renderer = ViewportRenderer::new(device, format);
            let plugin = TerrainPlugin::new()
                .with_patch_cells(DEFAULT_PATCH_CELLS)
                .with_max_lod(DEFAULT_MAX_LOD)
                .with_skirt_depth(1.5)
                .with_lod_distance(DEFAULT_LOD_DISTANCE);
            renderer.with_item_type_plugin(device, Box::new(plugin));

            rs.renderer.write().callback_resources.insert(renderer);
            Ok(Box::new(App::new()))
        }),
    )
}

const DEFAULT_GRID: u32 = 3;
const DEFAULT_PATCH_CELLS: u32 = 32;
const DEFAULT_MAX_LOD: u32 = 4;
const DEFAULT_LOD_DISTANCE: f32 = 64.0;

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    items: Vec<TerrainItem>,
    /// Grid edge length. The scene holds `grid * grid` items.
    grid: u32,
    /// Patch resolution used the next time the grid is rebuilt.
    pending_patch_cells: u32,
    /// LOD count used the next time the grid is rebuilt.
    pending_max_lod: u32,
    /// LOD distance applied to the plugin the next time the grid is rebuilt.
    pending_lod_distance: f32,
    /// Per-item heightmap resolution applied on the next rebuild. Larger
    /// values multiply patch counts faster than growing the grid.
    pending_item_dim: u32,
    /// Currently applied values, mirrored from the plugin construction.
    patch_cells: u32,
    max_lod: u32,
    lod_distance: f32,
    /// Per-item heightmap resolution backing the live `items`.
    item_dim: u32,
    /// Shared default layer set; cheap to clone into every item.
    layers: [TerrainLayer; LAYER_COUNT],
    /// Cached non-empty splatmap so every item points at the same bytes.
    splat_a: SplatmapData,
    splat_b: SplatmapData,

    last_eye: glam::Vec3,
    last_pixels_per_point: f32,
    cursor_viewport: Option<glam::Vec2>,
    left_pressed_this_frame: bool,
    drag_in_viewport: bool,
    /// Frame time exponential moving average, milliseconds. Smoothed so the
    /// HUD does not flicker when individual frames spike.
    frame_ms_ema: f32,
    last_frame: std::time::Instant,
    /// When `true` the camera slowly orbits so the cull workload is
    /// constantly shuffling. Useful for steady measurement.
    auto_orbit: bool,
}

impl App {
    fn new() -> Self {
        let layers = default_layers();
        let splat_a = SplatmapData::solid_layer0();
        let splat_b = SplatmapData::empty();
        let items = build_grid(
            DEFAULT_GRID,
            DEFAULT_ITEM_DIM,
            &layers,
            &splat_a,
            &splat_b,
            /* pick_id_base */ 1,
        );
        Self {
            camera: Camera {
                distance: ITEM_WORLD_SIZE * DEFAULT_GRID as f32 * 1.3,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            items,
            grid: DEFAULT_GRID,
            pending_patch_cells: DEFAULT_PATCH_CELLS,
            pending_max_lod: DEFAULT_MAX_LOD,
            pending_lod_distance: DEFAULT_LOD_DISTANCE,
            pending_item_dim: DEFAULT_ITEM_DIM,
            patch_cells: DEFAULT_PATCH_CELLS,
            max_lod: DEFAULT_MAX_LOD,
            lod_distance: DEFAULT_LOD_DISTANCE,
            item_dim: DEFAULT_ITEM_DIM,
            layers,
            splat_a,
            splat_b,
            last_eye: glam::Vec3::ZERO,
            last_pixels_per_point: 1.0,
            cursor_viewport: None,
            left_pressed_this_frame: false,
            drag_in_viewport: false,
            frame_ms_ema: 16.0,
            last_frame: std::time::Instant::now(),
            auto_orbit: false,
        }
    }

    /// Number of patches per side per item, given the current `patch_cells`.
    fn patches_per_side(&self) -> u32 {
        // The plugin clips the heightmap so an integer number of patches fits.
        ((self.item_dim - 1) / self.patch_cells).max(1)
    }

    fn total_items(&self) -> u32 {
        self.grid * self.grid
    }

    fn total_patches(&self) -> u32 {
        let pps = self.patches_per_side();
        self.total_items() * pps * pps
    }
}

fn default_layers() -> [TerrainLayer; LAYER_COUNT] {
    let mut layers = [TerrainLayer::default(); LAYER_COUNT];
    layers[0] = TerrainLayer {
        albedo: [0.32, 0.46, 0.22],
        metallic: 0.0,
        roughness: 0.90,
        height_bias: 0.0,
    };
    layers
}

/// Construct `grid * grid` terrain items laid out on a centred XY grid. Each
/// item gets a unique `PickId` starting from `pick_id_base` so the plugin
/// keeps them in distinct baked entries.
fn build_grid(
    grid: u32,
    dim: u32,
    layers: &[TerrainLayer; LAYER_COUNT],
    splat_a: &SplatmapData,
    splat_b: &SplatmapData,
    pick_id_base: u64,
) -> Vec<TerrainItem> {
    let mut out = Vec::with_capacity((grid * grid) as usize);
    let total_world = ITEM_WORLD_SIZE * grid as f32;
    let half = total_world * 0.5;
    let mut pick_id: u64 = pick_id_base;
    for gy in 0..grid {
        for gx in 0..grid {
            let heightmap = build_heightmap(dim as usize, gx, gy);
            let mut item = TerrainItem::new(heightmap, [dim, dim]);
            item.world_size = [ITEM_WORLD_SIZE, ITEM_WORLD_SIZE];
            item.height_range = [HEIGHT_MIN, HEIGHT_MAX];
            item.origin = glam::Vec3::new(
                -half + gx as f32 * ITEM_WORLD_SIZE,
                -half + gy as f32 * ITEM_WORLD_SIZE,
                0.0,
            );
            item.surface_layers = *layers;
            // Re-use the same splatmap bytes across every item: this keeps
            // GPU memory linear in the number of items rather than quadratic.
            // SplatmapData carries its bytes behind an Arc, so .clone() is
            // a couple of ref-count bumps, no per-item byte copy.
            item.splatmaps = [splat_a.clone(), splat_b.clone()];
            item.height_blend_strength = 12.0;
            item.height_blend_noise_scale = 96.0;
            let mut settings = ItemSettings::default();
            settings.pick_id = PickId(pick_id);
            item.settings = settings;
            pick_id += 1;
            out.push(item);
        }
    }
    out
}

/// Per-item heightmap. Each grid cell gets a slightly different seed so the
/// items don't look identical, but the shape is cheap to compute.
fn build_heightmap(dim: usize, gx: u32, gy: u32) -> Vec<u16> {
    let mut out = Vec::with_capacity(dim * dim);
    let n = (dim - 1) as f32;
    let offset_x = gx as f32 * 1.37;
    let offset_y = gy as f32 * 0.93;
    for y in 0..dim {
        let v = y as f32 / n + offset_y;
        for x in 0..dim {
            let u = x as f32 / n + offset_x;
            let mut h = 0.0_f32;
            h += (u * std::f32::consts::TAU * 1.1).sin() * 0.45;
            h += (v * std::f32::consts::TAU * 0.9).cos() * 0.35;
            h += ((u * 4.7 + v * 3.1).sin() * 0.55 + (v * 2.3).cos() * 0.25) * 0.30;
            h += ((u * 11.0).sin() * (v * 9.0).cos()) * 0.15;
            let unit = (h * 0.5 + 0.5).clamp(0.0, 1.0);
            out.push((unit * u16::MAX as f32) as u16);
        }
    }
    out
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = std::time::Instant::now();
        let dt_ms = now.duration_since(self.last_frame).as_secs_f32() * 1000.0;
        self.last_frame = now;
        self.frame_ms_ema = self.frame_ms_ema * 0.9 + dt_ms * 0.1;
        self.last_pixels_per_point = ctx.pixels_per_point();

        egui::SidePanel::right("hud")
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Terrain stress");
                ui.label(format!(
                    "frame: {:.2} ms ({:.0} fps)",
                    self.frame_ms_ema,
                    1000.0 / self.frame_ms_ema.max(0.001),
                ));
                ui.label(format!("items:   {}", self.total_items()));
                ui.label(format!(
                    "patches: {} ({}x{} per item)",
                    self.total_patches(),
                    self.patches_per_side(),
                    self.patches_per_side(),
                ));
                ui.label(format!(
                    "eye: ({:.0}, {:.0}, {:.0})",
                    self.last_eye.x, self.last_eye.y, self.last_eye.z,
                ));
                ui.separator();

                ui.label("Grid (NxN items)");
                let mut new_grid = self.grid;
                if ui
                    .add(egui::Slider::new(&mut new_grid, 1..=10).text("N"))
                    .changed()
                {
                    self.grid = new_grid;
                }

                ui.label("Per-item heightmap dim");
                ui.horizontal(|ui| {
                    for choice in ITEM_DIM_CHOICES {
                        let selected = self.pending_item_dim == *choice;
                        if ui
                            .selectable_label(selected, format!("{choice}"))
                            .clicked()
                        {
                            self.pending_item_dim = *choice;
                        }
                    }
                });

                ui.label("Patch cells");
                ui.add(egui::Slider::new(&mut self.pending_patch_cells, 4..=64).text("cells"));

                ui.label("Max LOD");
                ui.add(egui::Slider::new(&mut self.pending_max_lod, 1..=5).text("levels"));

                ui.label("LOD distance");
                ui.add(
                    egui::Slider::new(&mut self.pending_lod_distance, 16.0..=256.0)
                        .text("metres"),
                );

                // Project what hitting Apply would build so the user can see
                // how much work they are about to ask for.
                let projected_patches_per_side =
                    ((self.pending_item_dim - 1) / self.pending_patch_cells.max(1)).max(1);
                let projected_total_patches = self.grid
                    * self.grid
                    * projected_patches_per_side
                    * projected_patches_per_side;
                ui.label(format!(
                    "next rebuild: {} items, {} patches",
                    self.grid * self.grid,
                    projected_total_patches,
                ));
                if projected_total_patches > 200_000 {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 140, 40),
                        "Warning: patches > 200k. Bake will allocate a lot of small \
                         GPU buffers and may take a few seconds.",
                    );
                }

                let dirty = self.pending_patch_cells != self.patch_cells
                    || self.pending_max_lod != self.max_lod
                    || self.pending_lod_distance != self.lod_distance
                    || self.pending_item_dim != self.item_dim
                    || self.items.len() as u32 != self.total_items();
                ui.add_enabled_ui(dirty, |ui| {
                    if ui.button("Apply and rebuild").clicked() {
                        self.apply_pending();
                    }
                });

                ui.separator();
                ui.checkbox(&mut self.auto_orbit, "Auto-orbit (steady workload)");
                ui.horizontal(|ui| {
                    if ui.button("Reset (overhead)").clicked() {
                        self.camera.distance = ITEM_WORLD_SIZE * self.grid as f32 * 1.3;
                        self.camera.center = glam::Vec3::ZERO;
                        self.camera.orientation =
                            glam::Quat::from_rotation_z(0.6) * glam::Quat::from_rotation_x(1.1);
                    }
                    if ui.button("Ground view").clicked() {
                        // Camera near terrain height, looking along +X so as
                        // much of the grid stays inside the frustum as
                        // possible. Maximises per-frame patches-visible.
                        self.camera.center = glam::Vec3::new(0.0, 0.0, HEIGHT_MAX * 0.5);
                        self.camera.distance = ITEM_WORLD_SIZE * self.grid as f32 * 0.55;
                        self.camera.orientation =
                            glam::Quat::from_rotation_z(0.0) * glam::Quat::from_rotation_x(1.55);
                    }
                });

                ui.separator();
                ui.collapsing("What this stresses", |ui| {
                    ui.label(
                        "The plugin's cull() walks every patch on the CPU and \
                         picks an LOD per patch. paint() then issues one draw \
                         per visible patch (with its own vbuf+ibuf bind). On a \
                         modern desktop GPU, neither cost shows up as lag until \
                         total patch counts cross ~100k.",
                    );
                    ui.label(
                        "When GPU cull lands, the per-patch CPU loop disappears \
                         and visible-patch selection moves into one compute \
                         dispatch. The gap is most visible (a) at very high \
                         patch counts, (b) once the plugin adds shadow casting \
                         (multiplies cull cost by cascade count), and (c) once \
                         occlusion / Hi-Z arrives.",
                    );
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            self.left_pressed_this_frame = false;
            self.controller.begin_frame(ViewportContext {
                hovered: response.hovered(),
                focused: response.has_focus(),
                viewport_size: [rect.width(), rect.height()],
            });

            ui.input(|i| {
                self.controller.push_event(ViewportEvent::ModifiersChanged(
                    viewport_lib::Modifiers {
                        alt: i.modifiers.alt,
                        shift: i.modifiers.shift,
                        ctrl: i.modifiers.command,
                    },
                ));
                let pointer_in_viewport = i
                    .pointer
                    .interact_pos()
                    .map_or(false, |p| rect.contains(p));
                if let Some(p) = i.pointer.interact_pos() {
                    let local = glam::Vec2::new(p.x - rect.left(), p.y - rect.top());
                    self.cursor_viewport = Some(local);
                    if pointer_in_viewport || self.drag_in_viewport {
                        self.controller
                            .push_event(ViewportEvent::PointerMoved { position: local });
                    }
                }
                for event in &i.events {
                    match event {
                        egui::Event::PointerButton {
                            button, pressed, ..
                        } => {
                            let vp_button = match button {
                                egui::PointerButton::Primary => viewport_lib::MouseButton::Left,
                                egui::PointerButton::Secondary => viewport_lib::MouseButton::Right,
                                egui::PointerButton::Middle => viewport_lib::MouseButton::Middle,
                                _ => continue,
                            };
                            if *pressed {
                                if !pointer_in_viewport {
                                    continue;
                                }
                                if *button == egui::PointerButton::Primary {
                                    self.drag_in_viewport = true;
                                    self.left_pressed_this_frame = true;
                                }
                                self.controller.push_event(ViewportEvent::MouseButton {
                                    button: vp_button,
                                    state: ButtonState::Pressed,
                                });
                            } else {
                                if self.drag_in_viewport
                                    || *button != egui::PointerButton::Primary
                                {
                                    self.controller.push_event(ViewportEvent::MouseButton {
                                        button: vp_button,
                                        state: ButtonState::Released,
                                    });
                                }
                                if *button == egui::PointerButton::Primary {
                                    self.drag_in_viewport = false;
                                }
                            }
                        }
                        egui::Event::MouseWheel { unit, delta, .. } => {
                            if !pointer_in_viewport {
                                continue;
                            }
                            let units = match unit {
                                egui::MouseWheelUnit::Line => ScrollUnits::Lines,
                                egui::MouseWheelUnit::Point => ScrollUnits::Pixels,
                                egui::MouseWheelUnit::Page => ScrollUnits::Pages,
                            };
                            self.controller.push_event(ViewportEvent::Wheel {
                                delta: glam::Vec2::new(delta.x, delta.y),
                                units,
                            });
                        }
                        _ => {}
                    }
                }
            });

            let w = rect.width();
            let h = rect.height();
            if self.auto_orbit {
                let yaw = (dt_ms / 1000.0) * 0.25;
                self.camera.orientation =
                    glam::Quat::from_rotation_z(yaw) * self.camera.orientation;
                ctx.request_repaint();
            }
            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(w, h);

            let mut camera_frame = CameraFrame::from_camera(&self.camera, [w, h]);
            camera_frame.pixels_per_point = self.last_pixels_per_point;
            self.last_eye = glam::Vec3::from(camera_frame.render_camera.eye_position);

            let mut frame_data = FrameData::new(camera_frame, SceneFrame::default());
            frame_data.effects.lighting = LightingSettings::default();

            // Submit every grid item this frame. TerrainItem owns the
            // heightmap behind an Arc so the per-frame clone is cheap.
            frame_data.scene.submit_plugin_items(
                viewport_lib_terrain::TYPE_NAME,
                TerrainCollection {
                    items: self.items.clone(),
                },
            );

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    ViewportCallback { frame: frame_data },
                ));

            if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        });
    }
}

impl App {
    /// Tear down the current item set and rebuild from the pending sliders.
    /// The plugin's baked-patch cache is keyed by `PickId`; bumping the base
    /// pick id every rebuild forces fresh entries so old caches age out as
    /// new items walk past them.
    fn apply_pending(&mut self) {
        self.patch_cells = self.pending_patch_cells;
        self.max_lod = self.pending_max_lod;
        self.lod_distance = self.pending_lod_distance;
        self.item_dim = self.pending_item_dim;
        let base = (self.items.len() as u64 + 1).max(1);
        self.items = build_grid(
            self.grid,
            self.item_dim,
            &self.layers,
            &self.splat_a,
            &self.splat_b,
            base,
        );
        // Note: the plugin holds its construction-time `patch_cells` /
        // `max_lod` / `lod_distance`. Changing them at runtime needs a new
        // plugin instance, which `ViewportRenderer` does not currently
        // re-register at runtime. The Apply button still rebuilds the item
        // set so grid-size changes take effect immediately.
        let _ = self.layers; // silence unused-field warning if layers ever change
        let _ = Arc::strong_count(&self.items[0].heightmap);
    }
}

struct ViewportCallback {
    frame: FrameData,
}

impl eframe::egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _sd: &eframe::egui_wgpu::ScreenDescriptor,
        _enc: &mut eframe::wgpu::CommandEncoder,
        resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        if let Some(renderer) = resources.get_mut::<ViewportRenderer>() {
            return renderer.pass().prepare(device, queue, &self.frame);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = resources.get::<ViewportRenderer>() {
            renderer.pass_view().paint(render_pass, &self.frame);
        }
    }
}
