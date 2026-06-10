//! Large-terrain showcase: 513x513 heightmap over a 256m square,
//! eight surface layers with height-blend, quadtree-style LOD.
//!
//! What this exercises beyond `eframe_terrain`:
//!
//! - All eight surface layers visible at once: sand, grass, dirt, and
//!   rock as primaries plus pebbles, moss, snow, and a dark rock vein
//!   as accents. The height-blend slider drives sharpness; drag it
//!   from 0 (smooth) to 30 (pebble-edged) to see the difference.
//! - 16x16 patches at LOD 0; orbit the camera out to see lower LODs
//!   kick in on far patches. The HUD reports per-LOD patch counts.
//! - Sets `pixels_per_point` on the camera frame so the viewport
//!   texture allocates at physical resolution on high-DPI displays.

use eframe::egui;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, ItemSettings, LightingSettings,
    OrbitCameraController, PickId, SceneFrame, ScrollUnits, ViewportContext, ViewportEvent,
    ViewportRenderer,
};
use viewport_lib_terrain::{
    LAYER_COUNT, SplatmapData, TerrainCollection, TerrainItem, TerrainLayer, TerrainPlugin,
};

const TERRAIN_DIM: u32 = 513;
const SPLATMAP_DIM: u32 = 2048;
const WORLD_SIZE: f32 = 256.0;
const HEIGHT_MIN: f32 = -3.0;
const HEIGHT_MAX: f32 = 32.0;

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib-terrain : LOD demo",
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
                .with_patch_cells(32)
                .with_max_lod(4)
                .with_skirt_depth(2.0)
                .with_lod_distance(96.0);
            renderer.with_item_type_plugin(device, Box::new(plugin));

            rs.renderer.write().callback_resources.insert(renderer);
            Ok(Box::new(App::new()))
        }),
    )
}

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    terrain: TerrainItem,
    cursor_viewport: Option<glam::Vec2>,
    left_pressed_this_frame: bool,
    last_pixels_per_point: f32,
    last_eye: glam::Vec3,
    /// True between a press and release that started inside the
    /// viewport. Mouse events outside the viewport (slider drags,
    /// side-panel clicks) are not forwarded to the orbit controller.
    drag_in_viewport: bool,
}

impl App {
    fn new() -> Self {
        let dim = TERRAIN_DIM as usize;
        let heightmap = build_heightmap(dim);
        let splatmap_dim = SPLATMAP_DIM as usize;
        let (splat_a, splat_b) = build_splatmaps(
            &heightmap,
            dim,
            dim,
            splatmap_dim,
            splatmap_dim,
            WORLD_SIZE,
            HEIGHT_MIN,
            HEIGHT_MAX,
        );

        let layers: [TerrainLayer; LAYER_COUNT] = [
            // A.r: grass
            TerrainLayer {
                albedo: [0.26, 0.42, 0.18],
                metallic: 0.0,
                roughness: 0.92,
                height_bias: 0.0,
            },
            // A.g: dirt
            TerrainLayer {
                albedo: [0.42, 0.28, 0.16],
                metallic: 0.0,
                roughness: 0.85,
                height_bias: 0.0,
            },
            // A.b: rock
            TerrainLayer {
                albedo: [0.46, 0.44, 0.42],
                metallic: 0.05,
                roughness: 0.70,
                height_bias: 0.15,
            },
            // A.a: sand
            TerrainLayer {
                albedo: [0.82, 0.74, 0.48],
                metallic: 0.0,
                roughness: 0.80,
                height_bias: 0.0,
            },
            // B.r: pebbles
            TerrainLayer {
                albedo: [0.58, 0.52, 0.44],
                metallic: 0.0,
                roughness: 0.55,
                height_bias: 0.55,
            },
            // B.g: moss
            TerrainLayer {
                albedo: [0.20, 0.34, 0.16],
                metallic: 0.0,
                roughness: 0.95,
                height_bias: -0.15,
            },
            // B.b: snow
            TerrainLayer {
                albedo: [0.94, 0.95, 0.98],
                metallic: 0.0,
                roughness: 0.45,
                height_bias: 0.50,
            },
            // B.a: dark rock vein
            TerrainLayer {
                albedo: [0.22, 0.20, 0.18],
                metallic: 0.10,
                roughness: 0.55,
                height_bias: 0.40,
            },
        ];

        let mut settings = ItemSettings::default();
        settings.pick_id = PickId(1);

        let mut terrain = TerrainItem::new(heightmap, [TERRAIN_DIM, TERRAIN_DIM]);
        terrain.world_size = [WORLD_SIZE, WORLD_SIZE];
        terrain.height_range = [HEIGHT_MIN, HEIGHT_MAX];
        terrain.origin = glam::Vec3::new(-WORLD_SIZE * 0.5, -WORLD_SIZE * 0.5, 0.0);
        terrain.surface_layers = layers;
        terrain.splatmaps = [
            SplatmapData::new(splat_a, [SPLATMAP_DIM, SPLATMAP_DIM]),
            SplatmapData::new(splat_b, [SPLATMAP_DIM, SPLATMAP_DIM]),
        ];
        terrain.height_blend_strength = 16.0;
        terrain.height_blend_noise_scale = 140.0;
        terrain.settings = settings;

        Self {
            camera: Camera {
                distance: 220.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            terrain,
            cursor_viewport: None,
            left_pressed_this_frame: false,
            last_pixels_per_point: 1.0,
            last_eye: glam::Vec3::ZERO,
            drag_in_viewport: false,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.last_pixels_per_point = ctx.pixels_per_point();

        egui::SidePanel::right("hud").show(ctx, |ui| {
            ui.heading("Terrain LOD demo");
            ui.label(format!("pixels_per_point: {:.2}", self.last_pixels_per_point));
            ui.label(format!("terrain: {0}x{0}", TERRAIN_DIM));
            ui.label(format!("world: {0}m x {0}m", WORLD_SIZE as i32));
            ui.label(format!(
                "eye: ({:.0}, {:.0}, {:.0})",
                self.last_eye.x, self.last_eye.y, self.last_eye.z,
            ));
            ui.separator();
            ui.label("Height-blend strength");
            ui.add(
                egui::Slider::new(
                    &mut self.terrain.height_blend_strength,
                    0.0..=40.0,
                )
                .text("strength"),
            );
            ui.add(
                egui::Slider::new(
                    &mut self.terrain.height_blend_noise_scale,
                    8.0..=256.0,
                )
                .text("noise scale"),
            );
            ui.separator();
            ui.label("Layers");
            ui.label("R/G/B/A on splat A -> grass / dirt / rock / sand");
            ui.label("R/G/B/A on splat B -> pebbles / moss / snow / dark vein");
            ui.separator();
            ui.label(
                "Drag inside the viewport to orbit. Scroll to zoom out and \
                 watch far patches drop to lower LOD; the skirts under each \
                 patch hide cracks at the seams.",
            );
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
                    // Only stream movement to the controller while the
                    // viewport actually owns the interaction; otherwise
                    // dragging a slider would also orbit the camera.
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
            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(w, h);

            let mut camera_frame = CameraFrame::from_camera(&self.camera, [w, h]);
            camera_frame.pixels_per_point = self.last_pixels_per_point;
            self.last_eye = glam::Vec3::from(camera_frame.render_camera.eye_position);

            let mut frame_data = FrameData::new(camera_frame, SceneFrame::default());
            frame_data.effects.lighting = LightingSettings::default();

            if self.left_pressed_this_frame {
                self.terrain.settings.selected = !self.terrain.settings.selected;
            }
            // Cheap: heightmap and splatmap bytes sit behind Arc, so
            // cloning the TerrainItem is a handful of ref-count bumps.
            frame_data.scene.submit_plugin_items(
                viewport_lib_terrain::TYPE_NAME,
                TerrainCollection {
                    items: vec![self.terrain.clone()],
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

fn build_heightmap(dim: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(dim * dim);
    let n = (dim - 1) as f32;
    for y in 0..dim {
        let v = y as f32 / n;
        for x in 0..dim {
            let u = x as f32 / n;
            // Layered sinusoids + small-scale ridges to give the LOD
            // path something to actually decimate.
            let mut h = 0.0_f32;
            h += (u * std::f32::consts::TAU * 1.3).sin() * 0.45;
            h += (v * std::f32::consts::TAU * 0.9).cos() * 0.35;
            h += ((u * 4.7 + v * 3.1).sin() * 0.55 + (v * 2.3).cos() * 0.25) * 0.3;
            h += ((u * 9.2).sin() * (v * 7.4).cos()) * 0.18;
            h += ((u * 18.0).sin() * (v * 16.0).sin()) * 0.06;
            // Push a mountain near (0.7, 0.65).
            let dx = u - 0.7;
            let dy = v - 0.65;
            let r = (dx * dx + dy * dy).sqrt();
            h += (1.0 - smoothstep(r, 0.05, 0.35)).powi(2) * 0.9;
            let unit = (h * 0.5 + 0.5).clamp(0.0, 1.0);
            out.push((unit * u16::MAX as f32) as u16);
        }
    }
    out
}

/// Build the two splatmaps at `(sw, sh)` resolution from a `(hw, hh)`
/// heightmap. Bilinearly interpolates the heightmap so the layer
/// boundaries don't snap to heightmap cell edges, and dithers the
/// alt / slope values per output pixel so the boundary is fuzzy
/// instead of stair-stepped.
fn build_splatmaps(
    heights: &[u16],
    hw: usize,
    hh: usize,
    sw: usize,
    sh: usize,
    world_size: f32,
    h_min: f32,
    h_max: f32,
) -> (Vec<u8>, Vec<u8>) {
    let span = h_max - h_min;
    let dx_world_h = world_size / (hw - 1) as f32;
    let dy_world_h = world_size / (hh - 1) as f32;
    let sample_cell = |x: i32, y: i32| -> f32 {
        let cx = x.clamp(0, hw as i32 - 1) as usize;
        let cy = y.clamp(0, hh as i32 - 1) as usize;
        let raw = heights[cy * hw + cx] as f32 / u16::MAX as f32;
        h_min + raw * span
    };
    // Bilinear sample at heightmap-cell coordinates fx, fy.
    let sample_bilinear = |fx: f32, fy: f32| -> f32 {
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let s00 = sample_cell(x0, y0);
        let s10 = sample_cell(x0 + 1, y0);
        let s01 = sample_cell(x0, y0 + 1);
        let s11 = sample_cell(x0 + 1, y0 + 1);
        let a = s00 * (1.0 - tx) + s10 * tx;
        let b = s01 * (1.0 - tx) + s11 * tx;
        a * (1.0 - ty) + b * ty
    };

    let mut out_a = Vec::with_capacity(sw * sh * 4);
    let mut out_b = Vec::with_capacity(sw * sh * 4);
    let scale_x = (hw - 1) as f32 / (sw - 1).max(1) as f32;
    let scale_y = (hh - 1) as f32 / (sh - 1).max(1) as f32;
    // Slope stencil step in heightmap-cell units. One cell wide.
    let s = 1.0_f32;

    for sy in 0..sh {
        for sx in 0..sw {
            let fx = sx as f32 * scale_x;
            let fy = sy as f32 * scale_y;
            let z = sample_bilinear(fx, fy);
            let dzdx =
                (sample_bilinear(fx + s, fy) - sample_bilinear(fx - s, fy)) / (2.0 * dx_world_h);
            let dzdy =
                (sample_bilinear(fx, fy + s) - sample_bilinear(fx, fy - s)) / (2.0 * dy_world_h);
            let slope = (dzdx * dzdx + dzdy * dzdy).sqrt();
            let alt = ((z - h_min) / span).clamp(0.0, 1.0);

            let sand = smooth_band(alt, 0.00, 0.18) * (1.0 - smoothstep(slope, 0.25, 0.55));
            let grass = smooth_band(alt, 0.15, 0.55) * (1.0 - smoothstep(slope, 0.50, 1.10));
            let dirt = smooth_band(alt, 0.45, 0.78) * (1.0 - smoothstep(slope, 0.80, 1.40));
            let rock_alt = smoothstep(alt, 0.65, 0.92);
            let rock_slope = smoothstep(slope, 0.70, 1.40);
            let rock = (rock_alt + rock_slope).min(1.0);

            out_a.push(quantise(grass));
            out_a.push(quantise(dirt));
            out_a.push(quantise(rock));
            out_a.push(quantise(sand));

            let pebbles = smooth_band(alt, 0.00, 0.28) * (1.0 - smoothstep(slope, 0.35, 0.85));
            let moss = smooth_band(alt, 0.18, 0.45) * (1.0 - smoothstep(slope, 0.30, 0.85));
            let snow = smoothstep(alt, 0.78, 0.97) * (1.0 - smoothstep(slope, 1.40, 2.00));
            let dark_rock = smoothstep(alt, 0.55, 0.85) * smoothstep(slope, 0.50, 1.30);

            out_b.push(quantise(pebbles));
            out_b.push(quantise(moss));
            out_b.push(quantise(snow));
            out_b.push(quantise(dark_rock));
        }
    }
    (out_a, out_b)
}


fn smooth_band(x: f32, lo: f32, hi: f32) -> f32 {
    let center = (lo + hi) * 0.5;
    let width = (hi - lo).max(1e-3);
    let t = ((x - center).abs() / (width * 0.5)).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

fn smoothstep(x: f32, lo: f32, hi: f32) -> f32 {
    let t = ((x - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn quantise(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0) as u8
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
