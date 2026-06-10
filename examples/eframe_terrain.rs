//! Heightmap + splatmap terrain demo (eframe + wgpu).
//!
//! Generates a procedural 257x257 heightmap and a matching RGBA
//! splatmap (sand low / grass mid / dirt mid-high / rock peaks +
//! steep), renders through the terrain plugin, orbits a camera.
//! Left-click toggles selection so the outline highlight exercises.

use eframe::egui;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, ItemSettings, LightingSettings,
    OrbitCameraController, PickId, SceneFrame, ScrollUnits, ViewportContext, ViewportEvent,
    ViewportRenderer,
};
use viewport_lib_terrain::{
    LAYER_COUNT, SplatmapData, TerrainCollection, TerrainItem, TerrainLayer, TerrainPlugin,
};

const TERRAIN_DIM: u32 = 257;
const WORLD_SIZE: f32 = 64.0;
const HEIGHT_MIN: f32 = -1.0;
const HEIGHT_MAX: f32 = 6.0;

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib-terrain : Splatmap demo",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
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
            renderer.with_item_type_plugin(device, Box::new(TerrainPlugin::new()));

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
}

impl App {
    fn new() -> Self {
        let dim = TERRAIN_DIM as usize;
        let heightmap = build_heightmap(dim);
        let (splat_a, splat_b) =
            build_splatmaps(&heightmap, dim, dim, WORLD_SIZE, HEIGHT_MIN, HEIGHT_MAX);

        // Eight layers. Splatmap A.r/g/b/a -> layers 0..4 (the primary
        // ground covers). Splatmap B.r/g/b/a -> layers 4..8 (accent
        // covers that compete with the primaries via height-blend).
        let layers: [TerrainLayer; LAYER_COUNT] = [
            // A.r: grass
            TerrainLayer {
                albedo: [0.30, 0.46, 0.20],
                metallic: 0.0,
                roughness: 0.92,
                height_bias: 0.0,
            },
            // A.g: dirt
            TerrainLayer {
                albedo: [0.40, 0.27, 0.16],
                metallic: 0.0,
                roughness: 0.85,
                height_bias: 0.0,
            },
            // A.b: rock
            TerrainLayer {
                albedo: [0.42, 0.40, 0.38],
                metallic: 0.05,
                roughness: 0.70,
                height_bias: 0.15,
            },
            // A.a: sand
            TerrainLayer {
                albedo: [0.78, 0.70, 0.46],
                metallic: 0.0,
                roughness: 0.78,
                height_bias: 0.0,
            },
            // B.r: pebbles (height-blends over sand and dirt)
            TerrainLayer {
                albedo: [0.55, 0.50, 0.42],
                metallic: 0.0,
                roughness: 0.60,
                height_bias: 0.55,
            },
            // B.g: moss (sneaks over grass at low spots)
            TerrainLayer {
                albedo: [0.18, 0.32, 0.16],
                metallic: 0.0,
                roughness: 0.95,
                height_bias: -0.15,
            },
            // B.b: snow (sits high)
            TerrainLayer {
                albedo: [0.92, 0.94, 0.97],
                metallic: 0.0,
                roughness: 0.50,
                height_bias: 0.30,
            },
            // B.a: dark rock vein (slightly raised over base rock)
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
            SplatmapData::new(splat_a, [TERRAIN_DIM, TERRAIN_DIM]),
            SplatmapData::new(splat_b, [TERRAIN_DIM, TERRAIN_DIM]),
        ];
        terrain.height_blend_strength = 14.0;
        terrain.height_blend_noise_scale = 96.0;
        terrain.settings = settings;

        Self {
            camera: Camera {
                distance: 70.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            terrain,
            cursor_viewport: None,
            left_pressed_this_frame: false,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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
                if let Some(p) = i.pointer.interact_pos() {
                    let local = glam::Vec2::new(p.x - rect.left(), p.y - rect.top());
                    self.cursor_viewport = Some(local);
                    self.controller
                        .push_event(ViewportEvent::PointerMoved { position: local });
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
                            self.controller.push_event(ViewportEvent::MouseButton {
                                button: vp_button,
                                state: if *pressed {
                                    ButtonState::Pressed
                                } else {
                                    ButtonState::Released
                                },
                            });
                            if *button == egui::PointerButton::Primary && *pressed {
                                self.left_pressed_this_frame = true;
                            }
                        }
                        egui::Event::MouseWheel { unit, delta, .. } => {
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
            camera_frame.pixels_per_point = ctx.pixels_per_point();

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
            let mut h = 0.0_f32;
            h += (u * std::f32::consts::TAU * 1.5).sin() * 0.4;
            h += (v * std::f32::consts::TAU * 1.2).cos() * 0.3;
            h += ((u * 4.7 + v * 3.1).sin() * 0.5 + (v * 2.3).cos() * 0.3) * 0.3;
            let unit = (h * 0.5 + 0.5).clamp(0.0, 1.0);
            out.push((unit * u16::MAX as f32) as u16);
        }
    }
    out
}

/// Paint the two splatmaps from the heightmap.
///
/// Splatmap A (primary layers 0..4):
/// - R (grass): mid altitudes, gentle slopes.
/// - G (dirt): mid-high altitudes, moderate slopes.
/// - B (rock): high altitudes or steep slopes.
/// - A (sand): low altitudes near sea level.
///
/// Splatmap B (accent layers 4..8) overlaps the primaries; the
/// shader's height-blend pass decides which accent wins per pixel:
/// - R (pebbles): low / sandy areas.
/// - G (moss): low slopes with grass.
/// - B (snow): peaks.
/// - A (dark rock vein): mid-high rock.
fn build_splatmaps(
    heights: &[u16],
    w: usize,
    h: usize,
    world_size: f32,
    h_min: f32,
    h_max: f32,
) -> (Vec<u8>, Vec<u8>) {
    let span = h_max - h_min;
    let dx_world = world_size / (w - 1) as f32;
    let dy_world = world_size / (h - 1) as f32;

    let height_at = |x: usize, y: usize| -> f32 {
        let raw = heights[y * w + x] as f32 / u16::MAX as f32;
        h_min + raw * span
    };

    let mut out_a = Vec::with_capacity(w * h * 4);
    let mut out_b = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let z = height_at(x, y);
            // Slope from central differences.
            let xl = if x == 0 { x } else { x - 1 };
            let xr = if x + 1 == w { x } else { x + 1 };
            let yl = if y == 0 { y } else { y - 1 };
            let yr = if y + 1 == h { y } else { y + 1 };
            let dzdx = (height_at(xr, y) - height_at(xl, y))
                / ((xr - xl).max(1) as f32 * dx_world);
            let dzdy = (height_at(x, yr) - height_at(x, yl))
                / ((yr - yl).max(1) as f32 * dy_world);
            let slope = (dzdx * dzdx + dzdy * dzdy).sqrt();

            let alt = ((z - h_min) / span).clamp(0.0, 1.0);

            // Primary covers (splatmap A).
            let sand = smooth_band(alt, 0.00, 0.18) * (1.0 - smoothstep(slope, 0.3, 0.7));
            let grass = smooth_band(alt, 0.15, 0.55) * (1.0 - smoothstep(slope, 0.5, 1.0));
            let dirt = smooth_band(alt, 0.45, 0.78) * (1.0 - smoothstep(slope, 0.8, 1.4));
            let rock_alt = smoothstep(alt, 0.7, 0.95);
            let rock_slope = smoothstep(slope, 0.8, 1.6);
            let rock = (rock_alt + rock_slope).min(1.0);

            out_a.push(quantise(grass));
            out_a.push(quantise(dirt));
            out_a.push(quantise(rock));
            out_a.push(quantise(sand));

            // Accent covers (splatmap B). These compete with the
            // primaries via the height-blend pass in the shader.
            let pebbles = smooth_band(alt, 0.00, 0.30) * (1.0 - smoothstep(slope, 0.4, 0.9));
            let moss = smooth_band(alt, 0.20, 0.45) * (1.0 - smoothstep(slope, 0.3, 0.8));
            let snow = smoothstep(alt, 0.82, 0.98) * (1.0 - smoothstep(slope, 1.4, 2.0));
            let dark_rock = smoothstep(alt, 0.60, 0.88) * smoothstep(slope, 0.6, 1.4);

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
