//! Grass + tree LOD demo.
//!
//! Builds on the LOD demo by scattering detail meshes and billboards
//! across the terrain every frame:
//!
//! - Grass tufts: world-space billboards scattered on the "grass"
//!   splatmap channel, density ramped by a slider.
//! - Trees: instanced cone-on-cylinder mesh near the camera, swapped
//!   for a flat green billboard impostor past the LOD switch distance.
//!
//! Both batches are produced by `scatter_terrain_details`, which runs
//! a Halton(2,3) sequence over a camera-centred disc and accepts
//! samples weighted by the painted layer at each point. Nothing is
//! retained between frames; placement is deterministic per camera
//! position.

use eframe::egui;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, ItemSettings, LightingSettings,
    OrbitCameraController, PickId, SceneFrame, ScrollUnits, ViewportContext, ViewportEvent,
    ViewportRenderer, primitives,
};
use viewport_lib_terrain::{
    DetailKind, DetailLayer, DetailScatterOutput, DetailScatterParams, LAYER_COUNT, SplatmapData,
    TerrainCollection, TerrainItem, TerrainLayer, TerrainPlugin, scatter_terrain_details,
};

const TERRAIN_DIM: u32 = 513;
const SPLATMAP_DIM: u32 = 1024;
const WORLD_SIZE: f32 = 256.0;
const HEIGHT_MIN: f32 = -3.0;
const HEIGHT_MAX: f32 = 32.0;

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib-terrain : grass + trees",
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

            // Tree mesh: simple stylised tree (a tall cylinder trunk
            // would be ideal, but a cube scaled non-uniformly via the
            // instance transform keeps the example small).
            let tree_mesh = primitives::cube(1.0);
            let tree_id = renderer
                .resources_mut()
                .upload_mesh_data(device, &tree_mesh)
                .expect("upload tree mesh")
                .index() as u64;

            rs.renderer.write().callback_resources.insert(renderer);
            Ok(Box::new(App::new(tree_id)))
        }),
    )
}

struct DetailUi {
    grass_density: f32,
    grass_size: f32,
    grass_radius: f32,
    tree_density: f32,
    tree_size: f32,
    tree_radius: f32,
    tree_switch: f32,
}

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    terrain: TerrainItem,
    cursor_viewport: Option<glam::Vec2>,
    last_pixels_per_point: f32,
    last_eye: glam::Vec3,
    last_forward: glam::Vec3,
    last_distance: f32,
    drag_in_viewport: bool,
    tree_mesh_id: u64,
    detail_out: DetailScatterOutput,
    ui: DetailUi,
}

impl App {
    fn new(tree_mesh_id: u64) -> Self {
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
            TerrainLayer { albedo: [0.26, 0.42, 0.18], metallic: 0.0, roughness: 0.92, height_bias: 0.0 },
            TerrainLayer { albedo: [0.42, 0.28, 0.16], metallic: 0.0, roughness: 0.85, height_bias: 0.0 },
            TerrainLayer { albedo: [0.46, 0.44, 0.42], metallic: 0.05, roughness: 0.70, height_bias: 0.15 },
            TerrainLayer { albedo: [0.82, 0.74, 0.48], metallic: 0.0, roughness: 0.80, height_bias: 0.0 },
            TerrainLayer { albedo: [0.58, 0.52, 0.44], metallic: 0.0, roughness: 0.55, height_bias: 0.55 },
            TerrainLayer { albedo: [0.20, 0.34, 0.16], metallic: 0.0, roughness: 0.95, height_bias: -0.15 },
            TerrainLayer { albedo: [0.94, 0.95, 0.98], metallic: 0.0, roughness: 0.45, height_bias: 0.50 },
            TerrainLayer { albedo: [0.22, 0.20, 0.18], metallic: 0.10, roughness: 0.55, height_bias: 0.40 },
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
                distance: 120.0,
                ..Camera::default()
            },
            controller: OrbitCameraController::viewport_primitives(),
            terrain,
            cursor_viewport: None,
            last_pixels_per_point: 1.0,
            last_eye: glam::Vec3::ZERO,
            last_forward: -glam::Vec3::Z,
            last_distance: 120.0,
            drag_in_viewport: false,
            tree_mesh_id,
            detail_out: DetailScatterOutput::default(),
            ui: DetailUi {
                grass_density: 0.5,
                grass_size: 1.0,
                grass_radius: 60.0,
                tree_density: 0.02,
                tree_size: 4.0,
                tree_radius: 120.0,
                tree_switch: 50.0,
            },
        }
    }

    fn build_detail_layers(&self) -> Vec<DetailLayer> {
        vec![
            // Grass tufts on the "grass" channel (A.r = layer 0).
            DetailLayer {
                kind: DetailKind::Billboard {
                    texture_id: None,
                    colour: [0.32, 0.58, 0.22, 1.0],
                },
                surface_layer: 0,
                weight_threshold: 0.35,
                size: self.ui.grass_size,
                size_jitter: 0.4,
                max_distance: self.ui.grass_radius,
                density: self.ui.grass_density,
            },
            // Trees on the "dirt" channel (A.g = layer 1).
            DetailLayer {
                kind: DetailKind::TreeLod {
                    mesh_id: self.tree_mesh_id,
                    mesh_texture_id: None,
                    impostor_texture_id: None,
                    colour: [0.18, 0.32, 0.14, 1.0],
                    switch_distance: self.ui.tree_switch,
                },
                surface_layer: 1,
                weight_threshold: 0.45,
                size: self.ui.tree_size,
                size_jitter: 0.25,
                max_distance: self.ui.tree_radius,
                density: self.ui.tree_density,
            },
        ]
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.last_pixels_per_point = ctx.pixels_per_point();

        egui::SidePanel::right("hud").show(ctx, |ui| {
            ui.heading("Grass + trees");
            ui.label(format!("eye: ({:.0}, {:.0}, {:.0})", self.last_eye.x, self.last_eye.y, self.last_eye.z));
            ui.separator();
            ui.label("Grass (billboards on grass layer)");
            ui.add(egui::Slider::new(&mut self.ui.grass_density, 0.0..=4.0).text("density /m^2"));
            ui.add(egui::Slider::new(&mut self.ui.grass_size, 0.2..=3.0).text("size"));
            ui.add(egui::Slider::new(&mut self.ui.grass_radius, 10.0..=150.0).text("radius"));
            ui.separator();
            ui.label("Trees (LOD mesh -> impostor)");
            ui.add(egui::Slider::new(&mut self.ui.tree_density, 0.0..=0.2).text("density /m^2"));
            ui.add(egui::Slider::new(&mut self.ui.tree_size, 1.0..=10.0).text("size"));
            ui.add(egui::Slider::new(&mut self.ui.tree_radius, 20.0..=250.0).text("radius"));
            ui.add(egui::Slider::new(&mut self.ui.tree_switch, 5.0..=150.0).text("switch dist"));
            ui.separator();
            ui.label(format!("sprites: {}", self.detail_out.sprites.iter().map(|s| s.positions.len()).sum::<usize>()));
            ui.label(format!("mesh instances: {}", self.detail_out.mesh_instances.iter().map(|m| m.transforms.len()).sum::<usize>()));
            ui.separator();
            ui.label(
                "Detail batches are rebuilt every frame around the camera \
                 from a Halton sequence. Trees within `switch dist` draw as \
                 cube meshes; trees beyond draw as flat impostor sprites.",
            );
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

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
                let pointer_in_viewport =
                    i.pointer.interact_pos().map_or(false, |p| rect.contains(p));
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
                        egui::Event::PointerButton { button, pressed, .. } => {
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
            self.last_forward = glam::Vec3::from(camera_frame.render_camera.forward);
            self.last_distance = camera_frame.render_camera.distance;

            let mut frame_data = FrameData::new(camera_frame, SceneFrame::default());
            frame_data.effects.lighting = LightingSettings::default();

            frame_data.scene.submit_plugin_items(
                viewport_lib_terrain::TYPE_NAME,
                TerrainCollection { items: vec![self.terrain.clone()] },
            );

            // Scatter around the orbit focal point on the ground, not
            // the eye; otherwise the detail footprint slides off the
            // look-at as the camera orbits.
            let focal = self.last_eye + self.last_forward * self.last_distance;
            let params = DetailScatterParams {
                camera_xy: glam::Vec2::new(focal.x, focal.y),
                camera_z: focal.z,
            };
            let layers = self.build_detail_layers();
            scatter_terrain_details(&self.terrain, &layers, &params, &mut self.detail_out);
            frame_data.scene.sprite_items.extend(self.detail_out.sprites.iter().cloned());
            frame_data
                .scene
                .mesh_instances
                .extend(self.detail_out.mesh_instances.iter().cloned());

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    ViewportCallback { frame: frame_data },
                ));

            if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
            ctx.request_repaint();
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
            h += (u * std::f32::consts::TAU * 1.3).sin() * 0.45;
            h += (v * std::f32::consts::TAU * 0.9).cos() * 0.35;
            h += ((u * 4.7 + v * 3.1).sin() * 0.55 + (v * 2.3).cos() * 0.25) * 0.3;
            h += ((u * 9.2).sin() * (v * 7.4).cos()) * 0.18;
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
    let s = 1.0_f32;

    for sy in 0..sh {
        for sx in 0..sw {
            let fx = sx as f32 * scale_x;
            let fy = sy as f32 * scale_y;
            let z = sample_bilinear(fx, fy);
            let dzdx = (sample_bilinear(fx + s, fy) - sample_bilinear(fx - s, fy))
                / (2.0 * dx_world_h);
            let dzdy = (sample_bilinear(fx, fy + s) - sample_bilinear(fx, fy - s))
                / (2.0 * dy_world_h);
            let slope = (dzdx * dzdx + dzdy * dzdy).sqrt();
            let alt = ((z - h_min) / span).clamp(0.0, 1.0);

            let sand = smooth_band(alt, 0.0, 0.18) * (1.0 - smoothstep(slope, 0.25, 0.55));
            let grass = smooth_band(alt, 0.15, 0.55) * (1.0 - smoothstep(slope, 0.50, 1.10));
            let dirt = smooth_band(alt, 0.45, 0.78) * (1.0 - smoothstep(slope, 0.80, 1.40));
            let rock = (smoothstep(alt, 0.65, 0.92) + smoothstep(slope, 0.70, 1.40)).min(1.0);

            out_a.push(quantise(grass));
            out_a.push(quantise(dirt));
            out_a.push(quantise(rock));
            out_a.push(quantise(sand));

            let pebbles = smooth_band(alt, 0.0, 0.28) * (1.0 - smoothstep(slope, 0.35, 0.85));
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
