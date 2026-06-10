//! Plugin-facing upload-job demo.
//!
//! Demonstrates `ItemFrameContext::jobs` with a small custom plugin that
//! synthesises a fresh heightmap on a background worker, then hands it to
//! the consumer to drive `TerrainPlugin`.
//!
//! The custom plugin paints nothing itself. Its only job is to spawn the
//! heightmap-generation work through `ctx.jobs.submit_cpu`, poll the
//! returned `JobId` each frame, and `take::<Vec<u16>>` the result when
//! the job lands. The consumer drops the resulting heightmap into a
//! `TerrainItem` so the rendered landscape changes whenever the user
//! clicks Run.

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use viewport_lib::plugin_api::{
    ItemFrameContext, ItemTypePlugin, PluginItemCollection, SharedBindings,
};
use viewport_lib::scene::material::ItemSettings;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, JobId, LightingSettings, OrbitCameraController,
    PickId, SceneFrame, ScrollUnits, UploadStatus, ViewportContext, ViewportEvent,
    ViewportRenderer,
};
use viewport_lib_terrain::{
    LAYER_COUNT, SplatmapData, TerrainCollection, TerrainItem, TerrainLayer, TerrainPlugin,
};

// ---------------------------------------------------------------------------
// Terrain dimensions and demo seed counter
// ---------------------------------------------------------------------------

const TERRAIN_DIM: u32 = 513;
const WORLD_SIZE: f32 = 64.0;
const HEIGHT_MIN: f32 = -1.5;
const HEIGHT_MAX: f32 = 6.5;

// ---------------------------------------------------------------------------
// Shared state between consumer and the demo plugin
// ---------------------------------------------------------------------------

#[derive(Clone)]
#[allow(dead_code)] // `Idle` is the implicit "no state" rendered when the
                   // shared slot is None; spelling it out clarifies the
                   // state machine.
enum JobUiState {
    Idle,
    Running { started: Instant },
    Done { samples: usize, took_ms: u64 },
    Failed { reason: String, took_ms: u64 },
}

impl JobUiState {
    fn status_line(&self) -> String {
        match self {
            JobUiState::Idle => "idle".to_string(),
            JobUiState::Running { started } => {
                format!("running ({} ms elapsed)", started.elapsed().as_millis())
            }
            JobUiState::Done { samples, took_ms } => format!(
                "done in {took_ms} ms ({} samples uploaded to TerrainPlugin)",
                samples
            ),
            JobUiState::Failed { reason, took_ms } => {
                format!("failed after {took_ms} ms: {reason}")
            }
        }
    }
}

#[derive(Default)]
struct SharedDemoState {
    ui: Mutex<Option<JobUiState>>,
    /// Generation counter the consumer monitors; bumped on every
    /// completed heightmap. The actual buffer is in `latest_heights`.
    generation: AtomicU64,
    latest_heights: Mutex<Option<Vec<u16>>>,
}

impl SharedDemoState {
    fn set_ui(&self, state: JobUiState) {
        *self.ui.lock().unwrap() = Some(state);
    }

    fn get_ui(&self) -> Option<JobUiState> {
        self.ui.lock().unwrap().clone()
    }

    fn publish_heights(&self, heights: Vec<u16>) {
        *self.latest_heights.lock().unwrap() = Some(heights);
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn take_heights_if(&self, last_seen_gen: u64) -> Option<(u64, Vec<u16>)> {
        let current = self.generation.load(Ordering::Relaxed);
        if current == last_seen_gen {
            return None;
        }
        let heights = self.latest_heights.lock().unwrap().take()?;
        Some((current, heights))
    }
}

// ---------------------------------------------------------------------------
// HeightGenPlugin: demonstrates ctx.jobs.submit_cpu / status / take
// ---------------------------------------------------------------------------

/// Plugin that does no rendering. Its only role is to spawn heightmap
/// generation work through `ItemFrameContext::jobs` and publish the
/// result for the consumer to wire into `TerrainPlugin`.
struct HeightGenPlugin {
    trigger: Arc<AtomicBool>,
    shared: Arc<SharedDemoState>,
    current_job: Option<JobId>,
    seed_counter: u64,
}

impl HeightGenPlugin {
    fn new(trigger: Arc<AtomicBool>, shared: Arc<SharedDemoState>) -> Self {
        Self {
            trigger,
            shared,
            current_job: None,
            seed_counter: 0,
        }
    }
}

/// Empty collection. The plugin needs `SceneFrame::submit_plugin_items`
/// to register an entry under its `type_name`, otherwise `prepare`
/// never runs. The collection carries no items because the plugin's
/// output flows through shared state, not through scene items.
struct EmptyCollection;

impl PluginItemCollection for EmptyCollection {
    fn len(&self) -> usize {
        0
    }

    fn item_settings(&self, _index: usize) -> &ItemSettings {
        unreachable!("collection is empty")
    }

    fn pick_id(&self, _index: usize) -> PickId {
        PickId::NONE
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl ItemTypePlugin for HeightGenPlugin {
    fn type_name(&self) -> &'static str {
        "demo:height_gen"
    }

    fn init_gpu(&mut self, _device: &wgpu::Device, _shared: &SharedBindings<'_>) {
        // No GPU state. The plugin only does CPU work via ctx.jobs.
    }

    fn prepare(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        ctx: &ItemFrameContext<'_>,
        _items: &dyn PluginItemCollection,
    ) -> Vec<wgpu::CommandBuffer> {
        // ---- 1. Trigger-edge handling: kick off a new job on click. ----
        if self.trigger.swap(false, Ordering::Relaxed) && self.current_job.is_none() {
            self.seed_counter = self.seed_counter.wrapping_add(1);
            let seed = self.seed_counter;
            self.shared.set_ui(JobUiState::Running {
                started: Instant::now(),
            });
            // Move the seed into the worker; the closure owns its inputs
            // because the `&Device` / `&Queue` references in `prepare`
            // are not `'static`.
            let id = ctx.jobs.submit_cpu(move || synthesise_heightmap(seed));
            self.current_job = Some(id);
        }

        // ---- 2. Poll the current job, take when Ready. ----
        if let Some(id) = self.current_job {
            match ctx.jobs.status(id) {
                UploadStatus::Pending { .. } => {
                    // Last "Running { started }" is still in the UI
                    // slot; the clock continues to tick.
                }
                UploadStatus::Ready => {
                    let elapsed = match self.shared.get_ui() {
                        Some(JobUiState::Running { started }) => {
                            started.elapsed().as_millis() as u64
                        }
                        _ => 0,
                    };
                    match ctx.jobs.take::<Vec<u16>>(id) {
                        Some(heights) => {
                            let samples = heights.len();
                            self.shared.publish_heights(heights);
                            self.shared.set_ui(JobUiState::Done {
                                samples,
                                took_ms: elapsed,
                            });
                        }
                        None => self.shared.set_ui(JobUiState::Failed {
                            reason: "result was the wrong type or already taken".to_string(),
                            took_ms: elapsed,
                        }),
                    }
                    self.current_job = None;
                }
                UploadStatus::Failed(e) => {
                    let elapsed = match self.shared.get_ui() {
                        Some(JobUiState::Running { started }) => {
                            started.elapsed().as_millis() as u64
                        }
                        _ => 0,
                    };
                    self.shared.set_ui(JobUiState::Failed {
                        reason: format!("{e}"),
                        took_ms: elapsed,
                    });
                    self.current_job = None;
                }
                UploadStatus::Unknown => {
                    self.current_job = None;
                }
            }
        }

        Vec::new()
    }

    // paint, paint_transparent, outline_mask, cull, cast_shadow_pass, pick
    // all default to no-ops. The plugin renders nothing.
}

/// CPU-heavy procedural function. Generates a heightmap by summing four
/// octaves of sine waves with a seed offset, then quantises to u16. The
/// work takes ~80-150 ms on a typical laptop CPU which is plenty to make
/// the worker-thread pattern observable: the orbit camera keeps spinning
/// while this runs.
fn synthesise_heightmap(seed: u64) -> Vec<u16> {
    let dim = TERRAIN_DIM as usize;
    let mut out = vec![0u16; dim * dim];
    let inv = 1.0 / (dim as f32 - 1.0);
    let phase = (seed as f32) * 0.91;

    for y in 0..dim {
        let v = y as f32 * inv;
        for x in 0..dim {
            let u = x as f32 * inv;
            let mut h = 0.0f32;
            let mut amp = 1.0f32;
            let mut freq = 1.5f32;
            for _ in 0..4 {
                h += amp
                    * ((u * freq * std::f32::consts::TAU + phase).sin()
                        + (v * freq * std::f32::consts::TAU - phase * 0.7).cos());
                amp *= 0.5;
                freq *= 2.0;
            }
            let unit = (h * 0.25 + 0.5).clamp(0.0, 1.0);
            out[y * dim + x] = (unit * u16::MAX as f32) as u16;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// eframe app
// ---------------------------------------------------------------------------

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib-terrain : Plugin job demo",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 760.0]),
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

            let trigger = Arc::new(AtomicBool::new(false));
            let shared = Arc::new(SharedDemoState::default());

            let mut renderer = ViewportRenderer::new(device, format);
            // Two plugins, registered together: the height generator
            // does the CPU work via ctx.jobs; TerrainPlugin renders the
            // resulting heightmap.
            renderer.with_item_type_plugin(
                device,
                Box::new(HeightGenPlugin::new(trigger.clone(), shared.clone())),
            );
            renderer.with_item_type_plugin(device, Box::new(TerrainPlugin::new()));

            rs.renderer.write().callback_resources.insert(renderer);

            Ok(Box::new(App::new(trigger, shared)))
        }),
    )
}

struct App {
    trigger: Arc<AtomicBool>,
    shared: Arc<SharedDemoState>,
    camera: Camera,
    controller: OrbitCameraController,
    terrain: TerrainItem,
    last_heights_gen: u64,
    cursor_viewport: Option<glam::Vec2>,
}

impl App {
    fn new(trigger: Arc<AtomicBool>, shared: Arc<SharedDemoState>) -> Self {
        // Start with an empty terrain; the first Run click fills it.
        let dim = TERRAIN_DIM as usize;
        let flat = vec![0u16; dim * dim];

        let mut terrain = TerrainItem::new(flat, [TERRAIN_DIM, TERRAIN_DIM]);
        terrain.world_size = [WORLD_SIZE, WORLD_SIZE];
        terrain.height_range = [HEIGHT_MIN, HEIGHT_MAX];
        terrain.origin = glam::Vec3::new(-WORLD_SIZE * 0.5, -WORLD_SIZE * 0.5, 0.0);
        terrain.surface_layers = make_layers();
        // A 1x1 splatmap pointing at layer 0; uniform colouring keeps the
        // demo focused on the heightmap (which is what changes between
        // Run clicks).
        terrain.splatmaps = [SplatmapData::solid_layer0(), SplatmapData::empty()];

        let camera = Camera {
            center: glam::Vec3::ZERO,
            distance: 90.0,
            orientation: glam::Quat::from_rotation_z(0.6)
                * glam::Quat::from_rotation_x(1.05),
            ..Camera::default()
        };

        Self {
            trigger,
            shared,
            camera,
            controller: OrbitCameraController::viewport_primitives(),
            terrain,
            last_heights_gen: 0,
            cursor_viewport: None,
        }
    }

    /// Swap the terrain's heightmap whenever the plugin publishes a new
    /// one.
    fn pump_heightmap(&mut self) {
        if let Some((new_gen, heights)) = self.shared.take_heights_if(self.last_heights_gen) {
            self.terrain
                .replace_heightmap(heights, [TERRAIN_DIM, TERRAIN_DIM]);
            self.last_heights_gen = new_gen;
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keep the UI repainting so the running-clock ticks and the
        // orbit camera stays smooth on its own.
        ctx.request_repaint_after(Duration::from_millis(33));

        self.pump_heightmap();

        egui::SidePanel::right("controls").show(ctx, |ui| {
            ui.heading("Plugin job demo");
            ui.add_space(4.0);
            ui.label(
                "Click Run to fire a heavy CPU job through \
                 ItemFrameContext::jobs. The plugin's prepare() submits, \
                 polls each frame, and takes the typed result once Ready. \
                 The resulting heightmap is pushed into TerrainPlugin and \
                 the landscape redraws.",
            );
            ui.add_space(8.0);

            if ui.button("Run").clicked() {
                self.trigger.store(true, Ordering::Relaxed);
            }
            ui.add_space(8.0);

            let status = self
                .shared
                .get_ui()
                .map(|s| s.status_line())
                .unwrap_or_else(|| "idle (click Run to generate a heightmap)".to_string());
            ui.label(format!("Status: {status}"));

            ui.add_space(12.0);
            ui.separator();
            ui.label("Plugin trait: ItemTypePlugin");
            ui.label("API:  ctx.jobs.submit_cpu(work) -> JobId");
            ui.label("      ctx.jobs.status(id) -> UploadStatus");
            ui.label("      ctx.jobs.take::<T>(id) -> Option<T>");
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

            let w = rect.width().max(1.0);
            let h = rect.height().max(1.0);
            self.controller.apply_to_camera(&mut self.camera);
            self.camera.set_aspect_ratio(w, h);

            let mut camera_frame = CameraFrame::from_camera(&self.camera, [w, h]);
            camera_frame.pixels_per_point = ctx.pixels_per_point();

            let mut frame_data = FrameData::new(camera_frame, SceneFrame::default());
            frame_data.effects.lighting = LightingSettings::default();

            // Empty collection drives the demo plugin's prepare.
            frame_data
                .scene
                .submit_plugin_items("demo:height_gen", EmptyCollection);
            // The real terrain item drives the renderer.
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

/// Single-layer-friendly defaults. The demo only uses layer 0 (grass),
/// so the other layers are present but unused.
fn make_layers() -> [TerrainLayer; LAYER_COUNT] {
    let neutral = TerrainLayer {
        albedo: [0.5, 0.5, 0.5],
        metallic: 0.0,
        roughness: 0.85,
        height_bias: 0.0,
    };
    let mut layers = [neutral; LAYER_COUNT];
    layers[0] = TerrainLayer {
        albedo: [0.32, 0.50, 0.22],
        metallic: 0.0,
        roughness: 0.90,
        height_bias: 0.0,
    };
    layers
}

// ---------------------------------------------------------------------------
// Paint callback bridging egui_wgpu to the viewport renderer
// ---------------------------------------------------------------------------

struct ViewportCallback {
    frame: FrameData,
}

impl eframe::egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(renderer) = callback_resources.get_mut::<ViewportRenderer>() {
            return renderer.pass().prepare(device, queue, &self.frame);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = callback_resources.get::<ViewportRenderer>() {
            renderer.pass_view().paint(render_pass, &self.frame);
        }
    }
}
