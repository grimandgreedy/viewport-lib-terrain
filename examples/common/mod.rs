//! Shared by the eframe examples: draw the viewport into an offscreen texture and
//! show it in egui as an image.
//!
//! The renderer outputs linear light and relies on its target format for the
//! final sRGB encode. Drawing straight into egui's pass (a paint callback) puts
//! that output through egui's own colour handling as well, so the viewport reads
//! encoded twice. `OffscreenViewportTarget` renders through an sRGB view, so the
//! encode happens once on write, and hands egui a non-sRGB view so egui samples
//! the already-encoded bytes verbatim. This is the route viewport-lib's own
//! `eframe_minimal` example takes.
//!
//! Build the renderer with [`render_format`] so its pipelines match the target.

#[allow(dead_code)]
pub mod assets;

use eframe::egui;
use viewport_lib::{FrameData, OffscreenViewportTarget, ViewportRenderer};

/// The format to create the `ViewportRenderer` with.
pub fn render_format(surface_format: eframe::wgpu::TextureFormat) -> eframe::wgpu::TextureFormat {
    OffscreenViewportTarget::render_format(surface_format)
}

/// The offscreen target and its egui registration, kept in egui's callback
/// resources next to the renderer so the app does not carry it.
struct Slot {
    target: OffscreenViewportTarget,
    id: egui::TextureId,
}

/// Render `frame_data` into `rect` of `ui`.
pub fn show(ui: &mut egui::Ui, frame: &eframe::Frame, rect: egui::Rect, frame_data: &FrameData) {
    let rs = frame.wgpu_render_state().expect("wgpu backend required");
    let ppp = ui.ctx().pixels_per_point();
    let size = [
        (rect.width() * ppp).round().max(1.0) as u32,
        (rect.height() * ppp).round().max(1.0) as u32,
    ];

    let mut guard = rs.renderer.write();
    let stale = guard
        .callback_resources
        .get::<Slot>()
        .map_or(true, |s| s.target.size() != size);
    if stale {
        if let Some(old) = guard.callback_resources.remove::<Slot>() {
            guard.free_texture(&old.id);
        }
        let target = OffscreenViewportTarget::new(&rs.device, rs.target_format, size);
        let id = guard.register_native_texture(
            &rs.device,
            target.sample_view(),
            eframe::wgpu::FilterMode::Linear,
        );
        guard.callback_resources.insert(Slot { target, id });
    }

    // Take the slot out so the renderer can be borrowed alongside it.
    let slot = guard.callback_resources.remove::<Slot>().unwrap();
    if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
        let cmd = renderer
            .owned()
            .render(&rs.device, &rs.queue, slot.target.render_view(), frame_data);
        rs.queue.submit(std::iter::once(cmd));
    }
    let id = slot.id;
    guard.callback_resources.insert(slot);
    drop(guard);

    ui.painter().image(
        id,
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
}
