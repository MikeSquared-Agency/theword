/// Floating overlay GUI for theword.
///
/// A small always-on-top window with a drag handle, close button, and a
/// central microphone button. Clicking the mic button records one utterance
/// (VAD-terminated) and dispatches the result exactly as `theword dictate`
/// does. The button turns red while recording and grey while processing.
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui::{self, Color32, FontId, Rounding, Stroke, Vec2};

// ── Recording state ──────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
enum State {
    Idle,
    Recording,
    Processing,
}

// ── App ──────────────────────────────────────────────────────────────────────

struct Overlay {
    state: Arc<Mutex<State>>,
    engine: Arc<crate::dictation::DictationEngine>,
    rt: tokio::runtime::Handle,
}

impl eframe::App for Overlay {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let state = self.state.lock().unwrap().clone();

        let panel_frame = egui::Frame {
            fill: Color32::WHITE,
            rounding: Rounding::same(12.0),
            stroke: Stroke::new(1.0, Color32::from_gray(210)),
            inner_margin: egui::Margin::same(0.0),
            ..Default::default()
        };

        egui::CentralPanel::default()
            .frame(panel_frame)
            .show(ctx, |ui| {
                // ── Title bar ────────────────────────────────────
                ui.horizontal(|ui| {
                    // Drag-sensitive area (most of the bar width)
                    let drag = ui.allocate_response(
                        Vec2::new(ui.available_width() - 30.0, 22.0),
                        egui::Sense::click_and_drag(),
                    );
                    if drag.dragged() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                    }
                    // Centred drag-handle indicator painted over the rect
                    ui.painter().text(
                        drag.rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "────",
                        FontId::proportional(9.0),
                        Color32::from_gray(190),
                    );

                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("✕")
                                    .size(11.0)
                                    .color(Color32::from_gray(140)),
                            )
                            .frame(false)
                            .min_size(Vec2::splat(22.0)),
                        )
                        .clicked()
                    {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.add(egui::Separator::default().spacing(2.0));

                // ── Button row ───────────────────────────────────
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.add_space(22.0);

                    // Settings
                    ui.add(
                        egui::Button::new(
                            egui::RichText::new("⚙")
                                .size(20.0)
                                .color(Color32::from_gray(130)),
                        )
                        .frame(false)
                        .min_size(Vec2::splat(30.0)),
                    );

                    ui.add_space(14.0);

                    // Mic
                    let (bg, icon): (Color32, &str) = match state {
                        State::Idle       => (Color32::from_rgb(66, 133, 244), "🎤"),
                        State::Recording  => (Color32::from_rgb(219, 68, 55),  "🎤"),
                        State::Processing => (Color32::from_gray(170),          "⏳"),
                    };

                    let mic = egui::Button::new(
                        egui::RichText::new(icon)
                            .size(22.0)
                            .color(Color32::WHITE),
                    )
                    .fill(bg)
                    .rounding(Rounding::same(24.0))
                    .min_size(Vec2::splat(48.0));

                    let cur = self.state.lock().unwrap().clone();
                    if cur == State::Idle {
                        if ui.add(mic).clicked() {
                            let state_ref = self.state.clone();
                            let engine    = self.engine.clone();
                            let repaint   = ctx.clone();
                            self.rt.spawn(async move {
                                *state_ref.lock().unwrap() = State::Recording;
                                repaint.request_repaint();
                                // dictate_once blocks (async) until VAD silence
                                let _ = engine.dictate_once(Duration::from_secs(30)).await;
                                *state_ref.lock().unwrap() = State::Idle;
                                repaint.request_repaint();
                            });
                        }
                    } else {
                        ui.add_enabled(false, mic);
                    }

                    ui.add_space(14.0);

                    // Help
                    ui.add(
                        egui::Button::new(
                            egui::RichText::new("?")
                                .size(14.0)
                                .color(Color32::from_gray(130)),
                        )
                        .stroke(Stroke::new(1.5, Color32::from_gray(200)))
                        .rounding(Rounding::same(12.0))
                        .min_size(Vec2::splat(28.0)),
                    );
                });

                ui.add_space(8.0);
            });

        // Keep repainting while active so the button colour stays current
        let s = self.state.lock().unwrap().clone();
        if matches!(s, State::Recording | State::Processing) {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

pub fn run(
    engine: Arc<crate::dictation::DictationEngine>,
    rt: tokio::runtime::Handle,
) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([240.0, 96.0])
            .with_always_on_top()
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(false),
        ..Default::default()
    };

    eframe::run_native(
        "theword",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(Overlay {
                state: Arc::new(Mutex::new(State::Idle)),
                engine,
                rt,
            }))
        }),
    )
}
