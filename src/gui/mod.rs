/// Floating overlay GUI for theword.
///
/// A small always-on-top window with a drag handle, close button, and a
/// central microphone button. Clicking the mic button records one utterance
/// (VAD-terminated) and dispatches the result exactly as `theword dictate`
/// does. The gear icon opens a settings window where every DictationConfig
/// field can be edited and saved to `~/.theword/config.toml`. Changes to
/// VAD, output method, rewrite settings, etc. take effect on the *next*
/// dictation call without requiring a restart. Whisper model/language
/// changes require restart because the model is loaded at startup.
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui::{self, Color32, FontId, Rounding, Stroke, Vec2};

use crate::config::{DictationConfig, OutputMethod, VadMode, WhisperModel};

// ── Recording state ──────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
enum RecordState {
    Idle,
    Recording,
    Processing,
}

// ── App ──────────────────────────────────────────────────────────────────────

struct Overlay {
    record_state: Arc<Mutex<RecordState>>,
    engine: Arc<crate::dictation::DictationEngine>,
    /// Shared with the engine — changes here are picked up on the next dictation call.
    config: Arc<Mutex<DictationConfig>>,
    rt: tokio::runtime::Handle,
    show_settings: bool,
    /// Edited copy; written back to `config` and disk only on Save.
    pending: DictationConfig,
}

impl eframe::App for Overlay {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.draw_overlay(ctx);
        if self.show_settings {
            self.draw_settings(ctx);
        }

        let rs = self.record_state.lock().unwrap().clone();
        if matches!(rs, RecordState::Recording | RecordState::Processing) {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
    }
}

impl Overlay {
    // ── Main overlay window ──────────────────────────────────────────────────

    fn draw_overlay(&mut self, ctx: &egui::Context) {
        let rs = self.record_state.lock().unwrap().clone();

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
                    let drag = ui.allocate_response(
                        Vec2::new(ui.available_width() - 30.0, 22.0),
                        egui::Sense::click_and_drag(),
                    );
                    if drag.dragged() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                    }
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
                                egui::RichText::new("✕").size(11.0).color(Color32::from_gray(140)),
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

                    // Gear
                    let gear_color = if self.show_settings {
                        Color32::from_rgb(66, 133, 244)
                    } else {
                        Color32::from_gray(130)
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("⚙").size(20.0).color(gear_color),
                            )
                            .frame(false)
                            .min_size(Vec2::splat(30.0)),
                        )
                        .clicked()
                    {
                        self.show_settings = !self.show_settings;
                        if self.show_settings {
                            // Snapshot current config into pending for editing
                            self.pending = self.config.lock().unwrap().clone();
                        }
                    }

                    ui.add_space(14.0);

                    // Mic
                    let (bg, icon): (Color32, &str) = match rs {
                        RecordState::Idle       => (Color32::from_rgb(66, 133, 244), "🎤"),
                        RecordState::Recording  => (Color32::from_rgb(219, 68, 55),  "🎤"),
                        RecordState::Processing => (Color32::from_gray(170),          "⏳"),
                    };

                    let mic = egui::Button::new(
                        egui::RichText::new(icon).size(22.0).color(Color32::WHITE),
                    )
                    .fill(bg)
                    .rounding(Rounding::same(24.0))
                    .min_size(Vec2::splat(48.0));

                    let cur = self.record_state.lock().unwrap().clone();
                    if cur == RecordState::Idle {
                        if ui.add(mic).clicked() {
                            let state_ref = self.record_state.clone();
                            let engine    = self.engine.clone();
                            let repaint   = ctx.clone();
                            self.rt.spawn(async move {
                                *state_ref.lock().unwrap() = RecordState::Recording;
                                repaint.request_repaint();
                                let _ = engine.dictate_once(Duration::from_secs(30)).await;
                                *state_ref.lock().unwrap() = RecordState::Idle;
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
                            egui::RichText::new("?").size(14.0).color(Color32::from_gray(130)),
                        )
                        .stroke(Stroke::new(1.5, Color32::from_gray(200)))
                        .rounding(Rounding::same(12.0))
                        .min_size(Vec2::splat(28.0)),
                    );
                });

                ui.add_space(8.0);
            });
    }

    // ── Settings window ──────────────────────────────────────────────────────

    fn draw_settings(&mut self, ctx: &egui::Context) {
        let mut open = self.show_settings;

        egui::Window::new("Settings")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .default_width(310.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // ── Whisper ──────────────────────────────────
                    ui.label(egui::RichText::new("Whisper").strong());
                    egui::Grid::new("whisper_grid")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Model");
                            egui::ComboBox::from_id_source("whisper_model")
                                .selected_text(self.pending.whisper_model.label())
                                .show_ui(ui, |ui| {
                                    for m in WhisperModel::all() {
                                        let label = m.label();
                                        ui.selectable_value(
                                            &mut self.pending.whisper_model,
                                            m,
                                            label,
                                        );
                                    }
                                });
                            ui.end_row();

                            ui.label("Language");
                            let mut lang = self.pending.language.clone().unwrap_or_default();
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut lang)
                                    .hint_text("en  (blank = auto)")
                                    .desired_width(120.0),
                            );
                            if resp.changed() {
                                self.pending.language =
                                    if lang.trim().is_empty() { None } else { Some(lang.trim().to_string()) };
                            }
                            ui.end_row();
                        });

                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("⚠ Model/language changes take effect after restart.")
                            .small()
                            .color(Color32::from_gray(150)),
                    );

                    ui.separator();

                    // ── VAD ──────────────────────────────────────
                    ui.label(egui::RichText::new("Voice Detection").strong());
                    egui::Grid::new("vad_grid")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("VAD mode");
                            egui::ComboBox::from_id_source("vad_mode")
                                .selected_text(self.pending.vad_mode.label())
                                .show_ui(ui, |ui| {
                                    for m in VadMode::all() {
                                        let label = m.label();
                                        ui.selectable_value(&mut self.pending.vad_mode, m, label);
                                    }
                                });
                            ui.end_row();

                            ui.label("Silence (ms)");
                            let mut silence = self.pending.silence_threshold_ms as f32;
                            if ui
                                .add(egui::Slider::new(&mut silence, 200.0..=2000.0).step_by(50.0))
                                .changed()
                            {
                                self.pending.silence_threshold_ms = silence as u64;
                            }
                            ui.end_row();

                            ui.label("Min speech (ms)");
                            let mut min_sp = self.pending.min_speech_ms as f32;
                            if ui
                                .add(egui::Slider::new(&mut min_sp, 100.0..=1000.0).step_by(50.0))
                                .changed()
                            {
                                self.pending.min_speech_ms = min_sp as u64;
                            }
                            ui.end_row();
                        });

                    ui.separator();

                    // ── Output ───────────────────────────────────
                    ui.label(egui::RichText::new("Output").strong());
                    ui.horizontal(|ui| {
                        for method in OutputMethod::all() {
                            ui.radio_value(&mut self.pending.output_method, method, method.label());
                        }
                    });
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.pending.rewrite_enabled, "LLM rewrite");
                        ui.add_enabled(
                            self.pending.rewrite_enabled,
                            egui::TextEdit::singleline(&mut self.pending.rewrite_model)
                                .hint_text("ollama model")
                                .desired_width(140.0),
                        );
                    });

                    ui.separator();

                    // ── Memory ───────────────────────────────────
                    ui.label(egui::RichText::new("Memory").strong());
                    ui.checkbox(&mut self.pending.learn_corrections, "Learn vocabulary corrections");

                    ui.separator();

                    // ── Hotkey (listen mode) ──────────────────────
                    ui.label(egui::RichText::new("Hotkey  (theword listen)").strong());
                    egui::Grid::new("hotkey_grid")
                        .num_columns(2)
                        .spacing([8.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Key");
                            let keys = ["AltGr", "Alt", "F9", "F10", "F11", "F12"];
                            egui::ComboBox::from_id_source("hotkey_key")
                                .selected_text(&self.pending.hotkey.key)
                                .show_ui(ui, |ui| {
                                    for k in &keys {
                                        ui.selectable_value(
                                            &mut self.pending.hotkey.key,
                                            k.to_string(),
                                            *k,
                                        );
                                    }
                                });
                            ui.end_row();

                            ui.label("Mode");
                            ui.horizontal(|ui| {
                                ui.radio_value(&mut self.pending.hotkey.hold_to_talk, true,  "Hold to talk");
                                ui.radio_value(&mut self.pending.hotkey.hold_to_talk, false, "Toggle");
                            });
                            ui.end_row();
                        });

                    ui.add_space(8.0);

                    // ── Save / Cancel ─────────────────────────────
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            self.show_settings = false;
                        }
                        ui.add_space(8.0);
                        let save_btn = egui::Button::new(
                            egui::RichText::new("Save").color(Color32::WHITE),
                        )
                        .fill(Color32::from_rgb(66, 133, 244))
                        .min_size(Vec2::new(60.0, 24.0));

                        if ui.add(save_btn).clicked() {
                            // Write through to the shared config (engine picks it up next call)
                            *self.config.lock().unwrap() = self.pending.clone();
                            // Persist to disk
                            if let Err(e) = crate::config::save_dictation_config(&self.pending) {
                                eprintln!("Failed to save config: {e}");
                            }
                            self.show_settings = false;
                        }
                    });
                });
            });

        self.show_settings = open;
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

pub fn run(
    engine: Arc<crate::dictation::DictationEngine>,
    config: Arc<Mutex<DictationConfig>>,
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
            let pending = config.lock().unwrap().clone();
            Ok(Box::new(Overlay {
                record_state: Arc::new(Mutex::new(RecordState::Idle)),
                engine,
                config,
                rt,
                show_settings: false,
                pending,
            }))
        }),
    )
}
