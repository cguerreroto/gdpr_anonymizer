#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example

mod dataset;
mod gui;
mod watcher;

use eframe::egui;
use rfd::FileDialog;
use std::{
    path::Path,
    sync::{Arc, RwLock},
};

use crate::{
    dataset::{DirectoryWatcher, ImagePurpose},
    gui::{SavePromptResult, show_save_prompt},
};

enum PendingAction {
    SaveOnly,
    Navigate(dataset::ImageReference),
}

#[derive(Clone, Copy)]
enum NavDirection {
    Previous,
    Next,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppMainTab {
    Annotation,
    Help,
}

/// Multiplier on top of fit-to-viewport scale (1.0 = legacy behavior).
const IMAGE_VIEW_ZOOM_MIN: f32 = 0.25;
const IMAGE_VIEW_ZOOM_MAX: f32 = 8.0;
/// `zoom *= exp(-scroll_y * factor)` -> tuned for trackpad and mouse wheel sensitivity.
const IMAGE_VIEW_ZOOM_SCROLL_EXP_FACTOR: f32 = 0.002;

fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 880.0]),
        ..Default::default()
    };
    eframe::run_native(
        "YOLO Segmentator",
        options,
        Box::new(|cc| {
            // This gives us image support:
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(MyApp::new(cc.egui_ctx.clone())))
        }),
    )
}

struct MyApp {
    dataset: Arc<RwLock<Option<dataset::Dataset>>>,
    worker: DirectoryWatcher,
    status_message: Option<String>,
    current_image: Option<ActiveImage>,
    /// Extra zoom on top of fit-to-viewport scale (wheel); not persisted.
    image_view_zoom: f32,
    /// Screen-space offset of the image center from the viewport center.
    image_view_pan: egui::Vec2,
    selected_segment_id: Option<usize>,
    /// Undo stack for polygon edits of `polygon_history_segment_id` only.
    polygon_undo_stack: Vec<Vec<dataset::SegmentPoint>>,
    polygon_redo_stack: Vec<Vec<dataset::SegmentPoint>>,
    polygon_history_segment_id: Option<usize>,
    data_dirty: bool,
    show_save_prompt: bool,
    pending_action: Option<PendingAction>,
    main_tab: AppMainTab,
}

impl MyApp {
    fn new(ctx: egui::Context) -> Self {
        let dataset = Arc::new(RwLock::new(None));
        let worker = DirectoryWatcher::new(dataset.clone(), ctx);
        Self {
            dataset,
            worker,
            status_message: None,
            current_image: None,
            image_view_zoom: 1.0,
            image_view_pan: egui::Vec2::ZERO,
            selected_segment_id: None,
            polygon_undo_stack: Vec::new(),
            polygon_redo_stack: Vec::new(),
            polygon_history_segment_id: None,
            data_dirty: false,
            show_save_prompt: false,
            pending_action: None,
            main_tab: AppMainTab::Annotation,
        }
    }
}

struct ActiveImage {
    reference: dataset::ImageReference,
    texture: egui::TextureHandle,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            self.toolbar(ui);
        });

        if self.main_tab == AppMainTab::Annotation && self.has_dataset() && !self.show_save_prompt {
            self.handle_annotation_keyboard_shortcuts(ctx);
        }

        egui::TopBottomPanel::bottom("status_bar")
            .resizable(false)
            .show(ctx, |ui| {
                if let Some(status) = &self.status_message {
                    ui.horizontal(|ui| {
                        ui.label(status);
                    });
                } else {
                    ui.horizontal(|ui| {
                        ui.label("Ready");
                    });
                }
            });

        match self.main_tab {
            AppMainTab::Help => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    Self::help_panel(ui);
                });
            }
            AppMainTab::Annotation if self.has_dataset() => {
                egui::SidePanel::left("class_panel")
                    .resizable(true)
                    .default_width(220.0)
                    .show(ctx, |ui| self.class_panel(ui));

                egui::SidePanel::right("segment_panel")
                    .resizable(true)
                    .default_width(260.0)
                    .show(ctx, |ui| self.segment_panel(ui));

                egui::CentralPanel::default().show(ctx, |ui| {
                    self.center_panel(ui);
                });
            }
            AppMainTab::Annotation => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.heading("YOLO Segmentator");
                    ui.add_space(8.0);
                    ui.label("Open or create a dataset using the toolbar to begin annotating.");
                });
            }
        }

        if self.show_save_prompt
            && let Some(choice) = show_save_prompt(ctx, "Save Changes?")
        {
            match choice {
                SavePromptResult::Yes => match self.save_current_image_segments() {
                    Ok(()) => {
                        self.data_dirty = false;
                        self.show_save_prompt = false;
                        self.execute_pending_action(ctx);
                    }
                    Err(err) => {
                        self.status_message = Some(err);
                    }
                },
                SavePromptResult::No => {
                    self.data_dirty = false;
                    self.show_save_prompt = false;
                    self.execute_pending_action(ctx);
                }
            }
        }
    }
}

impl MyApp {
    fn handle_annotation_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        let mut nav_request = None;
        let mut recenter_view = false;
        let mut undo_polygon = false;
        let mut redo_polygon = false;
        let mut new_segment_key = false;
        let mut save_key = false;
        ctx.input(|input| {
            if input.key_pressed(egui::Key::ArrowLeft) {
                nav_request = Some(NavDirection::Previous);
            } else if input.key_pressed(egui::Key::ArrowRight) {
                nav_request = Some(NavDirection::Next);
            } else if input.key_pressed(egui::Key::R) {
                recenter_view = true;
            } else if input.key_pressed(egui::Key::Z) {
                undo_polygon = true;
            } else if input.key_pressed(egui::Key::Y) {
                redo_polygon = true;
            } else if input.key_pressed(egui::Key::N) {
                new_segment_key = true;
            } else if input.key_pressed(egui::Key::S) {
                save_key = true;
            }
        });
        if let Some(dir) = nav_request {
            self.navigate_image(dir, ctx);
        }
        if recenter_view && self.current_image.is_some() && !ctx.wants_keyboard_input() {
            self.recenter_image_view();
        }
        if !ctx.wants_keyboard_input() && self.current_image.is_some() {
            if undo_polygon {
                self.undo_selected_segment_polygon();
            } else if redo_polygon {
                self.redo_selected_segment_polygon();
            }
        }
        if !ctx.wants_keyboard_input() {
            if new_segment_key {
                self.create_new_segment_for_current_image();
            }
            if save_key {
                self.handle_save_dataset();
            }
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.main_tab, AppMainTab::Annotation, "Annotation");
                ui.selectable_value(&mut self.main_tab, AppMainTab::Help, "Help");
            });
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                if self.has_dataset() {
                    if ui.button("Close Dataset").clicked() {
                        self.handle_close_dataset();
                    }
                } else if ui.button("New Dataset").clicked() {
                    self.handle_new_dataset();
                }

                if ui.button("Open Directory").clicked() {
                    self.handle_open_dataset();
                }

                if ui.button("Save").clicked() {
                    self.handle_save_dataset();
                }

                if ui.button("Previous").clicked() {
                    self.navigate_image(NavDirection::Previous, ui.ctx());
                }
                if ui.button("Next").clicked() {
                    self.navigate_image(NavDirection::Next, ui.ctx());
                }
            });
        });
    }

    fn help_panel(ui: &mut egui::Ui) {
        ui.heading("Help");
        ui.separator();
        ui.label(
            "This page lists keyboard shortcuts and related controls for the YOLO Segmentator.",
        );
        ui.add_space(12.0);

        ui.heading("Keyboard shortcuts");
        ui.add_space(6.0);
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label(
                egui::RichText::new("General").strong(),
            );
            ui.add_space(4.0);
            ui.label("These shortcuts run only on the Annotation tab when a dataset is open and no save confirmation dialog is showing. They do not run while focus is in a text field, so you can type class names and paths normally. Letter keys R, Z, Y, N, and S all follow these rules.");
            ui.add_space(10.0);

            ui.label(egui::RichText::new("Arrow Left").strong());
            ui.label("Moves to the previous image in the dataset order.");
            ui.add_space(8.0);

            ui.label(egui::RichText::new("Arrow Right").strong());
            ui.label("Moves to the next image in the dataset order.");
            ui.add_space(8.0);

            ui.label(egui::RichText::new("R").strong());
            ui.label(
                "Recenters the image in the main panel when an image is loaded. Zoom and pan return to the same state as right after loading. The segment you were editing stays selected.",
            );
            ui.add_space(8.0);

            ui.label(egui::RichText::new("Z").strong());
            ui.label(
                "Undoes the last point you added to the polygon for the segment that is currently selected. Requires a loaded image and a selected segment. Only that segment’s polygon history is affected.",
            );
            ui.add_space(8.0);

            ui.label(egui::RichText::new("Y").strong());
            ui.label(
                "Redoes a polygon point for the currently selected segment after you used undo. The same requirements apply as for undo.",
            );
            ui.add_space(8.0);

            ui.label(egui::RichText::new("N").strong());
            ui.label(
                "Creates a new segment on the currently loaded image and selects it for editing, matching the New Segment control in the Segments panel. A loaded image is required, along with the same conditions as the other letter shortcuts.",
            );
            ui.add_space(8.0);

            ui.label(egui::RichText::new("S").strong());
            ui.label(
                "Saves annotations for the current image, matching the Save control in the toolbar. If there are unsaved changes, the save confirmation dialog appears first.",
            );
            ui.add_space(12.0);

            ui.label(egui::RichText::new("Image view").strong());
            ui.add_space(4.0);
            ui.label(
                "With the pointer over the image in the main panel, scroll up or down to zoom in or out. The view stays anchored under the pointer while zoom changes.",
            );
        });
    }

    fn class_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Dataset");
        let dataset_overview = {
            let guard = self.dataset.read().expect("dataset lock poisoned");
            guard
                .as_ref()
                .map(|dataset| (dataset.root.clone(), dataset.name(), dataset.images_view()))
        };
        let Some((root_path, dataset_name, images)) = dataset_overview else {
            ui.label("(No dataset open)");
            return;
        };
        ui.label(format!("Path: {}", root_path.display()));
        ui.label(format!("Name: {dataset_name}"));
        ui.separator();

        ui.heading("Classes");
        ui.separator();

        let mut dataset_guard = self.dataset.write().expect("dataset lock poisoned");
        let Some(dataset) = dataset_guard.as_mut() else {
            ui.label("Open or create a dataset to edit classes.");
            return;
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            if dataset.classes.is_empty() {
                ui.label("No classes defined yet");
            }
            egui::Grid::new("class_grid")
                .striped(true)
                .num_columns(2)
                .show(ui, |ui| {
                    for entry in &mut dataset.classes {
                        ui.label(format!("ID {:02}", entry.id));
                        ui.text_edit_singleline(&mut entry.name);
                        ui.end_row();
                    }
                });
        });

        ui.separator();
        ui.vertical_centered(|ui| {
            ui.label("Add New Class");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut dataset.new_class_name);
                let name_ready = !dataset.new_class_name.trim().is_empty();
                if ui
                    .add_enabled(name_ready, egui::Button::new("Add"))
                    .clicked()
                    && name_ready
                {
                    let new_id = dataset.next_class_id();
                    dataset.classes.push(dataset::ClassEntry {
                        id: new_id,
                        name: dataset.new_class_name.trim().to_owned(),
                    });
                    dataset.new_class_name.clear();
                }
            });
        });
        drop(dataset_guard);

        ui.separator();
        ui.heading("Images");
        egui::ScrollArea::vertical()
            .id_salt("images_scroll")
            .show(ui, |ui| {
                for purpose in ImagePurpose::ALL {
                    let title = purpose.display_name();
                    let list = images.list(purpose);
                    ui.label(egui::RichText::new(title).strong());
                    if list.is_empty() {
                        ui.label("  (no images)");
                    } else {
                        egui::Grid::new(format!("image_grid_{title}"))
                            .striped(true)
                            .num_columns(3)
                            .show(ui, |ui| {
                                for entry in list {
                                    ui.label(entry.path.display().to_string());
                                    ui.label(if entry.has_labels { "" } else { "(new)" });
                                    if ui.small_button("Load").clicked() {
                                        self.load_image(ui.ctx(), purpose, &entry.path);
                                    }
                                    ui.end_row();
                                }
                            });
                    }
                    ui.add_space(4.0);
                }
            });
    }

    fn handle_new_dataset(&mut self) {
        let Some(selected_dir) = FileDialog::new()
            .set_title("Create or select dataset folder")
            .pick_folder()
        else {
            self.status_message = Some("Dataset creation canceled".to_owned());
            return;
        };

        let dataset = dataset::Dataset::new(selected_dir);
        let dataset_root = dataset.root.clone();
        match dataset.initialize_on_disk() {
            Ok(()) => match dataset::Dataset::load(dataset_root.clone()) {
                Ok(loaded) => {
                    let dataset_path = dataset_root.display().to_string();
                    {
                        let mut guard = self.dataset.write().expect("dataset lock poisoned");
                        *guard = Some(loaded);
                    }
                    self.current_image = None;
                    self.selected_segment_id = None;
                    self.clear_polygon_edit_history();
                    self.data_dirty = false;
                    self.show_save_prompt = false;
                    self.worker.notify_dataset_changed();
                    self.status_message = Some(format!(
                        "New dataset created at {dataset_path} and ready for editing"
                    ));
                }
                Err(err) => {
                    self.status_message = Some(format!(
                        "Created dataset but failed to reload metadata: {err}"
                    ));
                }
            },
            Err(err) => {
                self.status_message = Some(format!("Failed to create dataset: {err}"));
            }
        }
    }

    fn handle_open_dataset(&mut self) {
        let Some(selected_dir) = FileDialog::new()
            .set_title("Select dataset folder")
            .pick_folder()
        else {
            self.status_message = Some("Dataset loading canceled".to_owned());
            return;
        };

        match dataset::Dataset::load(selected_dir.clone()) {
            Ok(dataset) => {
                let dataset_path = selected_dir.display().to_string();
                {
                    let mut guard = self.dataset.write().expect("dataset lock poisoned");
                    *guard = Some(dataset);
                }
                self.current_image = None;
                self.selected_segment_id = None;
                self.clear_polygon_edit_history();
                self.data_dirty = false;
                self.show_save_prompt = false;
                self.worker.notify_dataset_changed();
                self.status_message = Some(format!("Loaded dataset from {dataset_path}"));
            }
            Err(err) => {
                self.status_message = Some(format!("Failed to load dataset: {err}"));
            }
        }
    }

    fn handle_close_dataset(&mut self) {
        let closed_path = {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            guard
                .take()
                .map(|dataset| dataset.root.display().to_string())
        };

        if let Some(path) = closed_path {
            self.status_message = Some(format!("Closed dataset {path}"));
        } else {
            self.status_message = Some("No dataset open".to_owned());
        }
        self.current_image = None;
        self.selected_segment_id = None;
        self.image_view_zoom = 1.0;
        self.image_view_pan = egui::Vec2::ZERO;
        self.clear_polygon_edit_history();
        self.data_dirty = false;
        self.show_save_prompt = false;
        self.worker.notify_dataset_changed();
    }

    fn handle_save_dataset(&mut self) {
        if self.data_dirty {
            self.pending_action = Some(PendingAction::SaveOnly);
            self.show_save_prompt = true;
        } else if let Err(err) = self.save_current_image_segments() {
            self.status_message = Some(err);
        }
    }

    /// Pushes a new segment onto `loaded_image`, marks it dirty, returns the new segment id.
    fn push_new_segment_core(
        loaded_image: &mut dataset::LoadedImage,
        default_class: usize,
    ) -> usize {
        let next_id = loaded_image.next_segment_id();
        loaded_image
            .segments
            .push(dataset::SegmentEntry::new(next_id, default_class));
        loaded_image.mark_dirty();
        next_id
    }

    /// Same as the New Segment control: requires a loaded image and an open dataset.
    fn create_new_segment_for_current_image(&mut self) {
        let (split, relative_path) = {
            let Some(active) = self.current_image.as_ref() else {
                self.status_message = Some("Load an image before creating a segment.".to_owned());
                return;
            };
            (
                active.reference.split,
                active.reference.relative_path.clone(),
            )
        };
        let next_id = {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            let Some(dataset) = guard.as_mut() else {
                self.status_message = Some("Open a dataset before creating a segment.".to_owned());
                return;
            };
            let classes_snapshot = dataset.classes.clone();
            let default_class = classes_snapshot.first().map_or(0, |c| c.id);
            let Ok(loaded_image) = dataset.ensure_image_loaded(split, &relative_path) else {
                self.status_message =
                    Some("Unable to load current image segments for a new segment.".to_owned());
                return;
            };
            Self::push_new_segment_core(loaded_image, default_class)
        };
        self.selected_segment_id = Some(next_id);
        self.data_dirty = true;
        self.status_message = Some(format!("Created segment #{next_id}"));
    }

    fn segment_panel(&mut self, ui: &mut egui::Ui) {
        let Some(active) = self.current_image.as_ref() else {
            ui.label("Load an image to manage segments");
            return;
        };
        let mut dataset_guard = self.dataset.write().expect("dataset lock poisoned");
        let Some(dataset) = dataset_guard.as_mut() else {
            ui.label("Open a dataset to manage segments");
            return;
        };
        let classes_snapshot = dataset.classes.clone();
        let default_class = classes_snapshot.first().map_or(0, |c| c.id);
        let Ok(loaded_image) =
            dataset.ensure_image_loaded(active.reference.split, &active.reference.relative_path)
        else {
            ui.label("Unable to load current image segments");
            return;
        };

        ui.horizontal(|ui| {
            ui.heading("Segments");
            if ui.button("New Segment").clicked() {
                let next_id = Self::push_new_segment_core(loaded_image, default_class);
                self.selected_segment_id = Some(next_id);
                self.data_dirty = true;
                self.status_message = Some(format!("Created segment #{next_id}"));
            }
        });
        ui.separator();

        if loaded_image.segments.is_empty() {
            ui.label("No segments created yet");
            return;
        }

        let mut segment_to_remove = None;
        let mut dirty = false;
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("segment_grid")
                .striped(true)
                .num_columns(5)
                .show(ui, |ui| {
                    for (idx, segment) in loaded_image.segments.iter_mut().enumerate() {
                        ui.label(format!("#{:02}", segment.id));
                        let mut class_changed = false;
                        egui::ComboBox::from_id_salt(("segment_class", segment.id))
                            .selected_text(Self::class_label_from_classes(
                                segment.class_index,
                                &classes_snapshot,
                            ))
                            .show_ui(ui, |ui| {
                                for class in &classes_snapshot {
                                    if ui
                                        .selectable_value(
                                            &mut segment.class_index,
                                            class.id,
                                            format!("{} - {}", class.id, class.name),
                                        )
                                        .clicked()
                                    {
                                        class_changed = true;
                                    }
                                }
                            });
                        if class_changed {
                            dirty = true;
                            self.data_dirty = true;
                        }
                        ui.label(format!("Pts: {}", segment.polygon.len()));
                        if ui.small_button("Edit").clicked() {
                            self.selected_segment_id = Some(segment.id);
                        }
                        if ui.small_button("Delete").clicked() {
                            segment_to_remove = Some(idx);
                        }
                        ui.end_row();
                    }
                });
        });

        if let Some(idx) = segment_to_remove.filter(|&i| i < loaded_image.segments.len()) {
            let removed = loaded_image.segments.remove(idx);
            if self.selected_segment_id == Some(removed.id) {
                self.selected_segment_id = None;
            }
            dirty = true;
            self.data_dirty = true;
        }
        if dirty {
            loaded_image.mark_dirty();
        }
    }

    fn class_label_from_classes(class_id: usize, classes: &[dataset::ClassEntry]) -> String {
        classes.iter().find(|c| c.id == class_id).map_or_else(
            || format!("{class_id} - Unknown"),
            |c| format!("{} - {}", c.id, c.name),
        )
    }

    fn current_segments_snapshot(&self) -> Vec<dataset::SegmentEntry> {
        let Some(active) = &self.current_image else {
            return Vec::new();
        };
        self.dataset
            .read()
            .ok()
            .and_then(|guard| {
                guard.as_ref().and_then(|dataset| {
                    dataset
                        .segments_snapshot(active.reference.split, &active.reference.relative_path)
                })
            })
            .unwrap_or_default()
    }

    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    fn draw_zoomable_image_in_viewport(
        &mut self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        viewport_clip: egui::Rect,
        texture_id: egui::TextureId,
        texture_size: egui::Vec2,
        segments_snapshot: &[dataset::SegmentEntry],
    ) {
        let zoom_factor = self.image_view_zoom;
        let stroke_width = 2.0 / zoom_factor;
        let handle_half = 3.0 / zoom_factor;
        let font_size = 14.0 / zoom_factor;

        let _ = ui.scope_builder(egui::UiBuilder::new().max_rect(viewport), |ui| {
            ui.set_clip_rect(viewport_clip);

            let fit_scale = (viewport.width() / texture_size.x)
                .min(viewport.height() / texture_size.y)
                .clamp(0.01, 1.0);

            let display_size_before_scroll = texture_size * (fit_scale * self.image_view_zoom);
            let image_center_before = viewport.center() + self.image_view_pan;
            let image_rect_before =
                egui::Rect::from_center_size(image_center_before, display_size_before_scroll);

            if let Some(pointer_pos) = ui.ctx().pointer_latest_pos()
                && image_rect_before.contains(pointer_pos)
            {
                let scroll_y = ui.ctx().input(|i| i.smooth_scroll_delta.y);
                if scroll_y.abs() > f32::EPSILON
                    && image_rect_before.width() > f32::EPSILON
                    && image_rect_before.height() > f32::EPSILON
                {
                    let rel = ((pointer_pos - image_rect_before.min) / image_rect_before.size())
                        .clamp(egui::vec2(0.0, 0.0), egui::vec2(1.0, 1.0));
                    let new_zoom = (self.image_view_zoom
                        * (-scroll_y * IMAGE_VIEW_ZOOM_SCROLL_EXP_FACTOR).exp())
                    .clamp(IMAGE_VIEW_ZOOM_MIN, IMAGE_VIEW_ZOOM_MAX);
                    if (new_zoom - self.image_view_zoom).abs() > f32::EPSILON {
                        self.image_view_zoom = new_zoom;
                        let new_display_size = texture_size * (fit_scale * self.image_view_zoom);
                        self.image_view_pan = pointer_pos.to_vec2() - rel * new_display_size
                            + new_display_size * 0.5
                            - viewport.center().to_vec2();
                    }
                }
            }

            let display_size = texture_size * (fit_scale * self.image_view_zoom);
            let image_rect =
                egui::Rect::from_center_size(viewport.center() + self.image_view_pan, display_size);

            let image_widget = egui::widgets::Image::new((texture_id, texture_size))
                .fit_to_exact_size(display_size)
                .sense(egui::Sense::click());
            let response = ui.put(image_rect, image_widget);
            let rect = response.rect;
            let painter = ui.painter_at(rect);
            for segment in segments_snapshot {
                let points: Vec<egui::Pos2> = segment
                    .polygon
                    .iter()
                    .map(|pt| {
                        egui::Pos2::new(
                            rect.left() + pt.x * rect.width(),
                            rect.top() + pt.y * rect.height(),
                        )
                    })
                    .collect();
                let is_selected = Some(segment.id) == self.selected_segment_id;
                let (fill, stroke_color) = Self::segment_colors(segment.id, is_selected);
                match points.len() {
                    0 | 1 => {}
                    2 => {
                        painter.add(egui::epaint::Shape::line_segment(
                            [points[0], points[1]],
                            egui::Stroke::new(stroke_width, stroke_color),
                        ));
                        let midpoint = egui::pos2(
                            (points[0].x + points[1].x) * 0.5,
                            (points[0].y + points[1].y) * 0.5,
                        );
                        painter.text(
                            midpoint,
                            egui::Align2::CENTER_CENTER,
                            format!("#{}", segment.id),
                            egui::FontId::proportional(font_size),
                            stroke_color,
                        );
                    }
                    _ => {
                        painter.add(egui::epaint::PathShape::convex_polygon(
                            points.clone(),
                            fill,
                            egui::epaint::Stroke::new(stroke_width, stroke_color),
                        ));
                        let (sum_x, sum_y) = points
                            .iter()
                            .fold((0.0, 0.0), |acc, p| (acc.0 + p.x, acc.1 + p.y));
                        let len = points.len() as f32;
                        let centroid = egui::pos2(sum_x / len, sum_y / len);
                        painter.text(
                            centroid,
                            egui::Align2::CENTER_CENTER,
                            format!("#{}", segment.id),
                            egui::FontId::proportional(font_size),
                            stroke_color,
                        );
                    }
                }

                for point in &points {
                    let handle_rect = egui::Rect::from_center_size(
                        *point,
                        egui::vec2(handle_half * 2.0, handle_half * 2.0),
                    );
                    painter.add(egui::epaint::Shape::rect_filled(
                        handle_rect,
                        1.0,
                        stroke_color,
                    ));
                }
            }

            if response.clicked()
                && let Some(pos) = response.interact_pointer_pos()
                && rect.width() > 0.0
                && rect.height() > 0.0
            {
                let rel_x = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                let rel_y = ((pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
                self.add_point_to_selected_segment(rel_x, rel_y);
            }
        });
    }

    fn center_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Image");
        ui.separator();
        if let Some(active) = &self.current_image {
            let segments_snapshot = self.current_segments_snapshot();
            if let Some(selected) = self.selected_segment_id {
                ui.label(format!("Editing segment #{selected}"));
            } else {
                ui.label("Select a segment to edit");
            }
            ui.label(active.reference.full_path.display().to_string());

            let texture_id = active.texture.id();
            let texture_size = active.texture.size_vec2();
            let viewport = ui.available_rect_before_wrap();
            let viewport_clip = viewport.intersect(ui.clip_rect());

            self.draw_zoomable_image_in_viewport(
                ui,
                viewport,
                viewport_clip,
                texture_id,
                texture_size,
                &segments_snapshot,
            );
        } else {
            ui.label("Select an image from the left pane to begin annotating.");
        }
    }

    fn has_dataset(&self) -> bool {
        self.dataset
            .read()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    fn clear_polygon_edit_history(&mut self) {
        self.polygon_undo_stack.clear();
        self.polygon_redo_stack.clear();
        self.polygon_history_segment_id = None;
    }

    fn sync_polygon_edit_history_segment(&mut self) {
        if self.polygon_history_segment_id != self.selected_segment_id {
            self.polygon_undo_stack.clear();
            self.polygon_redo_stack.clear();
            self.polygon_history_segment_id = self.selected_segment_id;
        }
    }

    fn undo_selected_segment_polygon(&mut self) {
        let Some(segment_id) = self.selected_segment_id else {
            self.status_message = Some("Select a segment before undo (Z)".to_owned());
            return;
        };
        let (split, relative_path) = {
            let Some(active) = self.current_image.as_ref() else {
                self.status_message = Some("Load an image before undo (Z)".to_owned());
                return;
            };
            (
                active.reference.split,
                active.reference.relative_path.clone(),
            )
        };
        self.sync_polygon_edit_history_segment();
        let Some(previous) = self.polygon_undo_stack.pop() else {
            self.status_message = Some("Nothing to undo".to_owned());
            return;
        };

        let mut segment_missing = false;
        {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            if let Some(dataset) = guard.as_mut() {
                let Ok(loaded_image) = dataset.ensure_image_loaded(split, &relative_path) else {
                    self.status_message = Some("Unable to load current image for undo".to_owned());
                    self.polygon_undo_stack.push(previous);
                    return;
                };
                if let Some(idx) = loaded_image
                    .segments
                    .iter()
                    .position(|seg| seg.id == segment_id)
                {
                    let count = {
                        let seg = &mut loaded_image.segments[idx];
                        self.polygon_redo_stack.push(seg.polygon.clone());
                        seg.polygon = previous;
                        seg.polygon.len()
                    };
                    loaded_image.mark_dirty();
                    self.data_dirty = true;
                    self.status_message = Some(format!(
                        "Undo (Z): segment #{segment_id} now has {count} point(s)"
                    ));
                } else {
                    self.status_message = Some("Selected segment no longer exists".to_owned());
                    self.selected_segment_id = None;
                    segment_missing = true;
                }
            }
        }
        if segment_missing {
            self.clear_polygon_edit_history();
        }
    }

    fn redo_selected_segment_polygon(&mut self) {
        let Some(segment_id) = self.selected_segment_id else {
            self.status_message = Some("Select a segment before redo (Y)".to_owned());
            return;
        };
        let (split, relative_path) = {
            let Some(active) = self.current_image.as_ref() else {
                self.status_message = Some("Load an image before redo (Y)".to_owned());
                return;
            };
            (
                active.reference.split,
                active.reference.relative_path.clone(),
            )
        };
        self.sync_polygon_edit_history_segment();
        let Some(next) = self.polygon_redo_stack.pop() else {
            self.status_message = Some("Nothing to redo".to_owned());
            return;
        };

        let mut segment_missing = false;
        {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            if let Some(dataset) = guard.as_mut() {
                let Ok(loaded_image) = dataset.ensure_image_loaded(split, &relative_path) else {
                    self.status_message = Some("Unable to load current image for redo".to_owned());
                    self.polygon_redo_stack.push(next);
                    return;
                };
                if let Some(idx) = loaded_image
                    .segments
                    .iter()
                    .position(|seg| seg.id == segment_id)
                {
                    let count = {
                        let seg = &mut loaded_image.segments[idx];
                        self.polygon_undo_stack.push(seg.polygon.clone());
                        seg.polygon = next;
                        seg.polygon.len()
                    };
                    loaded_image.mark_dirty();
                    self.data_dirty = true;
                    self.status_message = Some(format!(
                        "Redo (Y): segment #{segment_id} now has {count} point(s)"
                    ));
                } else {
                    self.status_message = Some("Selected segment no longer exists".to_owned());
                    self.selected_segment_id = None;
                    segment_missing = true;
                }
            }
        }
        if segment_missing {
            self.clear_polygon_edit_history();
        }
    }

    /// Reset zoom/pan to match a freshly loaded image (does not reload texture or clear selection).
    fn recenter_image_view(&mut self) {
        if self.current_image.is_none() {
            return;
        }
        self.image_view_zoom = 1.0;
        self.image_view_pan = egui::Vec2::ZERO;
        self.status_message = Some("Recentered image view (R)".to_owned());
    }

    fn load_image(&mut self, ctx: &egui::Context, split: ImagePurpose, relative_path: &Path) {
        let (color_image, reference, segment_count, dirty) = {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            let Some(dataset) = guard.as_mut() else {
                self.status_message = Some("Open a dataset before loading images".to_owned());
                return;
            };
            let (color_image, segment_count, dirty) = {
                let loaded = match dataset.ensure_image_loaded(split, relative_path) {
                    Ok(img) => img,
                    Err(err) => {
                        self.status_message = Some(err);
                        return;
                    }
                };
                (loaded.as_color_image(), loaded.segments.len(), loaded.dirty)
            };
            let reference = dataset.image_reference(split, relative_path);
            (color_image, reference, segment_count, dirty)
        };

        let texture = ctx.load_texture(
            format!("loaded_image_{}", reference.full_path.display()),
            color_image,
            egui::TextureOptions::LINEAR,
        );
        self.current_image = Some(ActiveImage { reference, texture });
        self.image_view_zoom = 1.0;
        self.image_view_pan = egui::Vec2::ZERO;
        self.selected_segment_id = None;
        self.clear_polygon_edit_history();
        self.data_dirty = dirty;
        if let Some(active) = &self.current_image {
            self.status_message = Some(format!(
                "Loaded image {} with {} segment(s)",
                active.reference.full_path.display(),
                segment_count
            ));
        }
    }

    fn navigate_image(&mut self, direction: NavDirection, ctx: &egui::Context) {
        let Some(target) = self.next_image_path(direction) else {
            self.status_message = Some("No images available".to_owned());
            return;
        };
        if self.data_dirty {
            self.pending_action = Some(PendingAction::Navigate(target));
            self.show_save_prompt = true;
        } else {
            self.load_image(ctx, target.split, &target.relative_path);
        }
    }

    fn next_image_path(&self, direction: NavDirection) -> Option<dataset::ImageReference> {
        let guard = self.dataset.read().ok()?;
        let dataset = guard.as_ref()?;
        let images = dataset.flattened_image_refs();
        if images.is_empty() {
            return None;
        }
        let current_idx = self.current_image.as_ref().and_then(|active| {
            images.iter().position(|entry| {
                entry.matches(active.reference.split, &active.reference.relative_path)
            })
        });
        let target_idx = match current_idx {
            Some(idx) => match direction {
                NavDirection::Previous => (idx + images.len() - 1) % images.len(),
                NavDirection::Next => (idx + 1) % images.len(),
            },
            None => match direction {
                NavDirection::Previous => images.len() - 1,
                NavDirection::Next => 0,
            },
        };
        images.get(target_idx).cloned()
    }

    fn execute_pending_action(&mut self, ctx: &egui::Context) {
        if let Some(action) = self.pending_action.take() {
            match action {
                PendingAction::SaveOnly => {}
                PendingAction::Navigate(reference) => {
                    self.load_image(ctx, reference.split, &reference.relative_path);
                }
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn segment_colors(id: usize, selected: bool) -> (egui::Color32, egui::Color32) {
        let hue = (((id as f32) * 137.5) % 360.0) / 360.0;
        let fill = hsva_to_color32(egui::epaint::Hsva::new(
            hue,
            0.7,
            0.9,
            if selected { 0.45 } else { 0.2 },
        ));
        let stroke = hsva_to_color32(egui::epaint::Hsva::new(hue, 0.8, 0.7, 1.0));
        (fill, stroke)
    }

    fn add_point_to_selected_segment(&mut self, x: f32, y: f32) {
        let Some(segment_id) = self.selected_segment_id else {
            self.status_message = Some("Select a segment before adding points".to_owned());
            return;
        };
        let (split, relative_path) = {
            let Some(active) = self.current_image.as_ref() else {
                self.status_message = Some("Load an image before editing segments".to_owned());
                return;
            };
            (
                active.reference.split,
                active.reference.relative_path.clone(),
            )
        };
        self.sync_polygon_edit_history_segment();

        let mut guard = self.dataset.write().expect("dataset lock poisoned");
        if let Some(dataset) = guard.as_mut() {
            let Ok(loaded_image) = dataset.ensure_image_loaded(split, &relative_path) else {
                self.status_message = Some("Unable to load current image for editing".to_owned());
                return;
            };
            if let Some(idx) = loaded_image
                .segments
                .iter()
                .position(|seg| seg.id == segment_id)
            {
                {
                    let seg = &mut loaded_image.segments[idx];
                    self.polygon_undo_stack.push(seg.polygon.clone());
                    seg.polygon.push(dataset::SegmentPoint { x, y });
                }
                self.polygon_redo_stack.clear();
                loaded_image.mark_dirty();
                self.data_dirty = true;
                self.status_message = Some(format!(
                    "Added point ({x:.3}, {y:.3}) to segment #{segment_id}"
                ));
            } else {
                self.status_message = Some("Selected segment no longer exists".to_owned());
                self.selected_segment_id = None;
            }
        }
    }

    fn save_current_image_segments(&mut self) -> Result<(), String> {
        let reference = self
            .current_image
            .as_ref()
            .map(|img| img.reference.clone())
            .ok_or_else(|| "Load an image before saving segments".to_owned())?;

        let count = {
            let mut guard = self.dataset.write().expect("dataset lock poisoned");
            let dataset = guard
                .as_mut()
                .ok_or_else(|| "No dataset loaded".to_owned())?;
            dataset
                .save_metadata()
                .map_err(|err| format!("Failed to save dataset metadata: {err}"))?;
            dataset.save_loaded_image(reference.split, &reference.relative_path)?
        };

        self.data_dirty = false;
        let txt_path = reference.full_path.with_extension("txt");
        self.status_message = Some(format!(
            "Saved {count} segment(s) to {}",
            txt_path.display()
        ));
        Ok(())
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn hsva_to_color32(hsva: egui::epaint::Hsva) -> egui::Color32 {
    let [r, g, b, a] = hsva.to_rgba_unmultiplied();
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgba_unmultiplied(to_u8(r), to_u8(g), to_u8(b), to_u8(a))
}
