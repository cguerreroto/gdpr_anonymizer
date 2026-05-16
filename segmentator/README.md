# YOLO Segmentator

Rust/egui desktop application for curating YOLO segmentation datasets. It helps create datasets, manage classes, inspect image splits, and draw polygon annotations that serialize to YOLO-compatible `.txt` files.

## Features
- Toolbar workflow: create/open/close datasets, save annotations, navigate images via buttons or arrow keys.
- Dataset panel shows metadata, editable class grid, and three image grids (train/val/test) with `(new)` label indicators and load buttons.
- Center view scales the loaded image to the panel, draws all segment polygons or line segments (with per-point squares), and lets users add normalized points by clicking.
- Segment panel lists segments in a compact grid with per-row controls for class selection, edit selection, and delete actions; new segments start empty.
- Background worker thread watches dataset image folders (using `notify`) and refreshes image lists plus triggers egui repaints.

## Usage
1. `cargo run --release` (requires a desktop environment).
2. Use **New Dataset** to create folder structure plus default YAML, or **Open Directory** to load an existing dataset.
3. Define classes, load images, draw/edit polygons, and press **Save** to persist YOLO annotations (prompts if unsaved changes exist).
4. Navigate through images via Previous/Next buttons or arrow keys.


## ml_pipeline (Python)

The `ml_pipeline` package provides `gdpr-yolo-normalize-labels` (label layout audit), `gdpr-yolo-fix-dataset-yaml` (`nc`, `names`, split paths, validation carving), `gdpr-yolo-train` (YOLO26 segmentation training), and `gdpr-yolo-validate` (mask + box mAP on a trained checkpoint). From `segmentator/ml_pipeline`, run `uv sync` for layout and yaml tools. Before `gdpr-yolo-train` or `gdpr-yolo-validate`, run `uv sync --extra train` so Ultralytics stays installed (a plain `uv sync` drops that extra). Run `uv run --group dev pytest` from the same directory to execute the package unit tests. See the repository root `README.md` for the full command line reference.

## Development
- Rust edition 2021; dependencies include `eframe`, `egui`, `serde_yaml_ng`, `futures`, and `notify`.
- Always run `cargo fmt` after making code changes and `cargo clippy -D warnings` to ensure lints stay clean.
- Files of interest: `src/main.rs` (UI + event handling) and `src/dataset.rs` (dataset model, IO, watcher).
