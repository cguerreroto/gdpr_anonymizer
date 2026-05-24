# gdpr_anonymizer

Tools for a video privacy workflow: sample frames, annotate instance segmentation masks (for example people or vehicles), train a YOLO26 segmentation model, and blur sensitive regions before sharing footage. Outputs use a YOLO-style dataset layout for custom training and evaluation.

The repository targets researchers and developers who need local, repeatable control over labeling, training, and anonymization. Datasets and model weights stay on your machine; you pass explicit paths to each command.

## Repository layout

| Path | Role |
| --- | --- |
| [`extractor/yolo_raw_extractor`](extractor/yolo_raw_extractor) | Frame extraction, segment crops, dataset augmentation |
| [`segmentator`](segmentator) | Desktop polygon editor (Rust / egui) |
| [`segmentator/ml_pipeline`](segmentator/ml_pipeline) | Dataset audit, training, validation, video blur |
| [`extractor/yolo-docs`](extractor/yolo-docs) | YOLO segmentation notes and label syntax |

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) matching `rust-version` in [`segmentator/Cargo.toml`](segmentator/Cargo.toml)
- [uv](https://github.com/astral-sh/uv) and Python 3.13+ for the extractor
- Python 3.11+ for `ml_pipeline`
- Optional GPU (CUDA or Apple Silicon) for Ultralytics training and inference

## Quick start

```bash
cd extractor/yolo_raw_extractor && uv sync
uv run yolo-raw-extractor /path/to/video.mp4 /path/to/dataset
```

```bash
cd segmentator && cargo run --release
```

```bash
cd segmentator/ml_pipeline && uv sync
uv run gdpr-yolo-normalize-labels /path/to/dataset
```

Full setup, per-package tests, and contribution workflow: [CONTRIBUTING.md](CONTRIBUTING.md). Command-line detail for every stage is in the reference sections below.

## Pipeline at a glance

| Order | Stage | Tool | Location |
| ---: | --- | --- | --- |
| 1 | Sample frames from video into a YOLO layout | `yolo-raw-extractor` | [`extractor/yolo_raw_extractor`](extractor/yolo_raw_extractor) |
| 2 | Harvest polygon crops for augmentation | `yolo-segment-extractor` | [`extractor/yolo_raw_extractor`](extractor/yolo_raw_extractor) |
| 3 | Draw and edit segmentation labels (polygons → YOLO `.txt`) | YOLO Segmentator (Rust / egui) | [`segmentator`](segmentator) |
| 4 | Build an augmented copy of the labeled dataset | `yolo-augmentor` | [`extractor/yolo_raw_extractor`](extractor/yolo_raw_extractor) |
| 5 | Audit dataset and mirror labels into `labels/<split>/` | `gdpr-yolo-normalize-labels` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |
| 6 | Audit and fix `dataset.yaml` (`nc`, `names`, splits) | `gdpr-yolo-fix-dataset-yaml` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |
| 7 | Train a YOLO26 segmentation model on the prepared dataset | `gdpr-yolo-train` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |
| 8 | Validate a trained checkpoint and report mask + box mAP | `gdpr-yolo-validate` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |
| 9 | Analyze FN/FP errors on validation split to guide labeling | `gdpr-yolo-analyze-errors` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |
| 10 | Predict on still images and save overlays for human review | `gdpr-yolo-predict` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |
| 11 | Export trained checkpoint to ONNX or other portable formats | `gdpr-yolo-export` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |
| 12 | Blur or pixelate predicted masks in a video and write a new file | `gdpr-yolo-blur-video` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |
| 13 | Quarantine dataset samples or remove old Ultralytics run folders | `gdpr-yolo-clean` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |
| 14 | Warm-start a new training run from a previous `best.pt` and compare metrics | `gdpr-yolo-iterate` | [`segmentator/ml_pipeline`](segmentator/ml_pipeline) |

The segmentator defaults to class index 0 = Person and 1 = Car. The `names` field in `dataset.yaml` must match the class indices present in the label files when the dataset is consumed by an external trainer.

## Where data and weights live

Large or sensitive assets are not tracked in Git. Local directory layout is left to the deployment environment. Typical placeholders:

### Datasets

`<path-to-data-root>/...` for raw frame dumps, hand-labeled trees, and augmented exports. The repository `.gitignore` excludes `data/`.

### Weights and training runs

`<path-to-runs-root>/...` for training outputs and downloaded `*.pt` weights. The repository `.gitignore` excludes `*.pt`.

Dataset roots and run directories are supplied as explicit paths on the command line when invoking the Python CLIs or when opening a dataset in the segmentator.

## Development

From the repository root:

```bash
make test
```

Per-package test commands: [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

- This tooling does not by itself guarantee legal GDPR compliance. You remain responsible for lawful processing and consent in your jurisdiction.
- Model quality depends on your labels, class definitions, and training data; poor annotations produce unreliable masks for blur.
- Training and video blur require optional Ultralytics dependencies and suitable hardware; CPU-only runs may be slow on long videos.
- The segmentator defaults to class index `0` = Person and `1` = Car; other use cases require consistent `dataset.yaml` and label indices.

## Command reference

Expand a section for install notes, examples, and flags.

<details>
<summary>Extractor and augmentor (Python)</summary>

## Extractor and augmentor (Python)

Installation, prerequisites, and full command examples are documented in [`extractor/yolo_raw_extractor/README.md`](extractor/yolo_raw_extractor/README.md).

Abbreviated sequence:

1. Install dependencies (from `extractor/yolo_raw_extractor`): `uv sync`
2. Sample frames: `uv run yolo-raw-extractor <video.mp4> <dataset-dir>`
3. After labeling: `uv run yolo-segment-extractor <labeled-dataset>`
4. Augment: `uv run yolo-augmentor <labeled-dataset> <output-dataset> --seed <n>`

The augmentor writes images under `images/<split>/` and co-located label `.txt` files with matching stems.

</details>

<details>
<summary>Dataset label layout (gdpr-yolo-normalize-labels)</summary>

## Dataset label layout (`ml_pipeline`)

The `gdpr-yolo-normalize-labels` command reads `dataset.yaml`, resolves each declared split directory under `images/<split>/`, and compares image stems to segmentation `.txt` stems. It prints a JSON report (and optional `--report-json` file) listing images without a matching label and labels without a matching image. It then ensures a parallel directory `labels/<split>/` exists for each split. By default each entry there is a symbolic link to the co-located `.txt` beside the image, so label content stays in one place on disk while `labels/<split>/` paths are also available.

In general, Ultralytics documentation recommends keeping `images/<split>/` and `labels/<split>/` as sibling trees with identical stem names so training does not miss label files. That convention is not implemented or validated inside this command: there is no Ultralytics dependency here. The tool only aligns the tree with that common layout and performs the audit so downstream training (Ultralytics or otherwise) receives consistent paths.

Install (from `segmentator/ml_pipeline`):

```bash
uv sync
```

Run:

```bash
uv run gdpr-yolo-normalize-labels <path-to-dataset-root>
```

### Ultralytics for train and validate (`ml_pipeline`)

Commands `gdpr-yolo-train` and `gdpr-yolo-validate` need the optional `[train]` dependency group (Ultralytics and PyTorch). Layout and `dataset.yaml` tools work after `uv sync` alone. Before every training or validation session, from `segmentator/ml_pipeline`:

```bash
uv sync --extra train
```

A plain `uv sync` without `--extra train` removes Ultralytics from the local environment; `gdpr-yolo-train` and `gdpr-yolo-validate` then fail with an error that repeats the command above.

Options include `--dry-run`, `--report-json <path>`, `--mode copy|move|symlink`, `--force`, and `--strict` (non-zero exit if any split has missing image–label pairs).

The `move` mode deletes co-located `.txt` files from `images/<split>/`, which breaks workflows that only resolve labels next to image files (including the segmentator). The default `symlink` mode keeps a single on-disk file while exposing `labels/<split>/` paths.

</details>

<details>
<summary>Dataset YAML (gdpr-yolo-fix-dataset-yaml)</summary>

## Dataset YAML (`ml_pipeline`)

The `gdpr-yolo-fix-dataset-yaml` command scans segmentation label files under the dataset root, compares class indices to `dataset.yaml`, and reports missing `nc` or `names` entries. With `--apply`, it writes an updated `dataset.yaml` (default class names follow the segmentator: 0 = Person, 1 = Car). Optional `--carve-val-fraction` copies or moves a deterministic share of labeled train items into `images/val` and `labels/val` for a held-out validation split.

From `segmentator/ml_pipeline` (after `uv sync`):

```bash
uv run gdpr-yolo-fix-dataset-yaml <path-to-dataset-root> --dry-run
uv run gdpr-yolo-fix-dataset-yaml <path-to-dataset-root> --apply
```

Use `--strict` to exit with a non-zero status when issues remain. Use `--carve-val-fraction` with `--carve-seed` and optionally `--carve-move` when building a validation split from train.

</details>

<details>
<summary>Training (gdpr-yolo-train)</summary>

## Training (`ml_pipeline`)

The `gdpr-yolo-train` command launches a YOLO26 segmentation training run on the dataset described by `dataset.yaml`. The training task is fixed to `segment`, and the default base model is `yolo26n-seg.pt`. The run writes checkpoints and logs under `<project>/<name>/`, where both values default to `runs/yolo26n_seg`. The CLI prints a JSON report describing the resolved arguments (and the save directory once training completes) and accepts `--report-json` to persist that report next to the run.

Before invoking Ultralytics, the CLI writes a copy of `dataset.yaml` named `dataset.resolved.yaml` inside the run directory, with `path` and any `train`, `val`, `test` entries rewritten to absolute paths anchored at the dataset root. Ultralytics resolves a relative `path` against its own dataset directory or the current working directory, so a yaml with `path: .` would otherwise fail when training is launched from outside the dataset folder. The original `dataset.yaml` is left untouched so it stays portable across machines.

Run a training pass (from `segmentator/ml_pipeline`, after `uv sync --extra train`):

```bash
uv run gdpr-yolo-train <path-to-dataset-root> \
    --epochs 100 --imgsz 640 --batch 16 \
    --project <path-to-runs-root> --name yolo26n_seg_v1
```

Useful options:

- `--model <name-or-path>`: select another YOLO26 segmentation weight (default `yolo26n-seg.pt`).
- `--weights <path>`: warm-start from a previous `best.pt` instead of the base model.
- `--device <id|cpu|mps>`: override the device autoselect.
- `--patience <n>` and `--save-period <n>`: early stopping and intermediate checkpoint cadence.
- `--resume`: continue an interrupted run with the same `--project` and `--name`.
- `--exist-ok`: keep outputs under `<project>/<name>` when that folder already exists. Without this flag, Ultralytics creates suffixed run folders (`name-2`, `name-3`, ...).
- `--dry-run`: print the resolved Ultralytics arguments without invoking training; useful before long jobs.

After a successful run, the JSON report includes `save_dir` and `validate_weights` (path to `weights/best.pt` under `save_dir`). Use `validate_weights` for `gdpr-yolo-validate --weights`.

Run outputs and downloaded weights are git-ignored (`runs/`, `*.pt`, `*.onnx`).

</details>

<details>
<summary>Validation (gdpr-yolo-validate)</summary>

## Validation (`ml_pipeline`)

The `gdpr-yolo-validate` command runs Ultralytics `model.val(...)` on a trained checkpoint and prints a JSON summary of segmentation quality. The task is fixed to `segment`. By default the command evaluates on the `val` split; `--split test` and `--split train` are also accepted.

Reported metrics (when the installed Ultralytics version exposes them):

- Mask mAP at IoU 0.5:0.95, 0.5, and 0.75.
- Box mAP at IoU 0.5:0.95 and 0.5 (auxiliary signal).
- Per-class mAP for both mask and box, keyed by the names declared in `dataset.yaml`.

Per-class numbers are the primary signal for deciding whether to add more annotated frames for a specific class (for example license plates versus people). Mask mAP is the headline metric since the downstream pipeline blurs masks rather than bounding boxes.

Each validation report includes an `interpretation` block that applies the thresholds documented in the validation driver: mask mAP@0.5:0.95 below 0.30 suggests more labels or training; between 0.30 and 0.50 is usable with per-class review; above 0.50 is a comfortable margin. The block lists `assessment` (band, summary, `ready_for_video_blur`) and `recommendations` (for example which class to label next). Use `--no-interpret` to omit it. To interpret a saved report without re-running Ultralytics:

```bash
uv run gdpr-yolo-validate --interpret-report <path-to-metrics.json>
```

Run a validation pass (from `segmentator/ml_pipeline`, after `uv sync --extra train`):

```bash
uv run gdpr-yolo-validate <path-to-dataset-root> \
    --weights <path-to-runs-root>/yolo26n_seg_v1/weights/best.pt \
    --imgsz 640 --batch 16 \
    --project <path-to-runs-root> --name yolo26n_seg_v1_val \
    --report-json <path-to-runs-root>/yolo26n_seg_v1_val/metrics.json
```

Useful options:

- `--split val|test|train`: choose the split to evaluate (default `val`).
- `--conf <float>` and `--iou <float>`: override Ultralytics defaults for confidence and NMS IoU thresholds.
- `--device <id|cpu|mps>`: override the device autoselect.
- `--save-json`: forward `save_json=True` to Ultralytics so it emits COCO-format predictions next to the val output.
- `--dry-run`: print the resolved Ultralytics arguments without invoking validation.
- `--no-interpret`: omit the `interpretation` block from the JSON report.
- `--interpret-report <path>`: print interpretation for an existing validation JSON file.

The same `dataset.resolved.yaml` rewrite used at training time is performed before validation, so the command also works when launched from outside the dataset directory.

</details>

<details>
<summary>Error analysis (gdpr-yolo-analyze-errors)</summary>

## Error analysis (`ml_pipeline`)

After validation, `gdpr-yolo-analyze-errors` identifies images with the most false negatives (missed detections) and false positives (incorrect detections), helping you decide which images need more labels.

The command runs predictions on the validation split with a configurable confidence threshold (default 0.25), matches predictions to ground truth using IoU overlap (default threshold 0.5), and reports errors aggregated by class and sorted by total error count per image.

Run error analysis (from `segmentator/ml_pipeline`, after `uv sync --extra train`):

```bash
uv run gdpr-yolo-analyze-errors <path-to-dataset-root> \
    --weights <path-to-runs-root>/yolo26n_seg_v1/weights/best.pt \
    --imgsz 640 --conf 0.25 --iou-threshold 0.5 \
    --project <path-to-runs-root> --name error_analysis \
    --report-json <path-to-runs-root>/error_analysis/errors.json
```

Useful options:

- `--conf <float>`: confidence threshold for predictions (default 0.25). Lower values find more detections but may increase false positives.
- `--iou-threshold <float>`: minimum IoU to match a prediction to ground truth (default 0.5). Lower values are more forgiving for matches.
- `--device <id|cpu|mps>`: override the device autoselect.
- `--dry-run`: print the resolved configuration without running analysis.

The JSON report includes:

- `summary.total_false_negatives_by_class`: count of GT objects missed, per class.
- `summary.total_false_positives_by_class`: count of spurious predictions, per class.
- `summary.worst_images`: list of up to 20 images with the most errors, sorted by total error count.

Use FN counts to prioritize which class needs more labeled examples. Use the worst images list to identify problematic frames for manual review or re-annotation.

</details>

<details>
<summary>Predict on stills (gdpr-yolo-predict)</summary>

## Predict on stills (`ml_pipeline`)

`gdpr-yolo-predict` runs a trained checkpoint on a folder of still images (or a single image) and writes Ultralytics overlay images for human review. It is a sanity check before moving to the video pipeline.

The command accepts an image file or a directory and recurses into subdirectories. Overlays go under `<project>/<name>/`. Optional `--save-polygons` writes predicted masks as YOLO polygon `.txt` files under `<project>/<name>/polygons/<stem>.txt` for downstream non-Python tooling.

Run a predict pass (from `segmentator/ml_pipeline`, after `uv sync --extra train`):

```bash
uv run gdpr-yolo-predict <path-to-runs-root>/yolo26n_seg_v1/weights/best.pt \
    --source <path-to-frames-folder> \
    --imgsz 640 --conf 0.25 --iou 0.7 \
    --project <path-to-runs-root> --name yolo26n_seg_predict \
    --report-json <path-to-runs-root>/yolo26n_seg_predict/predictions.json
```

Useful options:

- `--conf <float>`: detection confidence threshold (default 0.25).
- `--iou <float>`: NMS IoU threshold (default 0.7).
- `--classes 0,1`: keep only the listed class ids.
- `--device <id|cpu|mps>`: override device autoselect.
- `--no-overlays`: skip overlay images (set `save=False` on Ultralytics).
- `--save-polygons`: also export YOLO polygon `.txt` files for predicted masks.
- `--exist-ok`: reuse `<project>/<name>` instead of creating a suffixed copy.
- `--dry-run`: print resolved kwargs without invoking Ultralytics.

The JSON report includes a `summary` block with `image_count`, `images_with_detections`, `total_detections`, and `detections_by_class`. Use this to spot frames where the model misses the target classes before video processing.

</details>

<details>
<summary>Export (gdpr-yolo-export)</summary>

## Export (`ml_pipeline`)

`gdpr-yolo-export` converts a trained `.pt` checkpoint to a portable runtime format. ONNX is the default because it runs in many runtimes without a CUDA toolchain. Other formats supported by Ultralytics (`torchscript`, `engine`, `coreml`, `openvino`, `tflite`, ...) are accepted as opt-in values.

Ultralytics writes the exported artifact next to the source weights (for example `weights/best.onnx` next to `weights/best.pt`). The CLI prints a JSON report with the resolved kwargs, the resolved output path, and whether the file was created.

Run an export (from `segmentator/ml_pipeline`, after `uv sync --extra train --extra export`):

```bash
uv run gdpr-yolo-export <path-to-runs-root>/yolo26n_seg_v1/weights/best.pt \
    --format onnx --imgsz 640 --batch 1 \
    --report-json <path-to-runs-root>/yolo26n_seg_v1/weights/export.json
```

Useful options:

- `--format <name>`: target format (default `onnx`; full list shown by `--help`).
- `--imgsz <int>`: inference size baked into the export (default 640).
- `--batch <int>`: batch size baked into the export (default 1).
- `--half` or `--int8`: precision flags (mutually exclusive).
- `--dynamic`: enable dynamic input axes (ONNX, TensorRT).
- `--no-simplify`: disable ONNX graph simplification.
- `--opset <int>`: override the ONNX opset version.
- `--nms`: embed NMS in the exported model when supported by the format.
- `--device <id|cpu|mps>`: override device autoselect (TensorRT requires a CUDA device).
- `--dry-run`: print resolved kwargs without invoking Ultralytics.

ONNX export needs the optional `[export]` group (`onnx`, `onnxslim`, `onnxruntime`). Ultralytics cannot install these automatically in a uv-managed environment. Install both extras before exporting:

```bash
uv sync --extra train --extra export
```

Exported artifacts are git-ignored alongside `*.pt` and `runs/`.

</details>

<details>
<summary>Video blur (gdpr-yolo-blur-video)</summary>

## Video blur (`ml_pipeline`)

`gdpr-yolo-blur-video` runs YOLO26 segmentation on every frame of an input video and writes a copy where the union of predicted masks is blurred or pixelated. This is the privacy-oriented endpoint of the pipeline: faces (class 0) and vehicles or plates (class 1) are obscured before the video is shared.

The OpenCV-backed loop reads frames with `cv2.VideoCapture`, runs `model.predict(frame, ...)`, builds a binary union mask from `result.masks.data` for the selected classes, and writes either Gaussian-blurred or pixelated content inside the masked regions back to disk via `cv2.VideoWriter`. The output container and codec are controlled by `--fourcc` and the path extension.

Run the video blur pipeline (from `segmentator/ml_pipeline`, after `uv sync --extra train`):

```bash
uv run gdpr-yolo-blur-video <path-to-runs-root>/yolo26n_seg_v1/weights/best.pt \
    --source "<path-to-your-actual-video-file>.mp4" \
    --classes 0,1 --conf 0.25 --iou 0.7 --imgsz 640 \
    --blur-method gaussian --blur-kernel 51 \
    --report-json <path-to-runs-root>/blur/report.json
```

By default the blurred video is written next to the source as `{stem}_blurred{suffix}` (for example `clip.mp4` → `clip_blurred.mp4`). Pass `--output <directory>` to place the auto-named file elsewhere (for example `<path-to-runs-root>/blur`).

Useful options:

- `--output <dir|file>`: output directory or explicit file path (default: same folder as `--source` with `_blurred` in the filename).
- `--classes <ids>`: comma-separated class ids to blur (default: every detected class).
- `--blur-method gaussian|pixelate`: choice of obfuscation; pixelate uses `--pixelate-block`.
- `--blur-kernel <odd int>`: Gaussian blur kernel size (must be odd).
- `--blur-sigma <float>`: Gaussian blur sigma (0 lets OpenCV derive it).
- `--mask-dilate <px>`: dilate the union mask before blurring (helps when predictions are tight).
- `--conf <float>`, `--iou <float>`, `--imgsz <int>`: per-frame predict thresholds and size.
- `--device <id|cpu|mps>`: override device autoselect.
- `--fps <float>`: override the output FPS (default copies the source FPS).
- `--fourcc <tag>`: OpenCV FourCC tag for the writer (default `mp4v`).
- `--dry-run`: print the resolved kwargs without running the pipeline.
- `--no-progress`: disable the stderr progress bar and status lines during processing.

While a video is being blurred, the CLI prints status to stderr (model load, source/output paths) and updates a frame progress bar (`|====----| 120/500 (24%)`). The final JSON report still goes to stdout (or `--report-json`).

The JSON report includes a `summary` block with `total_frames`, `frames_with_detections`, `frames_with_detections_pct`, `total_detections`, `detections_by_class`, and `average_mask_area_fraction`. These KPIs sit next to the validation mAP numbers and answer how often the model intervened on real footage and how much of each frame was masked.

The output video is git-ignored alongside the rest of `runs/` and other large local artifacts. Refusing to overwrite the source is enforced.

</details>

<details>
<summary>Dataset cleaning and quarantine (gdpr-yolo-clean)</summary>

### Manual dataset cleaning (researcher workflow)

`gdpr-yolo-analyze-errors` only writes a report. It does not delete images, change splits, or remove samples from training. Treat the step as an optional cleaning pass: you use the report to decide what to fix, then re-run training and validation.

Typical sequence after error analysis:

1. Open `errors.json` together with the latest validation `metrics.json` (from `gdpr-yolo-validate`).
2. For false negatives, add or correct polygons in the segmentator for the filenames listed under `summary.worst_images`. Labels live under `<path-to-dataset-root>/labels/val/<stem>.txt` (and under `labels/train/` when the same stem is in train).
3. For false positives, remove incorrect polygons or add missing ground truth in the same `.txt` files.
4. Optionally label more source frames and run `yolo-augmentor` to grow `<path-to-dataset-root>`.
5. Re-run `gdpr-yolo-normalize-labels` and `gdpr-yolo-fix-dataset-yaml` if the dataset layout or `dataset.yaml` changed.
6. Re-train with `gdpr-yolo-train`, warm-starting from `<path-to-runs-root>/.../weights/best.pt` when continuing from a previous run.
7. Re-run `gdpr-yolo-validate` and `gdpr-yolo-analyze-errors` on the same validation split so results stay comparable.

Use mask mAP@0.5:0.95 from validation as the gate before relying on masks for blur. The validation report `interpretation` block marks `ready_for_video_blur` when overall mask mAP is at least 0.30 (usable band). Below 0.30, prioritize labeling and training over predict or video steps.

Removing files from train or val is uncommon. Reserve it for corrupt frames, duplicates, or labels you will not fix. High false-negative counts usually mean the model or labels need improvement, not that the image should be dropped from the dataset. When removal is appropriate, use `gdpr-yolo-clean` (see below) instead of deleting files by hand.

`gdpr-yolo-clean` automates quarantine and run-folder cleanup when you choose to remove samples or reset Ultralytics output directories.

Quarantine worst validation images from an error report (files move to `<path-to-dataset-root>/_quarantine/<timestamp>/` by default):

```bash
uv run gdpr-yolo-clean <path-to-dataset-root> \
    --val --from-errors <path-to-runs-root>/error_analysis/errors.json \
    --yes
```

Remove versioned train run folders and keep only the newest match (for example `yolo26n_seg_v1`, `yolo26n_seg_v1-2`, `yolo26n_seg_v1-3`):

```bash
uv run gdpr-yolo-clean --runs-dir <path-to-runs-root> \
    --run-prefix yolo26n_seg_v1 --keep-latest --yes
```

Useful options:

- `--train`, `--val` (or `--validation`), `--test`: splits to affect.
- `--stems <stem>,<stem>`: explicit image stems.
- `--from-errors <report.json>` and optional `--worst N`: stems from `summary.worst_images`.
- `--all-in-split`: every image in the selected splits (requires `--yes`).
- `--delete`: permanently delete instead of quarantine (requires `--yes`).
- `--cache`: delete `labels/**/*.cache` after dataset changes.
- `--run-names <name>,<name>`: remove explicit run subfolders under `--runs-dir`.
- `--runs-all`: remove every run subfolder except names passed to `--keep`.
- `--dry-run`: show the JSON plan without changing disk.

Always run with `--dry-run` first. Destructive actions require `--yes`.

</details>

<details>
<summary>Iteration and warm-start (gdpr-yolo-iterate)</summary>

## Iteration and warm-start (`ml_pipeline`)

`gdpr-yolo-iterate` packages the most common research loop into one command: warm-start a new YOLO26-seg training run from a previous run's `weights/best.pt`, optionally re-validate against the same split, and produce a JSON delta against a prior validation report. The command does not invent dataset paths or version names. It reads the previous run directory, derives the next run name from the existing one, and forwards through the same `gdpr-yolo-train` and `gdpr-yolo-validate` drivers.

Run name derivation:

- `<base>_vN` becomes `<base>_v<N+1>` (for example `yolo26n_seg_v1` → `yolo26n_seg_v2`).
- A trailing Ultralytics auto-suffix (`-2`, `-3`, ...) is stripped before the bump (so `yolo26n_seg_v1-3` also becomes `yolo26n_seg_v2`).
- A name without a `_vN` suffix gets `_v2` appended.

Use `--name` to override the derived name explicitly. The new run is written to `<project>/<new-name>/`; `--project` defaults to the parent of the previous run directory, so iterations land next to each other.

Run an iteration (from `segmentator/ml_pipeline`, after `uv sync --extra train`):

```bash
uv run gdpr-yolo-iterate <path-to-runs-root>/yolo26n_seg_v1 <path-to-dataset-root> \
    --epochs 100 --imgsz 640 --batch 16 \
    --compare-to <path-to-runs-root>/yolo26n_seg_v1_val/metrics.json \
    --report-json <path-to-runs-root>/yolo26n_seg_v2/iterate.json
```

Useful options:

- `--name <run-name>`: override the derived name.
- `--project <path>`: override the run parent directory (default: the previous run's parent).
- `--epochs <n>`, `--imgsz <int>`, `--batch <int>`, `--device <id|cpu|mps>`, `--patience <n>`, `--save-period <n>`, `--workers <n>`, `--exist-ok`: forwarded to `gdpr-yolo-train`.
- `--skip-validation`: train only; do not run `gdpr-yolo-validate` after training.
- `--val-name <run-name>`: validation run name (default: `<new-name>_val`).
- `--val-split val|test|train`: split to validate against (default: `val`). For comparable curves, keep the same split between iterations.
- `--compare-to <path>`: previous validation JSON. The new validation metrics are diffed against this file (overall mask + box mAP and per-class) with verdicts `improved`, `regressed`, `unchanged`, or `unknown`.
- `--dry-run`: print the resolved train (and val) kwargs without invoking Ultralytics.
- `--report-json <path>`: write the full iteration report to disk.

The JSON report includes:

- `previous_weights`, `next_run_name`, `next_project`: resolved paths and the derived name.
- `train_report`: the full `gdpr-yolo-train` JSON report (with `save_dir` and `validate_weights`).
- `validate_report`: the full `gdpr-yolo-validate` JSON report (metrics + interpretation).
- `comparison.overall`, `comparison.per_class`: deltas per metric (`previous`, `current`, `delta`, `verdict`).
- `comparison.overall_verdict`, `comparison.summary`: a single verdict for mask mAP@0.5:0.95 and a one-line text summary.

For comparable iteration curves, freeze the validation split (`val` in `dataset.yaml`) between runs and use the same `--imgsz` so val metrics are directly comparable. If you change the split or augmentation, treat the next iteration as a new baseline.

</details>

<details>
<summary>Segmentator desktop app (Rust)</summary>

## Segmentator (Rust)

The segmentator is a desktop application for dataset creation, class management, and polygon editing. Annotations are serialized as YOLO segmentation `.txt` files beside the corresponding images.

Further detail appears in [`segmentator/README.md`](segmentator/README.md).

From the `segmentator/` crate root:

```bash
cargo run --release
```

</details>

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may use, modify, and redistribute the software for noncommercial purposes when you include the license and the author notices described there and in [AUTHORS.md](AUTHORS.md). Commercial use is not permitted under this license.

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Reference docs in the repo

Supplementary notes, including YOLO segmentation label syntax, are under [`extractor/yolo-docs`](extractor/yolo-docs); see for instance [`extractor/yolo-docs/dataset.md`](extractor/yolo-docs/dataset.md).

## Links

- [Ultralytics train mode](https://docs.ultralytics.com/modes/train)
- [Ultralytics predict mode](https://docs.ultralytics.com/modes/predict)
