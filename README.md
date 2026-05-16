# gdpr_anonymizer

The repository provides tooling for a video-oriented privacy workflow: frame sampling, instance segmentation labels (for example person or vehicle regions, depending on annotation scope), and deterministic dataset augmentation. The output is a YOLO-style layout suitable for external training or evaluation.

The following sections summarize the components and a conventional processing order.

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

The segmentator defaults to class index 0 = Person and 1 = Car. The `names` field in `dataset.yaml` must match the class indices present in the label files when the dataset is consumed by an external trainer.

## Where data and weights live

Large or sensitive assets are not tracked in Git. Local directory layout is left to the deployment environment. Typical placeholders:

### Datasets

`<path-to-data-root>/...` for raw frame dumps, hand-labeled trees, and augmented exports. The repository `.gitignore` excludes `data/`.

### Weights and training runs

`<path-to-runs-root>/...` for training outputs and downloaded `*.pt` weights. The repository `.gitignore` excludes `*.pt`.

Dataset roots and run directories are supplied as explicit paths on the command line when invoking the Python CLIs or when opening a dataset in the segmentator.

## Extractor and augmentor (Python)

Installation, prerequisites, and full command examples are documented in [`extractor/yolo_raw_extractor/README.md`](extractor/yolo_raw_extractor/README.md).

Abbreviated sequence:

1. Install dependencies (from `extractor/yolo_raw_extractor`): `uv sync`
2. Sample frames: `uv run yolo-raw-extractor <video.mp4> <dataset-dir>`
3. After labeling: `uv run yolo-segment-extractor <labeled-dataset>`
4. Augment: `uv run yolo-augmentor <labeled-dataset> <output-dataset> --seed <n>`

The augmentor writes images under `images/<split>/` and co-located label `.txt` files with matching stems.

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

## Dataset YAML (`ml_pipeline`)

The `gdpr-yolo-fix-dataset-yaml` command scans segmentation label files under the dataset root, compares class indices to `dataset.yaml`, and reports missing `nc` or `names` entries. With `--apply`, it writes an updated `dataset.yaml` (default class names follow the segmentator: 0 = Person, 1 = Car). Optional `--carve-val-fraction` copies or moves a deterministic share of labeled train items into `images/val` and `labels/val` for a held-out validation split.

From `segmentator/ml_pipeline` (after `uv sync`):

```bash
uv run gdpr-yolo-fix-dataset-yaml <path-to-dataset-root> --dry-run
uv run gdpr-yolo-fix-dataset-yaml <path-to-dataset-root> --apply
```

Use `--strict` to exit with a non-zero status when issues remain. Use `--carve-val-fraction` with `--carve-seed` and optionally `--carve-move` when building a validation split from train.

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

## Segmentator (Rust)

The segmentator is a desktop application for dataset creation, class management, and polygon editing. Annotations are serialized as YOLO segmentation `.txt` files beside the corresponding images.

Further detail appears in [`segmentator/README.md`](segmentator/README.md).

From the `segmentator/` crate root:

```bash
cargo run --release
```

## Reference docs in the repo

Supplementary notes, including YOLO segmentation label syntax, are under [`extractor/yolo-docs`](extractor/yolo-docs); see for instance [`extractor/yolo-docs/dataset.md`](extractor/yolo-docs/dataset.md).

## Links

- [Ultralytics train mode](https://docs.ultralytics.com/modes/train)
- [Ultralytics predict mode](https://docs.ultralytics.com/modes/predict)
