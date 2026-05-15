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
