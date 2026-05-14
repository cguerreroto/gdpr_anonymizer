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
