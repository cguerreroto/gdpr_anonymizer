## yolo-raw-extractor

Deterministically sample raw frames from source videos and arrange them into a
YOLO-style dataset (`images/train`, `images/val`, `images/test`) ready for
labeling and later augmentation.

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

### Installation

```bash
uv sync
```

### Usage

The extractor takes two positional arguments: the source video file and the
target dataset directory that should (or will) contain the YOLO dataset.

```bash
uv run yolo-raw-extractor /path/to/video.mp4 /data/yolo-dataset
```

Every 100th frame is sampled and written to `images/train`, `images/val`, and
`images/test` following a deterministic 10:1:1 split pattern so repeated runs
with the same source always produce the same files.

### Segment harvesting

Once a dataset has been labeled, the segment extractor can cut out each labeled
positive (polygon masks) and store them as PNGs (w/ alpha) for downstream data
augmentation.

```bash
uv run yolo-segment-extractor /path/to/labeled-dataset
```

Outputs are stored under `<dataset>/segments/<split>/<class>/...png` with matching
JSON metadata (polygon coordinates relative to each PNG). Pass `--output-dir` to
override the destination if needed.

### Augmentation

With labeled data (and optional harvested segments), the augmentor can create a
deterministic derived dataset containing the original image along with:

- 4 saturation/exposure variants
- 2 occlusion variants (noise patches covering parts of positives)
- 4 insertion variants composed from harvested segment PNGs (random pose/hue)
- 1 off-white rectangle variant sized after the largest segment in the image

```bash
uv run yolo-augmentor /path/to/labeled-dataset /path/to/output-dataset --seed 42
```

The generated dataset mirrors the YOLO layout at the target path, writing each
label `.txt` beside its augmented image as required by the YOLO docs. The seed
is stored in `seed.txt` for reproducibility.

### YOLO bbox dataset import

Import external YOLO **object-detection** datasets (bbox labels) and convert
them into the segmentation layout this pipeline expects (class `0` Person
polygons). Compatible sources include any standard YOLO detection export with
`data.yaml` and `train/valid/test` split folders — for example a
[Roboflow Universe face dataset](https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i).

1. Download a YOLO detection export and place it under a bbox source directory
   (e.g. `data/bbox_src/`).
2. Import into your raw dataset:

```bash
# Bbox source stays separate. Import converts to segmentation under dataset_raw:
uv run yolo-bbox-import /path/to/bbox_src /path/to/data/dataset_raw \
  --no-prefix --overwrite

uv run yolo-segment-extractor /path/to/data/dataset_raw
uv run yolo-augmentor /path/to/data/dataset_raw /path/to/data/dataset_augmented --seed 42
```

Use `--no-prefix` when `dataset_raw` contains only this import. The default
`imported_face_` prefix avoids collisions when mixing with video frames from
`yolo-raw-extractor`. Use `--dry-run` to preview and `--map-all-classes` if class
names are unexpected.

Do not run `yolo-augmentor` on the bbox source directly. Its labels are object
detection boxes (`class cx cy w h`), not segmentation polygons.

### Testing

Unit tests are located in [`tests/`](tests). Run them with `uv run --group dev pytest` from this directory, or `make test-extractor` from the repository root. For the full layout, coverage workflow, and contribution conventions, see [`tests/README.md`](../../tests/README.md) at the repository root.
