# Performance notes

Use this page when you profile the pipeline or need to understand, compare, or explain where time is spent. It gives runnable commands per stage, typical bottlenecks, and recorded decisions about what this repository does not try to speed up. For everyday CLI usage and flags, see [README.md](README.md).

Profiling here uses Python stdlib (`cProfile`), optional `py-spy` from each package’s dev group, and optionally `cargo flamegraph` for the segmentator. Save profiles and graphs under gitignored `local/` or `out/`.

## Pipeline stages worth measuring

| Stage | Command | Package directory |
| --- | --- | --- |
| Frame sampling | `yolo-raw-extractor` | `extractor/yolo_raw_extractor` |
| Dataset augmentation | `yolo-augmentor` | `extractor/yolo_raw_extractor` |
| Video anonymization | `gdpr-yolo-blur-video` | `segmentator/ml_pipeline` |

Training (`gdpr-yolo-train`) and validation are dominated by Ultralytics and GPU runtime. Tune batch size, image size, and device there rather than micro-optimizing this repository’s thin CLI wrappers.

The Rust segmentator is interactive (egui). Profile it only if UI responsiveness is an issue. Batch throughput is not its goal.

Numbered sections below are top-level toggles. Lettered items (for example 1.A) are nested under the parent number.

<details>
<summary><strong>1. Prerequisites</strong> (install profiling tools once per machine, not run in CI)</summary>

| Tool | Install | Verify |
| --- | --- | --- |
| `cProfile` / `pstats` | Shipped with Python | `uv run python -m cProfile --help` |
| `py-spy` | `uv sync --group dev` in each Python package (see below) | `uv run py-spy --version` |
| `cargo flamegraph` | `cargo install flamegraph` (not a `Cargo.toml` dependency; it is a Cargo subcommand) | `cargo flamegraph --version` |

Python packages: from each directory, sync dev tools (includes `pytest` and `py-spy`):

```bash
cd extractor/yolo_raw_extractor && uv sync --group dev
cd segmentator/ml_pipeline && uv sync --group dev
```

For video blur profiling, also run `uv sync --extra train` in `segmentator/ml_pipeline`.

Rust segmentator: optional flamegraph support is configured in [`segmentator/Cargo.toml`](segmentator/Cargo.toml) via a `[profile.profiling]` profile (debug symbols for readable graphs). You still install the `flamegraph` subcommand yourself with `cargo install flamegraph`.

<details>
<summary><strong>1.A macOS: `cargo flamegraph` and Xcode</strong></summary>

On macOS, `cargo flamegraph` uses Apple’s `xctrace`. Command Line Tools alone are not enough. If you see:

```text
xcode-select: error: tool 'xctrace' requires Xcode, but active developer directory
'/Library/Developer/CommandLineTools' is a command line tools instance
```

do one of the following:

1. Install Xcode from the App Store (not Command Line Tools). Verify `ls /Applications/Xcode.app` succeeds, then:

   ```bash
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   sudo xcodebuild -license accept   # only after Xcode is installed and selected above
   ```

2. Or skip Rust flamegraphs on this Mac and profile Python stages only (`cProfile`, `py-spy`).

</details>

<details>
<summary><strong>1.B Virtual environments and output directory</strong></summary>

Use each package’s own `.venv` via `uv run` from that package directory. If you see a warning that `VIRTUAL_ENV=.../gdpr_anonymizer/.venv` does not match the project environment, run `deactivate` (or open a shell without activating the repository-root venv) before `cd extractor/yolo_raw_extractor` or `cd segmentator/ml_pipeline`.

From the repository root:

```bash
mkdir -p out
```

</details>

</details>

<details>
<summary><strong>2. Smoke tests</strong> (confirm tooling before long runs, no real video or weights)</summary>

<details>
<summary><strong>2.A Python `cProfile` (extractor)</strong></summary>

```bash
cd extractor/yolo_raw_extractor
uv sync --group dev
uv run python -m cProfile -o ../../out/smoke-extractor.prof \
  "$(uv run python -c 'import shutil; print(shutil.which("yolo-raw-extractor"))')" \
  --help
uv run python -m pstats ../../out/smoke-extractor.prof -c quit
```

</details>

<details>
<summary><strong>2.B Python `py-spy` (extractor)</strong></summary>

```bash
cd extractor/yolo_raw_extractor
uv run py-spy --version
uv run py-spy record -o ../../out/smoke-extractor.svg -- \
  yolo-raw-extractor --help
```

(`--help` exits quickly. For a real profile, pass a video path and dataset directory.)

</details>

<details>
<summary><strong>2.C Rust `cargo flamegraph` (segmentator)</strong></summary>

Requires `cargo install flamegraph` and macOS full Xcode if on Mac (see section 1), then:

```bash
cd segmentator
cargo flamegraph --profile profiling --bin yolo-segmentator
```

Use the app briefly, then quit. Output is usually `flamegraph.svg` in `segmentator/`. The `profiling` profile avoids the “profiling without debuginfo” warning for release builds.

On macOS, `flamegraph.svg` is often missing while `cargo-flamegraph.trace/` is still valid. For recording commands and a Call Tree screenshot workflow in Instruments (slides or PowerPoint), see [docs/flamegraph/README.md](docs/flamegraph/README.md) (under gitignored `docs/`).

</details>

</details>

<details>
<summary><strong>3. Profiling commands with real inputs</strong> (output under `out/` at repository root)</summary>

Paths below assume you run from the package directory shown. Profile output goes to `out/` at the repository root (`../../out/...` from these packages).

<details>
<summary><strong>3.A Frame extraction (`yolo-raw-extractor`)</strong></summary>

```bash
cd extractor/yolo_raw_extractor
uv sync --group dev
uv run python -m cProfile -o ../../out/extractor.prof \
  "$(uv run python -c 'import shutil; print(shutil.which("yolo-raw-extractor"))')" \
  /path/to/video.mp4 /path/to/dataset
```

Or:

```bash
uv run py-spy record -o ../../out/extractor.svg -- \
  yolo-raw-extractor /path/to/video.mp4 /path/to/dataset
```

Inspect `cProfile` output:

```bash
uv run python -m pstats ../../out/extractor.prof
```

</details>

<details>
<summary><strong>3.B Augmentation (`yolo-augmentor`)</strong></summary>

```bash
cd extractor/yolo_raw_extractor
uv run python -m cProfile -o ../../out/augment.prof \
  "$(uv run python -c 'import shutil; print(shutil.which("yolo-augmentor"))')" \
  /path/to/dataset
```

</details>

<details>
<summary><strong>3.C Video blur (`gdpr-yolo-blur-video`)</strong></summary>

```bash
cd segmentator/ml_pipeline
uv sync --group dev --extra train
uv run py-spy record -o ../../out/blur.svg -- \
  gdpr-yolo-blur-video /path/to/best.pt --source /path/to/input.mp4 --output /path/to/out.mp4
```

</details>

<details>
<summary><strong>3.D Rust segmentator (optional)</strong></summary>

```bash
cd segmentator
cargo flamegraph --profile profiling --bin yolo-segmentator
```

</details>

</details>

<details>
<summary><strong>4. Scenario: Frame extraction</strong> (bottlenecks and decision)</summary>

### Input

A single MP4 at 1080p or 4K, default frame stride (`FRAME_STRIDE = 100` in `yolo_raw_extractor/__init__.py`): every 100th frame is decoded and written as JPEG under `images/<split>/`.

### Tool

`cProfile` or `py-spy` commands in section 3 above.

### Bottlenecks

- `cv2.VideoCapture.read` and `cv2.imwrite` dominate: sequential decode and disk I/O per saved frame.
- Python loop overhead is small compared to OpenCV and the filesystem.

### Decision

No in-repo optimization applied. Throughput scales with stride (fewer writes) and storage speed. For very long sources, increasing stride or preprocessing with a dedicated ffmpeg pipeline (outside this repo) is the practical lever.

</details>

<details>
<summary><strong>5. Scenario: Augmentation</strong> (bottlenecks and decision)</summary>

### Input

A labeled dataset with hundreds of images and segment assets. Default augmentor pass composites segment crops onto training images and rewrites labels.

### Tool

`yolo-augmentor` profiling commands in section 3 above.

### Bottlenecks

- Per-image `cv2` reads, alpha blending, and PNG/JPEG writes under `images/` and `labels/`.
- YAML and label parsing per file. Cost grows with image count, not model size.

### Decision

No in-repo optimization applied. The augmentor is run occasionally during dataset preparation, not on every video frame. Correctness and deterministic layout matter more than shaving milliseconds per image.

</details>

<details>
<summary><strong>6. Scenario: Video blur</strong> (bottlenecks and decision)</summary>

### Input

A trained `best.pt`, a test clip (for example 30 to 120 seconds at 1080p), default blur settings.

### Tool

`gdpr-yolo-blur-video` py-spy command in section 3 above, or Ultralytics progress logging. Compare CPU vs GPU (`device` in predict kwargs).

### Bottlenecks

- Per-frame Ultralytics `predict` (segmentation inference) is the largest cost when the model runs on CPU.
- OpenCV mask union, dilation, and Gaussian or pixelate blur on detected regions are secondary but visible at high resolution or high FPS.
- `cv2.VideoCapture` and writing the output video add fixed I/O overhead.

### Decision

No in-repo optimization applied. Use GPU when available, lower input resolution or frame rate for drafts, and restrict `--classes` when only some categories need anonymization. Further gains belong in model choice and Ultralytics export (ONNX, TensorRT), documented in the README export section.

</details>

<details>
<summary><strong>7. Scenario: Rust segmentator (optional)</strong> (UI profiling, bottlenecks and decision)</summary>

### Input

Opening a dataset with hundreds of images and stepping through annotations.

### Tool

`cargo flamegraph --profile profiling` as in section 3 and [docs/flamegraph/README.md](docs/flamegraph/README.md). On macOS you need full Xcode (section 1.A).

### Bottlenecks

- Image decode and egui texture upload for large frames.
- Label file read/write on save. Watcher notifications add cost if the dataset directory is on a slow network mount.

### Decision

No in-repo optimization applied. The editor is human-paced. Optimize only if profiling shows noticeable UI lag on your hardware.

</details>

<details>
<summary><strong>8. Recording new results</strong> (what to note after a local profiling run)</summary>

When you re-run profiling locally, append dated notes under `local/` (gitignored), for example:

- Hardware (CPU, GPU, RAM, OS)
- Input file names and sizes (not committed)
- Command line used
- Top functions or flamegraph takeaway
- Whether you changed code or only parameters

Update the scenario sections above if measured bottlenecks differ on your setup.

</details>

## Related documentation

- Pipeline overview: [README.md](README.md)
- Commands and flags: collapsible sections in [README.md](README.md)
- Segmentator Instruments and Call Tree slides: [docs/flamegraph/README.md](docs/flamegraph/README.md) (gitignored `docs/`)
- Tests and CI: [CONTRIBUTING.md](CONTRIBUTING.md)
