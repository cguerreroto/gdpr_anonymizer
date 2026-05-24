# Testing

Test layout, commands, and coverage workflow for `gdpr_anonymizer`.

This document describes how tests are organized and executed. Setup and the contribution workflow are documented in [CONTRIBUTING.md](../CONTRIBUTING.md). Profiling is documented in [PERFORMANCE.md](../PERFORMANCE.md).

## Test layout

| Test suite | Directory | Tool | Run target |
| --- | --- | --- | --- |
| Extractor (Python) | [`extractor/yolo_raw_extractor/tests`](../extractor/yolo_raw_extractor/tests) | `pytest` | `make test-extractor` |
| ML pipeline (Python) | [`segmentator/ml_pipeline/tests`](../segmentator/ml_pipeline/tests) | `pytest` | `make test-ml-pipeline` |
| Segmentator (Rust) | inline `#[cfg(test)]` blocks (mainly [`segmentator/src/dataset.rs`](../segmentator/src/dataset.rs)) | `cargo test` | `make test-rust` |

CI runs the same suites on push and pull request to `main` (see [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).

## Run all tests

From the repository root:

```bash
make test
```

This target runs `cargo test` in `segmentator`, followed by `pytest` in both Python packages.

## Run a single suite

```bash
make test-rust          # segmentator only
make test-extractor     # extractor/yolo_raw_extractor only
make test-ml-pipeline   # segmentator/ml_pipeline only
```

The underlying tools may also be invoked directly:

```bash
cd extractor/yolo_raw_extractor && uv run --group dev pytest
cd segmentator/ml_pipeline && uv run --group dev pytest
cd segmentator && cargo test
```

Additional `pytest` flags (for example `-k expression`, `-x`, `-vv`) are appended to the `uv run --group dev pytest` invocation.

## Coverage

Coverage is measured with [pytest-cov](https://pytest-cov.readthedocs.io/) and [coverage.py](https://coverage.readthedocs.io/). Each Python package defines `[tool.coverage]` in its `pyproject.toml` (`branch = true`, source set to the package).

From the repository root:

```bash
make coverage
```

This target runs `pytest --cov` in both Python packages and produces the following outputs:

| Output | Location | Purpose |
| --- | --- | --- |
| Terminal report | stdout | Summary of percentages and missing line numbers |
| HTML report | `<package>/htmlcov/index.html` | Browseable report for identifying gaps |
| XML report | `<package>/coverage.xml` | Upload for CI (Codecov, Cobertura) |

All three outputs are listed in `.gitignore`.

Coverage for a single package:

```bash
make coverage-extractor
make coverage-ml-pipeline
```

Equivalent direct invocations:

```bash
cd extractor/yolo_raw_extractor && uv run --group dev pytest --cov --cov-report=term-missing
cd segmentator/ml_pipeline && uv run --group dev pytest --cov --cov-report=term-missing
```

The `--cov-report=term-missing` option lists line ranges that no test executed. Those ranges indicate where additional tests are warranted.

Rust coverage is not included in `make coverage`. Local Rust coverage may be produced with `cargo llvm-cov` (install via `cargo install cargo-llvm-cov`). That workflow is optional and is not run in CI.

## Writing tests

### Python

Tests follow the AAA pattern (Arrange, Act, Assert). Each test should remain focused on a single behavior:

```python
def test_normalize_labels_creates_split_directories(tmp_path):
    dataset = build_minimal_dataset(tmp_path)

    normalize_labels(dataset)

    assert (dataset / "labels" / "train").is_dir()
```

Conventions used in the existing suites:

- File names: `tests/test_<module>.py`, mirroring the source module name.
- Test names: `test_<unit>_<expected behavior>` so that failures remain self-explanatory.
- Filesystem fixtures: `tmp_path` is preferred over manual cleanup.
- External tools (Ultralytics, OpenCV writers) are mocked. Weights are not downloaded and real video files are not opened in tests.
- Shared state between tests is avoided. Each test constructs a fresh dataset or fixture.
- Heavy or training-dependent commands are excluded from CI. The default workflow runs without `--extra train`.

### Rust

`cfg(test)` modules are kept adjacent to the code under test (see [`segmentator/src/dataset.rs`](../segmentator/src/dataset.rs)). `cargo fmt` and `cargo clippy -- -D warnings` should be run before a pull request is opened.

## Mocking (Python)

The Python standard library [`unittest.mock`](https://docs.python.org/3/library/unittest.mock.html) is used together with pytest [`monkeypatch`](https://docs.pytest.org/en/stable/how-to/monkeypatch.html) to form the standard approach in this repository for mocking external dependencies in Python unit tests.

### Dependency constraints that require mocks

CI and the default local install use `uv sync --group dev` in each Python package. That environment does not include the Ultralytics `[train]` extra. In `segmentator/ml_pipeline`, `numpy` and `opencv` are also absent from the dev environment unless optional extras are installed locally. Heavy dependencies must therefore be replaced in tests rather than loaded from real videos, weights, or GPU inference.

The extractor package includes `opencv-python` as a dependency. Tests in that package may use real NumPy arrays, while file I/O or `cv2` calls remain mocked when isolation is required.

### Patterns used in this repo

1. Inject factories. Callable arguments such as `model_factory` or `pipeline_runner` are passed into drivers such as `run_video_blur` or `run_export` so that Ultralytics is never loaded in the test.
2. `monkeypatch.setattr`. A function or method on the module under test is replaced (for example `extract_frames` or `_default_video_pipeline_runner`).
3. Lazy imports via `sys.modules`. When production code performs `import cv2` or `import numpy` inside a function, lightweight fake modules are registered with `monkeypatch.setitem(sys.modules, "cv2", fake_cv2)` before the function is called. Shared helpers for video tests are defined in [`segmentator/ml_pipeline/tests/fakes.py`](../segmentator/ml_pipeline/tests/fakes.py) when present.
4. `builtins.__import__`. For optional stacks such as `onnx` or `ultralytics`, `__import__` is wrapped temporarily to simulate `ImportError` or to supply a fake package (see [`test_ultralytics_extra.py`](../segmentator/ml_pipeline/tests/test_ultralytics_extra.py) and [`test_export.py`](../segmentator/ml_pipeline/tests/test_export.py)).

### Minimal examples

A module-level collaborator may be stubbed as follows:

```python
def test_run_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(my_module, "heavy_work", lambda: 0)
    assert my_module.run() == 0
```

`cv2` may be faked for a function that imports it locally:

```python
import sys
from types import ModuleType

def test_blur_uses_fake_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cv2 = ModuleType("cv2")
    fake_cv2.GaussianBlur = lambda frame, k, s: frame
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    result = video_mod._apply_blur(frame, config)
    assert result is frame
```

### What not to mock

- Types that are not owned by this project should not be mocked without a clear boundary. Factory injection at the project API is preferred.
- `.pt` weights must not be downloaded and large video fixtures must not be committed for unit tests.
- The default `make test` and CI workflow must not require `uv sync --extra train`.

Further reading: [unittest.mock](https://docs.python.org/3/library/unittest.mock.html), [pytest monkeypatch](https://docs.pytest.org/en/stable/how-to/monkeypatch.html), and [Python mock library overview](https://www.geeksforgeeks.org/python/python-mock-library/) (third-party tutorial).

## Adding a test for an uncovered line

1. `make coverage` or the per-package coverage target is executed.
2. A missing line is identified from the terminal report or from `htmlcov/index.html`.
3. A case in `tests/test_<module>.py` is added to exercise that path, with external dependencies mocked.
4. `make coverage` is run again to confirm that the line is covered.
5. The new test is committed in the same pull request as the related code change.

## Related documentation

- Setup, dependencies, and lockfiles: [CONTRIBUTING.md](../CONTRIBUTING.md)
- Profiling and bottlenecks: [PERFORMANCE.md](../PERFORMANCE.md)
- Pipeline and CLI reference: [README.md](../README.md)
