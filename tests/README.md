# Testing

Test layout, commands, and coverage workflow for `gdpr_anonymizer`.

This file is the entry point for contributors who run or write tests. For setup and the contribution workflow, see [CONTRIBUTING.md](../CONTRIBUTING.md). For profiling, see [PERFORMANCE.md](../PERFORMANCE.md).

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

That runs `cargo test` in `segmentator`, then `pytest` in both Python packages.

## Run a single suite

```bash
make test-rust          # segmentator only
make test-extractor     # extractor/yolo_raw_extractor only
make test-ml-pipeline   # segmentator/ml_pipeline only
```

Or directly with the underlying tool:

```bash
cd extractor/yolo_raw_extractor && uv run --group dev pytest
cd segmentator/ml_pipeline && uv run --group dev pytest
cd segmentator && cargo test
```

Pass extra flags after `pytest` (for example `-k expression`, `-x`, `-vv`) by appending them to the `uv run --group dev pytest` invocation.

## Coverage

Coverage uses [pytest-cov](https://pytest-cov.readthedocs.io/) and [coverage.py](https://coverage.readthedocs.io/). Both Python packages ship a `[tool.coverage]` configuration in their `pyproject.toml` (`branch = true`, source set to the package).

From the repository root:

```bash
make coverage
```

That runs `pytest --cov` in both Python packages and writes:

| Output | Location | Purpose |
| --- | --- | --- |
| Terminal report | stdout | Quick read of percentage and missing line numbers |
| HTML report | `<package>/htmlcov/index.html` | Browseable report for spotting gaps |
| XML report | `<package>/coverage.xml` | CI upload (Codecov, Cobertura) |

All three outputs are gitignored.

Run coverage for a single package:

```bash
make coverage-extractor
make coverage-ml-pipeline
```

Or directly:

```bash
cd extractor/yolo_raw_extractor && uv run --group dev pytest --cov --cov-report=term-missing
cd segmentator/ml_pipeline && uv run --group dev pytest --cov --cov-report=term-missing
```

`--cov-report=term-missing` prints the line ranges that no test executed. Use it to decide where to add tests next.

Rust coverage is not part of `make coverage`. For local Rust coverage use `cargo llvm-cov` (install with `cargo install cargo-llvm-cov`); it is optional and not run in CI.

## Writing tests

### Python

Follow the AAA pattern (Arrange, Act, Assert) and keep each test focused:

```python
def test_normalize_labels_creates_split_directories(tmp_path):
    dataset = build_minimal_dataset(tmp_path)

    normalize_labels(dataset)

    assert (dataset / "labels" / "train").is_dir()
```

Conventions used in the existing suites:

- File names: `tests/test_<module>.py` mirroring the source module name.
- Test names: `test_<unit>_<expected behavior>` so failures are self-explaining.
- Filesystem fixtures: prefer `tmp_path` over manual cleanup.
- External tools (Ultralytics, OpenCV writers) are mocked. Do not download weights or open real video files in tests.
- No shared state between tests. Build a fresh dataset or fixture per test.
- Heavy or training-dependent commands stay out of CI. Tests should run without `--extra train`.

### Rust

Keep `#[cfg(test)]` modules next to the code they exercise (see [`segmentator/src/dataset.rs`](../segmentator/src/dataset.rs)). Use `cargo fmt` and `cargo clippy -- -D warnings` before opening a pull request.

## Adding a test for an uncovered line

1. Run `make coverage` (or the per-package target).
2. Open the terminal report or `htmlcov/index.html` and pick a missing line.
3. Add a `tests/test_<module>.py` case that drives the code path. Mock external dependencies.
4. Re-run `make coverage` to confirm the line is now covered.
5. Commit the test next to the code change in the same pull request.

## Related documentation

- Setup, dependencies, and lockfiles: [CONTRIBUTING.md](../CONTRIBUTING.md)
- Profiling and bottlenecks: [PERFORMANCE.md](../PERFORMANCE.md)
- Pipeline and CLI reference: [README.md](../README.md)
