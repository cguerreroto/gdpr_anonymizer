# Contributing to gdpr_anonymizer

Thank you for your interest in this project. Contributions are welcome under the [PolyForm Noncommercial License 1.0.0](LICENSE). Please read [AUTHORS.md](AUTHORS.md) for attribution expectations.

## Before you start

1. Read the [README](README.md) for project scope and the pipeline overview.
2. Open an [Issue](https://github.com/cguerreroto/gdpr_anonymizer/issues) to discuss larger changes before investing significant effort.
3. Do not commit datasets, model weights, or other large binary artifacts. Paths such as `data/` and `*.pt` are excluded via [`.gitignore`](.gitignore).

## Development setup

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (see `rust-version` in [`segmentator/Cargo.toml`](segmentator/Cargo.toml))
- [uv](https://github.com/astral-sh/uv) for Python packages
- Python 3.13+ for [`extractor/yolo_raw_extractor`](extractor/yolo_raw_extractor)
- Python 3.11+ for [`segmentator/ml_pipeline`](segmentator/ml_pipeline)

### Install dependencies

#### Extractor

From `extractor/yolo_raw_extractor`:

```bash
uv sync
```

#### ML pipeline

From `segmentator/ml_pipeline`:

```bash
uv sync
```

For training, validation, or export commands that call Ultralytics, also install the optional extras documented in the README (`uv sync --extra train` and, when exporting ONNX, `uv sync --extra export`).

#### Segmentator

From `segmentator`:

```bash
cargo build --release
```

### Run tests

From the repository root:

```bash
make test
```

This runs `cargo test` in `segmentator`, then `pytest` in both Python packages.

To run tests in a single package:

```bash
cd extractor/yolo_raw_extractor && uv run --group dev pytest
cd segmentator/ml_pipeline && uv run --group dev pytest
cd segmentator && cargo test
```

## Git workflow

This project follows [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):

1. Create a branch from `main` with a short descriptive name (for example `fix/normalize-labels-symlink`).
2. Make focused commits on that branch.
3. Open a pull request against `main`.
4. Link the pull request to an Issue (`Fixes #123` or `Closes #123` in the description when applicable).
5. Request review and address feedback.
6. Merge after checks pass and review is complete.

Direct pushes to `main` should be avoided once branch protection is enabled.

## Pull requests

Every change should go through a pull request, including documentation updates.

A good pull request:

- Describes what changed and why.
- References the related Issue when one exists.
- Keeps the diff focused on one topic when possible.
- Includes or updates tests when behavior changes.
- Passes `make test` locally before submission.

Use the pull request template provided in [`.github/pull_request_template.md`](.github/pull_request_template.md).

## Code style

### Rust (`segmentator`)

From `segmentator/`:

```bash
cargo fmt
cargo clippy -D warnings
```

Follow existing patterns in `src/`. See [`segmentator/AGENTS.md`](segmentator/AGENTS.md) for component notes aimed at contributors working on the desktop app.

### Python (`extractor`, `ml_pipeline`)

- Match the style of surrounding modules (typing, naming, CLI structure).
- Prefer small functions with clear responsibilities.
- Add or update `pytest` tests for behavioral changes.

## Reporting bugs and suggesting features

Use the GitHub Issue templates under [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/):

- Bug report for incorrect behavior or regressions.
- Feature request for new capabilities or improvements.

Search existing Issues before opening a duplicate.

## Repository settings on GitHub

These items are configured on [github.com](https://github.com/cguerreroto/gdpr_anonymizer), not in Git. Repository owners should complete them once. Contributors can skim this section to understand project expectations.

### About box

Open the repository home page, then click the gear icon next to About.

| Field | What to set |
| --- | --- |
| Description | `Tools for video privacy: YOLO segmentation dataset pipeline and anonymization CLIs.` |
| Website | Leave empty unless you later publish a homepage or GitHub Pages site. The repository URL is enough for most visitors. |
| Topics | Add: `yolo`, `segmentation`, `privacy`, `video`, `rust`, `python` |
| Release | Optional. Not required for development use. |

### Visibility and Issues

Under Settings → General:

- This project is intended to stay public so others can read and contribute. Owners who need a private fork for a specific reason can adjust visibility under the same settings page.
- Under Features, enable Issues so bug reports and feature requests use the templates in [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).

### Issue labels

The issue templates reference these labels. Create them under Settings → Labels if they do not exist yet:

| Label | Suggested color | Used by |
| --- | --- | --- |
| `bug` | Red | Bug report template |
| `enhancement` | Blue | Feature request template |

### Branch protection

Under Settings → Branches → Branch protection rules, add a rule for `main`:

1. Enable Require a pull request before merging. If you work with others, require at least one approval.
2. After continuous integration is configured (see repository workflows), enable Require status checks to pass before merging. Select the CI jobs that must succeed.
3. Avoid force pushes to `main` unless you understand the impact on collaborators.

Reserve direct commits to `main` for urgent documentation typos from maintainers. Feature work should use branches and pull requests as described above.

### Collaborator access

Under Settings → Collaborators, invite anyone who must review pull requests or evaluate the repository. Ensure they can open the repository. For a private fork, confirm their GitHub account has been granted access.

### Optional: Discussions and Projects

Discussions and Projects are not required. Enable them only if you want forum-style threads or a kanban board beyond Issues and pull requests.

## Code of conduct

This project adopts the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.

## Questions

Open a [Discussion](https://github.com/cguerreroto/gdpr_anonymizer/discussions) or an Issue if Discussions are not enabled. For security-sensitive reports, contact the maintainers privately before filing a public Issue.
