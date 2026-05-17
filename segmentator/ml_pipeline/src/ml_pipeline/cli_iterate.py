"""CLI entrypoint for the YOLO26-seg warm-start iteration workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_pipeline.cli_train import _resolve_dataset_yaml
from ml_pipeline.iterate import IterateConfig, run_iteration


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-iterate",
        description=(
            "Warm-start a new YOLO26-seg training run from a previous run's "
            "weights/best.pt, optionally re-validate against the same split, "
            "and compare the new metrics with a prior validation report. "
            "Requires Ultralytics for non-dry runs (uv sync --extra train). "
            "Tests inject stub runners."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Iterate with auto-named run (yolo26n_seg_v1 -> yolo26n_seg_v2):\n"
            "    gdpr-yolo-iterate <runs>/yolo26n_seg_v1 <dataset>\n"
            "  Iterate, validate, and compare with the previous metrics report:\n"
            "    gdpr-yolo-iterate <runs>/yolo26n_seg_v1 <dataset> "
            "--compare-to <runs>/yolo26n_seg_v1_val/metrics.json\n"
            "  Train only (skip validation):\n"
            "    gdpr-yolo-iterate <runs>/yolo26n_seg_v1 <dataset> --skip-validation\n"
        ),
    )
    parser.add_argument(
        "previous_run",
        type=Path,
        help="Previous Ultralytics run directory (must contain weights/best.pt).",
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Directory containing dataset.yaml (resolved automatically).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "New run name. Default: bump the previous run's _vN suffix "
            "(or append _v2 when the name has no version)."
        ),
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help=(
            "Directory that will contain the new run subfolder. "
            "Default: the parent of the previous run directory."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help='CUDA device id, "cpu", or "mps". Default: Ultralytics autoselect.',
    )
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every N epochs (-1 disables intermediate saves).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Reuse <project>/<name> when the new run folder already exists.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Train only; do not run gdpr-yolo-validate after training.",
    )
    parser.add_argument(
        "--val-name",
        default=None,
        help="Validation run name. Default: <new-name>_val.",
    )
    parser.add_argument(
        "--val-split",
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to validate against (default: val).",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        metavar="REPORT_JSON",
        help=(
            "Previous validation report to compare with. The new validation "
            "metrics are diffed against this file (overall + per-class)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved train (and val) kwargs without invoking ultralytics.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the iteration report to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_yaml = _resolve_dataset_yaml(args.dataset_root)
    config = IterateConfig(
        previous_run=args.previous_run.resolve(),
        dataset_yaml=dataset_yaml,
        project=args.project.resolve() if args.project is not None else None,
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        save_period=args.save_period,
        workers=args.workers,
        exist_ok=args.exist_ok,
        skip_validation=args.skip_validation,
        val_name=args.val_name,
        val_split=args.val_split,
        compare_to=args.compare_to.resolve() if args.compare_to is not None else None,
    )

    report = run_iteration(config, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
