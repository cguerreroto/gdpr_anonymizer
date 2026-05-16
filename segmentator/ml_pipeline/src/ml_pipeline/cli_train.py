"""CLI entrypoint for YOLO26-seg training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from ml_pipeline.train import TrainConfig, run_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-train",
        description=(
            "Train a YOLO26 segmentation model on a YOLO-style dataset. "
            "Requires Ultralytics (install from segmentator/ml_pipeline with "
            "uv sync --extra train). Tests inject a stub factory."
        ),
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Directory containing dataset.yaml (resolved automatically).",
    )
    parser.add_argument(
        "--model",
        default="yolo26n-seg.pt",
        help="Base segmentation model reference (default: yolo26n-seg.pt).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Warm-start from a local .pt file instead of the base model.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help='CUDA device id, "cpu", or "mps". Default: Ultralytics autoselect.',
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs"),
        help="Directory that will contain the run subfolder (default: runs).",
    )
    parser.add_argument(
        "--name",
        default="yolo26n_seg",
        help="Run name; final outputs go under <project>/<name>.",
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
        "--resume",
        action="store_true",
        help="Resume training from the run referenced by --project and --name.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help=(
            "Reuse <project>/<name> even when that folder already exists. "
            "Default: Ultralytics creates suffixed names (name-2, name-3, ...)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved training kwargs without invoking ultralytics.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the training report to this path.",
    )
    return parser


def _resolve_dataset_yaml(dataset_root: Path) -> Path:
    if dataset_root.is_file() and dataset_root.suffix in {".yaml", ".yml"}:
        return dataset_root.resolve()
    return (dataset_root / "dataset.yaml").resolve()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_yaml = _resolve_dataset_yaml(args.dataset_root)
    config = TrainConfig(
        dataset_yaml=dataset_yaml,
        model=args.model,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project.resolve(),
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        workers=args.workers,
        resume=args.resume,
        exist_ok=args.exist_ok,
    )

    report = run_training(config, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
