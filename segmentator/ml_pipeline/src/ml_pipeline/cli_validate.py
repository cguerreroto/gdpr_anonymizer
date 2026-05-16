"""CLI entrypoint for YOLO26-seg validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_pipeline.cli_train import _resolve_dataset_yaml
from ml_pipeline.validate import ValidateConfig, run_validation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-validate",
        description=(
            "Validate a trained YOLO26 segmentation model on a YOLO-style "
            "dataset and print mask + box mAP. Requires the ultralytics "
            "package; tests inject a stub factory."
        ),
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Directory containing dataset.yaml (or a yaml file path).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help=(
            "Path to the .pt checkpoint to validate "
            "(for example: <project>/<name>/weights/best.pt)."
        ),
    )
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
        help="Directory that will contain the val subfolder (default: runs).",
    )
    parser.add_argument(
        "--name",
        default="yolo26n_seg_val",
        help="Run name; outputs go under <project>/<name>.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to validate against (default: val).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (default: Ultralytics default).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold for NMS (default: Ultralytics default).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Pass save_json=True to Ultralytics to emit COCO-format predictions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved val kwargs without invoking ultralytics.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the validation report to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_yaml = _resolve_dataset_yaml(args.dataset_root)
    config = ValidateConfig(
        weights=args.weights.resolve(),
        dataset_yaml=dataset_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project.resolve(),
        name=args.name,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
    )

    report = run_validation(config, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
