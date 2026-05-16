"""CLI entrypoint for YOLO26-seg batch prediction on still images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_pipeline.predict import PredictConfig, run_predict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-predict",
        description=(
            "Run YOLO26 segmentation predict on a folder of still images and "
            "save Ultralytics overlays for human review. Optionally export "
            "predicted masks as YOLO polygon .txt files. Requires Ultralytics "
            "(install from segmentator/ml_pipeline with uv sync --extra train). "
            "Tests inject a stub factory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Predict overlays on a sample folder:\n"
            "    gdpr-yolo-predict <runs>/run/weights/best.pt --source <frames> "
            "--project <runs> --name yolo26n_seg_predict\n"
            "  Also export predicted polygons next to overlays:\n"
            "    gdpr-yolo-predict <runs>/run/weights/best.pt --source <frames> "
            "--save-polygons\n"
        ),
    )
    parser.add_argument(
        "weights",
        type=Path,
        help="Path to the trained .pt checkpoint (for example weights/best.pt).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file or directory of still images to predict on.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS (default: 0.7).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='CUDA device id, "cpu", or "mps". Default: Ultralytics autoselect.',
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs"),
        help="Directory that will contain the predict subfolder (default: runs).",
    )
    parser.add_argument(
        "--name",
        default="yolo26n_seg_predict",
        help="Run name; outputs go under <project>/<name>.",
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
        "--no-overlays",
        action="store_true",
        help="Skip Ultralytics overlay images (save=False).",
    )
    parser.add_argument(
        "--save-polygons",
        action="store_true",
        help=(
            "Write predicted masks as YOLO polygon .txt files under "
            "<project>/<name>/polygons/<stem>.txt."
        ),
    )
    parser.add_argument(
        "--classes",
        default=None,
        metavar="ID,ID,...",
        help="Comma-separated class ids to keep (default: all classes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved predict kwargs without invoking ultralytics.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the predict report to this path.",
    )
    return parser


def _parse_classes(value: str | None) -> list[int] | None:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = PredictConfig(
        weights=args.weights.resolve(),
        source=args.source.resolve(),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project.resolve(),
        name=args.name,
        exist_ok=args.exist_ok,
        save_overlays=not args.no_overlays,
        save_polygons=args.save_polygons,
        classes=_parse_classes(args.classes),
    )

    report = run_predict(config, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
