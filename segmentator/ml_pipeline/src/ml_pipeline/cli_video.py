"""CLI entrypoint for YOLO26-seg video blur pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_pipeline.video import (
    VALID_BLUR_METHODS,
    VideoBlurConfig,
    resolve_blurred_output_path,
    run_video_blur,
)


def _parse_classes(value: str | None) -> list[int] | None:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-blur-video",
        description=(
            "Run YOLO26 segmentation on a video and write a copy with the "
            "union of predicted masks blurred or pixelated. Default classes "
            "are 0=Person and 1=Car as set by the segmentator. Requires "
            "Ultralytics + OpenCV (install from segmentator/ml_pipeline with "
            "uv sync --extra train). Tests inject a stub pipeline runner."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Gaussian blur faces and vehicles in a video:\n"
            "    gdpr-yolo-blur-video <runs>/run/weights/best.pt "
            "--source <input.mp4>\n"
            "  Write blurred output under runs/blur/ (auto-named):\n"
            "    gdpr-yolo-blur-video <runs>/run/weights/best.pt "
            "--source <input.mp4> --output <runs>/blur\n"
            "  Pixelate only Person regions:\n"
            "    gdpr-yolo-blur-video <runs>/run/weights/best.pt "
            "--source <input.mp4> "
            "--classes 0 --blur-method pixelate --pixelate-block 24\n"
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
        help="Input video file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. Omitted: write next to --source as "
            "{stem}_blurred{suffix}. Directory (or path without a video "
            "extension): write the auto-named file inside it. Otherwise: "
            "use this exact file path (must differ from --source)."
        ),
    )
    parser.add_argument(
        "--classes",
        default=None,
        metavar="ID,ID,...",
        help="Comma-separated class ids to blur (default: all detected classes).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='CUDA device id, "cpu", or "mps". Default: Ultralytics autoselect.',
    )
    parser.add_argument(
        "--blur-method",
        default="gaussian",
        choices=sorted(VALID_BLUR_METHODS),
        help="Blur strategy applied inside masked regions (default: gaussian).",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=51,
        help="Gaussian blur kernel size; must be odd (default: 51).",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=0.0,
        help="Gaussian blur sigma (default: 0.0 lets OpenCV derive it).",
    )
    parser.add_argument(
        "--pixelate-block",
        type=int,
        default=16,
        help="Block size for the pixelate method (default: 16).",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=0,
        help="Pixels to dilate the union mask before blurring (default: 0).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override the output FPS (default: copied from the source).",
    )
    parser.add_argument(
        "--fourcc",
        default="mp4v",
        help="OpenCV FourCC tag for the writer (default: mp4v).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved kwargs without running the pipeline.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the run report to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    source = args.source.resolve()
    output = resolve_blurred_output_path(
        source,
        args.output.resolve() if args.output is not None else None,
    )
    config = VideoBlurConfig(
        weights=args.weights.resolve(),
        source=source,
        output=output,
        classes=_parse_classes(args.classes),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        blur_method=args.blur_method,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        pixelate_block=args.pixelate_block,
        mask_dilate=args.mask_dilate,
        fps_override=args.fps,
        fourcc=args.fourcc,
    )

    report = run_video_blur(config, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
