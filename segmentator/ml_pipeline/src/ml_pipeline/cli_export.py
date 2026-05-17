"""CLI entrypoint for YOLO26-seg checkpoint export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from ml_pipeline.export import VALID_FORMATS, ExportConfig, run_export


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdpr-yolo-export",
        description=(
            "Export a trained YOLO26 segmentation checkpoint to a portable "
            "format (default ONNX). Ultralytics writes the exported artifact "
            "next to the source weights. Requires Ultralytics (install from "
            "segmentator/ml_pipeline with uv sync --extra train). Tests "
            "inject a stub factory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Export best.pt to ONNX with default opset:\n"
            "    gdpr-yolo-export <runs>/run/weights/best.pt\n"
            "  Export to TorchScript instead:\n"
            "    gdpr-yolo-export <runs>/run/weights/best.pt --format torchscript\n"
        ),
    )
    parser.add_argument(
        "weights",
        type=Path,
        help="Path to the trained .pt checkpoint (for example weights/best.pt).",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        default="onnx",
        choices=sorted(VALID_FORMATS),
        help="Target export format (default: onnx).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size baked into the export (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size baked into the export (default: 1).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 weights (incompatible with --int8).",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export INT8 quantized weights (incompatible with --half).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input axes (ONNX/TensorRT).",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX graph simplification (default is to simplify).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Override the ONNX opset version (default: Ultralytics chooses).",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Embed NMS in the exported model (where supported).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='CUDA device id, "cpu", or "mps". Default: Ultralytics autoselect.',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved export kwargs without invoking ultralytics.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write the export report to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = ExportConfig(
        weights=args.weights.resolve(),
        fmt=args.fmt,
        imgsz=args.imgsz,
        batch=args.batch,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
        opset=args.opset,
        nms=args.nms,
        device=args.device,
    )

    report = run_export(config, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
