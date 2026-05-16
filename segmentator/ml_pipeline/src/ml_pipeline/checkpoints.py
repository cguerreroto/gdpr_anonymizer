"""Helpers for locating YOLO training checkpoints under a runs directory."""

from __future__ import annotations

from pathlib import Path


def best_pt_in_run_dir(save_dir: Path) -> Path | None:
    """Return ``<save_dir>/weights/best.pt`` when that file exists."""
    candidate = save_dir / "weights" / "best.pt"
    return candidate if candidate.is_file() else None


def find_best_pt_under_project(
    project: Path,
    *,
    limit: int = 10,
) -> list[Path]:
    """List ``best.pt`` files under ``project``, newest modification time first."""
    if not project.is_dir():
        return []
    matches = [p for p in project.glob("**/weights/best.pt") if p.is_file()]
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[:limit]


def missing_weights_message(requested: Path, project: Path) -> str:
    """Build an error message when ``--weights`` does not exist on disk."""
    lines = [
        f"Missing weights at {requested.resolve()}",
        "",
        "Ultralytics may have written checkpoints under a suffixed run folder "
        f"(for example {project.name}/<name>-2 instead of <name>).",
    ]
    found = find_best_pt_under_project(project)
    if found:
        lines.append("")
        lines.append(f"Checkpoints found under {project.resolve()}:")
        for path in found:
            lines.append(f"  {path.resolve()}")
        lines.append("")
        lines.append(
            "Pass one of these paths to --weights, or read validate_weights "
            "from the gdpr-yolo-train JSON report."
        )
    else:
        lines.append("")
        lines.append(f"No weights/best.pt files found under {project.resolve()}.")
        lines.append("Train a model first, then pass --weights to gdpr-yolo-validate.")
    return "\n".join(lines)
