"""YOLO26-seg training driver for ml_pipeline.

Ultralytics is imported lazily inside the default model factory so unit tests
can inject a stub and run without installing the heavy ML stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
import yaml


class _ModelLike(Protocol):
    def train(self, **kwargs: Any) -> Any: ...


ModelFactory = Callable[[str], _ModelLike]


@dataclass
class TrainConfig:
    dataset_yaml: Path
    model: str = "yolo26n-seg.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str | None = None
    project: Path = Path("runs")
    name: str = "yolo26n_seg"
    patience: int = 50
    save_period: int = -1
    workers: int = 8
    resume: bool = False
    weights: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _default_model_factory(model_ref: str) -> _ModelLike:
    """Import ultralytics only when needed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        msg = (
            "ultralytics is not installed. Install it in this environment "
            "(for example: uv add ultralytics) to run training."
        )
        raise RuntimeError(msg) from exc
    return YOLO(model_ref)


def build_train_kwargs(config: TrainConfig) -> dict[str, Any]:
    """Build keyword arguments passed to YOLO.train()."""
    kwargs: dict[str, Any] = {
        "data": str(config.dataset_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "project": str(config.project),
        "name": config.name,
        "patience": config.patience,
        "save_period": config.save_period,
        "workers": config.workers,
        "resume": config.resume,
        "task": "segment",
    }
    if config.device is not None:
        kwargs["device"] = config.device
    kwargs.update(config.extra)
    return kwargs


def resolve_model_reference(config: TrainConfig) -> str:
    """Return the model path or name passed to the factory.

    When ``weights`` is set, warm-start from those weights. Otherwise use
    ``model`` (typically a YOLO26-seg weight name such as ``yolo26n-seg.pt``).
    """
    if config.weights:
        return config.weights
    return config.model


def validate_dataset_yaml(config: TrainConfig) -> None:
    if not config.dataset_yaml.is_file():
        msg = f"Missing dataset.yaml at {config.dataset_yaml}"
        raise FileNotFoundError(msg)


def materialize_resolved_dataset_yaml(
    dataset_yaml: Path,
    output_dir: Path,
) -> Path:
    """Write a copy of ``dataset.yaml`` with absolute paths.

    Ultralytics resolves a relative ``path`` field against ``DATASETS_DIR``
    or the current working directory, which is fragile when training or
    validation is launched from a folder other than the dataset root. The
    resolved copy has ``path`` rewritten to the absolute dataset root and
    any per-split entry rewritten to an absolute path on disk. The original
    ``dataset.yaml`` is not modified, so it stays portable across machines.
    """
    src_yaml = dataset_yaml.resolve()
    with src_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    declared_path = data.get("path")
    if declared_path is None:
        dataset_root = src_yaml.parent
    else:
        candidate = Path(str(declared_path)).expanduser()
        dataset_root = (
            candidate
            if candidate.is_absolute()
            else (src_yaml.parent / candidate).resolve()
        )
    data["path"] = str(dataset_root)

    for split_key in ("train", "val", "test"):
        value = data.get(split_key)
        if not isinstance(value, str):
            continue
        split_path = Path(value).expanduser()
        if not split_path.is_absolute():
            split_path = (dataset_root / split_path).resolve()
        data[split_key] = str(split_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "dataset.resolved.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return target


def run_training(
    config: TrainConfig,
    *,
    model_factory: ModelFactory | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute training and return a serializable report.

    Parameters
    ----------
    config:
        Training configuration.
    model_factory:
        Callable returning an object with ``.train(**kwargs)``. Defaults to
        the lazy Ultralytics loader. Tests pass a stub.
    dry_run:
        When true, only build the kwargs and skip the call to ``train``.
    """
    validate_dataset_yaml(config)
    kwargs = build_train_kwargs(config)
    model_ref = resolve_model_reference(config)

    report: dict[str, Any] = {
        "config": _config_to_jsonable(config),
        "model_reference": model_ref,
        "train_kwargs": kwargs,
        "dry_run": dry_run,
    }

    if dry_run:
        return report

    resolved_yaml = materialize_resolved_dataset_yaml(
        config.dataset_yaml, config.project / config.name
    )
    kwargs["data"] = str(resolved_yaml)
    report["resolved_dataset_yaml"] = str(resolved_yaml)
    report["train_kwargs"] = kwargs

    factory = model_factory or _default_model_factory
    model = factory(model_ref)
    results = model.train(**kwargs)

    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        report["save_dir"] = str(save_dir)
    else:
        report["save_dir"] = str(config.project / config.name)
    return report


def _config_to_jsonable(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    data["dataset_yaml"] = str(config.dataset_yaml)
    data["project"] = str(config.project)
    return data
