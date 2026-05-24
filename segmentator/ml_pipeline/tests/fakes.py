"""Shared fakes for ml_pipeline video tests (no real cv2, numpy, or Ultralytics)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable
import pytest


class FakeArray:
    """Minimal ndarray stand-in for lazy-imported numpy in video.py."""

    def __init__(
        self,
        data: list[list[float]] | list[list[list[int]]],
        *,
        dtype: str = "uint8",
    ) -> None:
        self._data = data
        self.dtype = dtype
        self.shape = _shape_of(data)
        self.ndim = len(self.shape)

    def astype(self, dtype: Any) -> FakeArray:
        return FakeArray(self._data, dtype=str(dtype))

    def squeeze(self) -> FakeArray:
        if self.ndim == 3 and self.shape[2] == 1:
            flat = self._data
            if flat and isinstance(flat[0][0], list):
                squeezed = [[row[0] for row in flat]]
                return FakeArray(squeezed, dtype=self.dtype)
        return self

    def sum(self) -> int:
        total = 0.0
        for row in self._data:
            if isinstance(row[0], list):
                for cell in row:
                    total += sum(cell)
            else:
                total += sum(row)
        return int(total)

    def __getitem__(self, key: Any) -> FakeArray:
        if isinstance(key, tuple) and len(key) == 3 and key[2] is None:
            h, w = self.shape[0], self.shape[1]
            if self.ndim == 2:
                expanded = [[[v] for v in row] for row in self._data]
                return FakeArray(expanded, dtype=self.dtype)
            return FakeArray(self._data, dtype=self.dtype)
        raise NotImplementedError(f"FakeArray slice {key!r} is not implemented")

    def __gt__(self, other: float) -> FakeArray:
        if self.ndim == 2:
            binary = [[1 if v > other else 0 for v in row] for row in self._data]
            return FakeArray(binary, dtype="uint8")
        return self

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: str = "uint8") -> FakeArray:
        if len(shape) == 2:
            h, w = shape
            return cls([[0] * w for _ in range(h)], dtype=dtype)
        if len(shape) == 3:
            h, w, c = shape
            return cls([[[0] * c for _ in range(w)] for _ in range(h)], dtype=dtype)
        raise ValueError(f"Unsupported zeros shape: {shape}")

    @classmethod
    def from_scalar_grid(
        cls,
        values: list[list[float]],
        *,
        dtype: str = "float32",
    ) -> FakeArray:
        return cls(values, dtype=dtype)


def _shape_of(data: list[Any]) -> tuple[int, ...]:
    if not data:
        return (0,)
    if isinstance(data[0][0], list):
        return (len(data), len(data[0]), len(data[0][0]))
    return (len(data), len(data[0]))


def make_fake_numpy_module() -> ModuleType:
    """Build a fake ``numpy`` module with the subset used by video.py."""
    np_mod = ModuleType("numpy")

    def zeros(shape: tuple[int, ...], dtype: str = "uint8") -> FakeArray:
        return FakeArray.zeros(shape, dtype=str(dtype))

    def asarray(value: Any) -> FakeArray:
        if isinstance(value, FakeArray):
            return value
        if hasattr(value, "numpy") and callable(value.numpy):
            value = value.numpy()
        if isinstance(value, FakeArray):
            return value
        if isinstance(value, list):
            return FakeArray(value)
        raise TypeError(f"fake numpy cannot asarray {type(value)!r}")

    def maximum(a: FakeArray, b: FakeArray) -> FakeArray:
        if a.shape != b.shape:
            return b
        merged: list[list[int]] = []
        for row_a, row_b in zip(a._data, b._data, strict=True):
            merged.append([max(int(x), int(y)) for x, y in zip(row_a, row_b, strict=True)])
        return FakeArray(merged, dtype="uint8")

    def where(condition: FakeArray, x: Any, y: Any) -> FakeArray:
        if not isinstance(x, FakeArray):
            x = FakeArray.zeros(y.shape, dtype="uint8")
        out_rows: list[list[list[int]]] = []
        for r_idx, cond_row in enumerate(condition._data):
            out_row: list[list[int]] = []
            for c_idx, flag in enumerate(cond_row):
                cell = cond_row[c_idx]
                pick = x._data[r_idx][c_idx] if flag else y._data[r_idx][c_idx]
                if isinstance(pick, list):
                    out_row.append(list(pick))
                else:
                    out_row.append([pick, pick, pick])
            out_rows.append(out_row)
        return FakeArray(out_rows, dtype="uint8")

    np_mod.zeros = zeros
    np_mod.asarray = asarray
    np_mod.maximum = maximum
    np_mod.where = where
    np_mod.uint8 = "uint8"
    np_mod.float32 = "float32"
    np_mod.bool_ = bool
    return np_mod


class FakeCapture:
    """Stand-in for ``cv2.VideoCapture``."""

    def __init__(
        self,
        path: str,
        *,
        frames: list[FakeArray] | None = None,
        opened: bool = True,
        width: int = 64,
        height: int = 48,
        fps: float = 10.0,
        frame_count: int | None = None,
    ) -> None:
        self.path = path
        self._frames = frames if frames is not None else [
            FakeArray.zeros((height, width, 3), dtype="uint8")
            for _ in range(2)
        ]
        self._index = 0
        self._opened = opened
        self._released = False
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count if frame_count is not None else len(self._frames)

    def isOpened(self) -> bool:
        return self._opened and not self._released

    def read(self) -> tuple[bool, FakeArray | None]:
        if not self.isOpened() or self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop: int) -> float:
        if prop == FakeCv2Constants.CAP_PROP_FPS:
            return float(self.fps)
        if prop == FakeCv2Constants.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == FakeCv2Constants.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == FakeCv2Constants.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        return 0.0

    def release(self) -> None:
        self._released = True


class FakeWriter:
    """Stand-in for ``cv2.VideoWriter``."""

    def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int], *, opened: bool = True) -> None:
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self._opened = opened
        self._released = False
        self.frames_written: list[FakeArray] = []

    def isOpened(self) -> bool:
        return self._opened and not self._released

    def write(self, frame: FakeArray) -> None:
        self.frames_written.append(frame)

    def release(self) -> None:
        self._released = True


class FakeCv2Constants:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FRAME_COUNT = 7
    INTER_LINEAR = 1
    INTER_NEAREST = 0
    MORPH_ELLIPSE = 2


def make_fake_cv2_module(
    *,
    capture_factory: Callable[[str], FakeCapture] | None = None,
    writer_factory: Callable[..., FakeWriter] | None = None,
) -> ModuleType:
    """Build a fake ``cv2`` module with the subset used by video.py."""
    cv2_mod = ModuleType("cv2")
    cv2_mod.CAP_PROP_FPS = FakeCv2Constants.CAP_PROP_FPS
    cv2_mod.CAP_PROP_FRAME_WIDTH = FakeCv2Constants.CAP_PROP_FRAME_WIDTH
    cv2_mod.CAP_PROP_FRAME_HEIGHT = FakeCv2Constants.CAP_PROP_FRAME_HEIGHT
    cv2_mod.CAP_PROP_FRAME_COUNT = FakeCv2Constants.CAP_PROP_FRAME_COUNT
    cv2_mod.INTER_LINEAR = FakeCv2Constants.INTER_LINEAR
    cv2_mod.INTER_NEAREST = FakeCv2Constants.INTER_NEAREST
    cv2_mod.MORPH_ELLIPSE = FakeCv2Constants.MORPH_ELLIPSE

    def GaussianBlur(frame: FakeArray, _kernel: tuple[int, int], _sigma: float) -> FakeArray:
        return frame

    def resize(
        image: FakeArray,
        size: tuple[int, int],
        *,
        interpolation: int = 0,
    ) -> FakeArray:
        _ = interpolation
        width, height = size
        if image.ndim == 2:
            return FakeArray.zeros((height, width), dtype="uint8")
        return FakeArray.zeros((height, width, 3), dtype="uint8")

    def getStructuringElement(_shape: int, ksize: tuple[int, int]) -> tuple[int, int]:
        return ksize

    def dilate(image: FakeArray, _kernel: tuple[int, int]) -> FakeArray:
        return image

    def VideoWriter_fourcc(*_chars: str) -> int:
        return 0x7634706D

    def VideoCapture(path: str) -> FakeCapture:
        factory = capture_factory or (lambda p: FakeCapture(p))
        return factory(path)

    def VideoWriter(path: str, fourcc: int, fps: float, size: tuple[int, int]) -> FakeWriter:
        if writer_factory is not None:
            return writer_factory(path, fourcc, fps, size)
        return FakeWriter(path, fourcc, fps, size)

    cv2_mod.GaussianBlur = GaussianBlur
    cv2_mod.resize = resize
    cv2_mod.getStructuringElement = getStructuringElement
    cv2_mod.dilate = dilate
    cv2_mod.VideoWriter_fourcc = VideoWriter_fourcc
    cv2_mod.VideoCapture = VideoCapture
    cv2_mod.VideoWriter = VideoWriter
    return cv2_mod


def install_fake_cv2_numpy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cv2_module: ModuleType | None = None,
    numpy_module: ModuleType | None = None,
) -> tuple[ModuleType, ModuleType]:
    """Register fake ``cv2`` and ``numpy`` modules for lazy imports in video.py."""
    cv2_mod = cv2_module or make_fake_cv2_module()
    np_mod = numpy_module or make_fake_numpy_module()
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)
    monkeypatch.setitem(sys.modules, "numpy", np_mod)
    return cv2_mod, np_mod


class FakeTensor:
    """Tensor-like mask value with optional ``cpu`` / ``numpy`` methods."""

    def __init__(self, array: FakeArray) -> None:
        self._array = array

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> FakeArray:
        return self._array

    def astype(self, dtype: Any) -> FakeArray:
        return self._array.astype(dtype)

    def squeeze(self) -> FakeArray:
        return self._array.squeeze()

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape


def fake_segmentation_result(
    classes: list[int] | None = None,
    masks: list[Any] | None = None,
    *,
    height: int = 48,
    width: int = 64,
) -> SimpleNamespace:
    """Build a YOLO-like result object for ``_result_classes`` and ``_union_mask``."""
    class_values = classes if classes is not None else [0]
    if masks is None:
        masks = [
            FakeArray.from_scalar_grid(
                [[0.0] * width for _ in range(height)],
                dtype="float32",
            )
        ]
        masks[0]._data[height // 4][width // 4] = 1.0

    mask_data = []
    for mask in masks:
        if isinstance(mask, FakeArray):
            mask_data.append(FakeTensor(mask))
        else:
            mask_data.append(mask)

    boxes = SimpleNamespace(cls=class_values)
    mask_container = SimpleNamespace(data=mask_data)
    return SimpleNamespace(boxes=boxes, masks=mask_container)
