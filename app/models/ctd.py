"""CTD-only page localization; no language filtering or story classification."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import cv2

from app.schemas.page import BoundingBox, TextRegion


def ctd_block_bbox(block: Any) -> BoundingBox:
    """Convert CTD ``bounding_rect()`` (x, y, width, height) to page xyxy."""
    rect = getattr(block, "bounding_rect", None)
    if rect is None:
        raise ValueError("CTD block has no bounding_rect")
    values = rect() if callable(rect) else rect
    if not isinstance(values, (list, tuple)) and not hasattr(values, "__iter__"):
        raise ValueError("CTD bounding_rect must contain four coordinates")
    try:
        x, y, width, height = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("CTD bounding_rect must contain four numeric values") from exc
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("CTD bounding_rect contains a non-finite coordinate")
    if width <= 0 or height <= 0:
        raise ValueError("CTD bounding_rect must have positive width and height")
    return BoundingBox(x1=x, y1=y, x2=x + width, y2=y + height)


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = width * height
    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def suppress_duplicate_ctd_blocks(
    blocks: list[Any],
    iou_threshold: float = 0.85,
) -> list[tuple[int, Any]]:
    """Keep the first CTD output for each near-identical bbox.

    Returned indices are the original detector output positions, preserving
    source identity even when an earlier duplicate is removed.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    kept: list[tuple[int, Any, BoundingBox]] = []
    for source_index, block in enumerate(blocks):
        bbox = ctd_block_bbox(block)
        if any(_iou(bbox, kept_bbox) >= iou_threshold for _, _, kept_bbox in kept):
            continue
        kept.append((source_index, block, bbox))
    return [(source_index, block) for source_index, block, _ in kept]


def _actual_confidence(block: Any) -> float | None:
    """Read only a genuine confidence-like attribute; CTD's `prob` is a constant."""
    for name in ("confidence", "score"):
        value = getattr(block, name, None)
        if value is None or isinstance(value, bool):
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result) and 0.0 <= result <= 1.0:
            return result
    return None


class _OnnxRuntimeNet:
    """Small OpenCV-DNN-compatible wrapper backed by local ONNX Runtime."""

    def __init__(self, model_path: Path, fallback_net: Any) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
        if not providers:
            raise RuntimeError("ONNX Runtime has no usable execution provider")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.execution_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_blob: Any | None = None
        self.fallback_net = fallback_net
        self.fallback_output_names = fallback_net.getUnconnectedOutLayersNames()

    def setInput(self, blob: Any) -> None:  # noqa: N802 - mirrors cv2.dnn.Net
        self.input_blob = blob

    def forward(self, output_names: list[str]) -> list[Any]:
        if self.input_blob is None:
            raise RuntimeError("CTD input was not set")
        try:
            return self.session.run(
                output_names,
                {self.input_name: self.input_blob},
            )
        except Exception:  # noqa: BLE001 - retry on the original OpenCV CPU net
            self.fallback_net.setInput(self.input_blob)
            self.execution_provider = "OpenCV-DNN-CPU fallback"
            return self.fallback_net.forward(self.fallback_output_names)


class CTDPageLocalizer:
    """Convert the vendored TextDetector's block output into validated regions."""

    def __init__(self, detector: Any, duplicate_iou_threshold: float = 0.85) -> None:
        self.detector = detector
        self.duplicate_iou_threshold = duplicate_iou_threshold

    def localize(self, image_path: str | Path) -> list[TextRegion]:
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read page image: {path}")
        _, _, blocks = self.detector(image)
        runtime_net = getattr(getattr(self.detector, "net", None), "model", None)
        if getattr(runtime_net, "execution_provider", None):
            self.detector.execution_provider = runtime_net.execution_provider
            if runtime_net.execution_provider.startswith("OpenCV-DNN-CPU"):
                self.detector.backend = "opencv"
        regions: list[TextRegion] = []
        for source_index, block in suppress_duplicate_ctd_blocks(
            list(blocks or []), self.duplicate_iou_threshold
        ):
            regions.append(
                TextRegion(
                    id=f"ctd-region-{source_index:04d}",
                    bbox=ctd_block_bbox(block),
                    raw_text="",
                    confidence=_actual_confidence(block),
                    category="unknown",
                )
            )
        return regions


def load_ctd_detector(
    model_path: str | Path,
    *,
    device: str = "cpu",
    input_size: int = 1024,
) -> Any:
    """Load the vendored CTD implementation explicitly, never at import time."""
    model_path = Path(model_path).resolve()
    vendor_dir = Path(__file__).resolve().parents[2] / "vendor" / "comic-text-detector"
    if not model_path.is_file():
        raise FileNotFoundError(f"CTD model not found: {model_path}")
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    try:
        from inference import TextDetector
    except ImportError as exc:
        raise RuntimeError("Unable to import the vendored CTD implementation") from exc
    detector = TextDetector(
        model_path=str(model_path),
        input_size=input_size,
        device=device,
    )
    if model_path.suffix.lower() == ".onnx":
        try:
            opencv_net = detector.net.model
            runtime_net = _OnnxRuntimeNet(model_path, opencv_net)
            detector.net.model = runtime_net
            detector.net.uoln = runtime_net.output_names
            detector.execution_provider = runtime_net.execution_provider
            detector.backend = f"onnxruntime:{runtime_net.execution_provider}"
        except Exception:  # noqa: BLE001 - retain the known OpenCV CPU fallback
            detector.execution_provider = "OpenCV-DNN-CPU"
    return detector
