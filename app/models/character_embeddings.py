"""Injectable character embedding providers.

The project uses MobileNetV3-Small (torchvision, BSD-3-Clause / Apache-2.0
ImageNet-1K weights) as its default visual backbone.

Model provenance:
    torchvision.models.mobilenet_v3_small
    Weights:  IMAGENET1K_V1 (9.8 MB, PyTorch model zoo)
    License:  BSD-3-Clause (torchvision); Apache-2.0 (weights per PyTorch hub)
    Input:    224×224 RGB, ImageNet normalisation
    Output:   576-d float32 L2-normalised embedding

The interface is a Protocol so that any embedding backend (ONNX, OpenCLIP,
custom fine-tuned, etc.) can be dropped in without touching the identity logic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, Sequence


class CharacterEmbeddingProvider(Protocol):
    """Embed one crop image into a fixed-length float vector."""

    def embed(self, image_path: Path) -> list[float] | None:
        """Return an L2-normalised float embedding, or None on failure."""
        ...


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two pre-normalised vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Clamp to [-1, 1] for robustness against floating-point noise
    return max(-1.0, min(1.0, dot))


def _l2_norm(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return vec
    return [x / norm for x in vec]


class MobileNetV3EmbeddingProvider:
    """Local MobileNetV3-Small visual embedding (CPU or CUDA).

    Weights are loaded once and cached on the instance.
    Input images are resized to 224×224 and ImageNet-normalised.
    Output: 576-d L2-normalised float32 vector.

    License:   BSD-3-Clause (torchvision) / Apache-2.0 (weights)
    Model zoo: https://pytorch.org/vision/stable/models/generated/
               torchvision.models.mobilenet_v3_small.html
    """

    def __init__(self, *, device: str = "cpu") -> None:
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as tvm
            import torchvision.transforms as T
        except ImportError as exc:
            raise RuntimeError(
                "torch and torchvision are required for MobileNetV3EmbeddingProvider"
            ) from exc

        model = tvm.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self._embedder = nn.Sequential(model.features, model.avgpool, nn.Flatten())
        self._embedder.eval()
        self._device = torch.device(device)
        self._embedder = self._embedder.to(self._device)
        self._torch = torch

        self._transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed(self, image_path: Path) -> list[float] | None:
        try:
            from PIL import Image
        except ImportError:
            # Fall back to cv2-based loading
            return self._embed_cv2(image_path)
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self._transform(img).unsqueeze(0).to(self._device)
            with self._torch.no_grad():
                vec = self._embedder(tensor)[0].cpu().tolist()
            return _l2_norm(vec)
        except Exception:  # noqa: BLE001 - embedding is optional; callers handle None
            return self._embed_cv2(image_path)

    def _embed_cv2(self, image_path: Path) -> list[float] | None:
        try:
            import cv2
            import numpy as np

            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            arr = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            tensor = self._torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
            with self._torch.no_grad():
                vec = self._embedder(tensor)[0].cpu().tolist()
            return _l2_norm(vec)
        except Exception:  # noqa: BLE001
            return None


class NullEmbeddingProvider:
    """Always returns None; useful for tests or when no visual model is available."""

    def embed(self, image_path: Path) -> list[float] | None:
        return None
