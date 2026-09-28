from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.page import TextRegion


class OCRBackend(ABC):

    @abstractmethod
    def recognize(self, image_path: str | Path) -> list[TextRegion]:
        """
        Run OCR on a single image and return detected text regions.
        """
        raise NotImplementedError