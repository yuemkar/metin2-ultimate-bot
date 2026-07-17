from __future__ import annotations

import logging
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OCRReader:
    """Optional passive OCR helper for offline analysis and metadata capture."""

    def __init__(self, config: Any | None = None) -> None:
        self.enabled = bool(self._config_get(config, "ocr.enabled", False))
        self.lang = str(self._config_get(config, "ocr.lang", "tur"))
        self.keywords = [str(item) for item in self._config_get(config, "ocr.keywords", [])]
        self.tesseract_path = str(
            self._config_get(config, "ocr.tesseract_path", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        )
        self.available = False
        self._pytesseract = None
        self._configure_tesseract()

    def read_text(self, roi: np.ndarray) -> str:
        if not self.enabled or not self.available or roi is None or roi.size == 0:
            return ""

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        from PIL import Image

        image = Image.fromarray(threshold)

        try:
            return self._pytesseract.image_to_string(image, lang=self.lang, config="--psm 6").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR passive read skipped: %s", exc)
            return ""

    def contains_keyword(self, text: str, keywords: list[str] | None = None) -> bool:
        if not text.strip():
            return False
        active_keywords = keywords or self.keywords
        normalized_text = self._normalize(text)
        return any(self._normalize(keyword) in normalized_text for keyword in active_keywords)

    def _configure_tesseract(self) -> None:
        if not self.enabled:
            return

        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract kurulu degil; OCR pasif kalacak")
            return

        configured_path = Path(self.tesseract_path)
        if configured_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(configured_path)
            self.available = True
        elif shutil.which("tesseract"):
            self.available = True
        else:
            logger.warning("Tesseract bulunamadi; OCR pasif kalacak")
            return

        self._pytesseract = pytesseract

    def _config_get(self, config: Any | None, dotted_key: str, default: Any) -> Any:
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(dotted_key, default)
        value: Any = config
        for part in dotted_key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def _normalize(self, value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value.casefold())
        return "".join(char for char in decomposed if not unicodedata.combining(char))
