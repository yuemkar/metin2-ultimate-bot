from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class DebugOverlay:
    """Draw passive analysis results onto images for visual inspection."""

    def draw_boxes(self, image: np.ndarray, detections: list[Any]) -> np.ndarray:
        output = image.copy()
        for detection in detections:
            x, y, width, height = self._bbox_from_detection(detection)
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.circle(output, (x + width // 2, y + height // 2), 3, (0, 255, 0), -1)
        return output

    def save_debug_image(self, image: np.ndarray, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Debug goruntusu kaydedilemedi: {output_path}")

    def _bbox_from_detection(self, detection: Any) -> tuple[int, int, int, int]:
        if isinstance(detection, dict):
            bbox = detection.get("bbox")
        else:
            bbox = getattr(detection, "bbox", None)
        if bbox is None:
            raise ValueError("Detection icinde bbox yok")
        x, y, width, height = bbox
        return int(x), int(y), int(width), int(height)
