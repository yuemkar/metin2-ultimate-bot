from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class HPTracker:
    """Passive HP-bar template matcher for offline frame analysis."""

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def get_hp_score(self, frame: np.ndarray, template: np.ndarray | str | Path) -> float:
        template_image = self._load_template(template)
        if frame is None or template_image is None:
            return 0.0
        if frame.size == 0 or template_image.size == 0:
            return 0.0

        frame_gray = self._to_gray(frame)
        template_gray = self._to_gray(template_image)
        if template_gray.shape[0] > frame_gray.shape[0] or template_gray.shape[1] > frame_gray.shape[1]:
            return 0.0

        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, _ = cv2.minMaxLoc(result)
        return float(max(0.0, min(1.0, max_value)))

    def is_visible(self, frame: np.ndarray, template: np.ndarray | str | Path) -> bool:
        return self.get_hp_score(frame, template) >= self.threshold

    def _load_template(self, template: np.ndarray | str | Path) -> np.ndarray | None:
        if isinstance(template, np.ndarray):
            return template
        return cv2.imread(str(template), cv2.IMREAD_COLOR)

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
