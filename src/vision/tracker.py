from __future__ import annotations

from src.vision.detector import Detection


class TargetTracker:
    def select_target(self, detections: list[Detection]) -> Detection | None:
        if not detections:
            return None
        return max(detections, key=lambda detection: detection.confidence)
