from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from src.vision.detector import Detection


@dataclass(frozen=True)
class OfflineRegion:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area: float
    aspect_ratio: float


class OfflineAnalyzer:
    """Passive HSV and contour analyzer for saved screenshots."""

    DEFAULT_PROFILES = {
        "red": [([0, 100, 80], [12, 255, 255]), ([168, 100, 80], [180, 255, 255])],
        "blue": [([90, 80, 70], [130, 255, 255])],
        "green": [([40, 70, 60], [85, 255, 255])],
    }

    def __init__(
        self,
        color_profile: str = "red",
        min_area: int = 60,
        max_area: int = 25000,
        min_aspect: float = 0.2,
        max_aspect: float = 5.0,
    ) -> None:
        self.color_profile = color_profile
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def analyze_image(self, image_path: str | Path) -> list[Detection]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Goruntu okunamadi: {image_path}")

        regions = self._find_text_regions(image)
        return [
            Detection(
                label="offline_text_region",
                confidence=min(1.0, region.area / 5000.0),
                bbox=region.bbox,
            )
            for region in regions
        ]

    def batch_analyze(self, folder_path: str | Path) -> dict[str, object]:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Klasor bulunamadi: {folder}")

        image_paths = [
            path
            for path in sorted(folder.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
        results = []
        for image_path in image_paths:
            try:
                detections = self.analyze_image(image_path)
                regions = [self._detection_to_dict(detection) for detection in detections]
                results.append({"image_path": str(image_path), "detections": regions, "error": None})
            except Exception as exc:  # noqa: BLE001
                results.append({"image_path": str(image_path), "detections": [], "error": str(exc)})

        total_detections = sum(len(item["detections"]) for item in results)
        report = {
            "folder_path": str(folder),
            "image_count": len(image_paths),
            "total_detections": total_detections,
            "total_regions": total_detections,
            "results": results,
        }
        report_path = folder / "offline_analysis_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _detection_to_dict(self, detection: Detection) -> dict[str, object]:
        return {
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox": detection.bbox,
            "center": detection.center,
        }

    def _find_text_regions(self, image_bgr: np.ndarray) -> list[OfflineRegion]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in self.DEFAULT_PROFILES.get(self.color_profile, self.DEFAULT_PROFILES["red"]):
            lower_array = np.array(lower, dtype=np.uint8)
            upper_array = np.array(upper, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_array, upper_array))

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._filter_text_regions(contours)

    def _filter_text_regions(self, contours: list[np.ndarray]) -> list[OfflineRegion]:
        regions: list[OfflineRegion] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if height <= 0:
                continue
            aspect_ratio = width / height
            if aspect_ratio < self.min_aspect or aspect_ratio > self.max_aspect:
                continue

            regions.append(
                OfflineRegion(
                    bbox=(x, y, width, height),
                    center=(x + width // 2, y + height // 2),
                    area=area,
                    aspect_ratio=aspect_ratio,
                )
            )

        return sorted(regions, key=lambda item: item.area, reverse=True)
