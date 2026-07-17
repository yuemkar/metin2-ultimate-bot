from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from src.vision.detector import Detection
from src.vision.ocr_reader import OCRReader


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
        ocr_reader: OCRReader | None = None,
    ) -> None:
        self.color_profile = color_profile
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.ocr_reader = ocr_reader
        self.last_metadata: list[dict[str, object]] = []

    def analyze_image(self, image_path: str | Path, auto_label: bool = False, class_id: int = 0) -> list[Detection]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Goruntu okunamadi: {image_path}")

        regions = self._find_text_regions(image)
        detections = [
            Detection(
                label="offline_text_region",
                confidence=min(1.0, region.area / 5000.0),
                bbox=region.bbox,
            )
            for region in regions
        ]
        self.last_metadata = [self._detection_to_dict(detection, image) for detection in detections]
        if auto_label:
            self.write_yolo_labels(image_path, detections, image.shape, class_id=class_id)
        return detections

    def auto_label_folder(self, folder_path: str | Path, class_id: int = 0) -> dict[str, object]:
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        image_paths = [
            path
            for path in sorted(folder.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
        labeled = []
        for image_path in image_paths:
            detections = self.analyze_image(image_path, auto_label=True, class_id=class_id)
            labeled.append({"image_path": str(image_path), "label_count": len(detections)})
        return {
            "folder_path": str(folder),
            "image_count": len(image_paths),
            "labeled_images": labeled,
        }

    def write_yolo_labels(
        self,
        image_path: str | Path,
        detections: list[Detection],
        image_shape: tuple[int, ...],
        class_id: int = 0,
    ) -> Path:
        image_path = Path(image_path)
        labels_dir = image_path.parent.parent / "labels" if image_path.parent.name == "images" else image_path.parent
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = labels_dir / f"{image_path.stem}.txt"
        height, width = image_shape[:2]

        lines = []
        for detection in detections:
            x, y, box_width, box_height = detection.bbox
            x_center = (x + box_width / 2) / width
            y_center = (y + box_height / 2) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{normalized_width:.6f} {normalized_height:.6f}"
            )

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return label_path

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
                regions = [self._metadata_for_detection(index, detection) for index, detection in enumerate(detections)]
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

    def _detection_to_dict(self, detection: Detection, image_bgr: np.ndarray | None = None) -> dict[str, object]:
        payload = {
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox": detection.bbox,
            "center": detection.center,
        }
        if self.ocr_reader and self.ocr_reader.enabled and image_bgr is not None:
            x, y, width, height = detection.bbox
            roi = image_bgr[y : y + height, x : x + width]
            text = self.ocr_reader.read_text(roi)
            payload["ocr_text"] = text
            payload["ocr_keyword_match"] = self.ocr_reader.contains_keyword(text)
        return payload

    def _metadata_for_detection(self, index: int, detection: Detection) -> dict[str, object]:
        if index < len(self.last_metadata):
            return self.last_metadata[index]
        return self._detection_to_dict(detection)

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
