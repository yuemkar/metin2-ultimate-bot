from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import AppConfig


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x, y, width, height = self.bbox
        return x + width // 2, y + height // 2


class ColorDetector:
    def __init__(self, config: AppConfig) -> None:
        self.min_area = int(config.get("detection.min_area", 80))
        self.hsv_ranges = config.get("detection.hsv_ranges", [])

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for item in self.hsv_ranges:
            lower = np.array(item["lower"], dtype=np.uint8)
            upper = np.array(item["upper"], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            detections.append(Detection("color_target", min(1.0, area / 5000.0), (x, y, width, height)))

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections


class YoloDetector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model: Any | None = None

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model

        from ultralytics import YOLO

        model_path = Path(self.config.root_dir) / self.config.get("detection.yolo.model_path", "models/yolov8n.pt")
        self.model = YOLO(str(model_path))
        return self.model

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        model = self._ensure_model()
        confidence = float(self.config.get("detection.yolo.confidence", 0.45))
        class_filter = self.config.get("detection.yolo.classes", [])
        classes = class_filter if class_filter else None

        results = model.predict(frame_bgr, conf=confidence, classes=classes, verbose=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                cls = int(box.cls[0].item())
                label = result.names.get(cls, str(cls))
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0].item()),
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                    )
                )
        return detections


class Detector:
    def __init__(self, config: AppConfig) -> None:
        self.mode = str(config.get("detection.mode", "color")).lower()
        self.color = ColorDetector(config)
        self.yolo = YoloDetector(config)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self.mode == "yolo":
            return self.yolo.detect(frame_bgr)
        if self.mode == "hybrid":
            return self.yolo.detect(frame_bgr) + self.color.detect(frame_bgr)
        return self.color.detect(frame_bgr)


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    debug = frame_bgr.copy()
    for detection in detections:
        x, y, width, height = detection.bbox
        cv2.rectangle(debug, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.circle(debug, detection.center, 4, (0, 0, 255), -1)
        cv2.putText(
            debug,
            f"{detection.label} {detection.confidence:.2f}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return debug
