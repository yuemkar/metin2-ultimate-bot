from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import win32con
import win32gui
import win32ui

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordResult:
    image_path: str
    label: str
    timestamp: str
    bbox: tuple[int, int, int, int] | None = None
    hp_region: tuple[int, int, int, int] | None = None
    text_regions: list[dict[str, Any]] | None = None


class DataRecorder:
    """Capture screenshots and store lightweight labels for later dataset curation."""

    def __init__(self, root_dir: str | Path = "training_data") -> None:
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"
        self.metadata_path = self.root_dir / "metadata.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def capture_and_save(
        self,
        label: str,
        region: dict[str, int] | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        hp_region: tuple[int, int, int, int] | None = None,
    ) -> RecordResult:
        if not label.strip():
            raise ValueError("label bos olamaz")

        image_bgr = self.capture_screen(region)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_label = self._safe_name(label)
        image_path = self.images_dir / f"{timestamp}_{safe_label}.png"

        if not cv2.imwrite(str(image_path), image_bgr):
            raise RuntimeError(f"Goruntu kaydedilemedi: {image_path}")

        result = RecordResult(
            image_path=str(image_path),
            label=label,
            timestamp=timestamp,
            bbox=bbox,
            hp_region=hp_region,
        )
        self._append_metadata(result)

        if bbox is not None:
            class_index = self._class_index(label)
            self._write_yolo_label(image_path, image_bgr.shape, bbox, class_index)

        logger.info("Training sample saved: %s", image_path)
        return result

    def capture_hp_region(
        self,
        label: str,
        hp_region: tuple[int, int, int, int],
        region: dict[str, int] | None = None,
    ) -> RecordResult:
        """Save a screenshot with passive HP-region metadata for offline analysis."""

        if not label.strip():
            raise ValueError("label bos olamaz")

        image_bgr = self.capture_screen(region)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_label = self._safe_name(f"{label}_hp")
        image_path = self.images_dir / f"{timestamp}_{safe_label}.png"

        if not cv2.imwrite(str(image_path), image_bgr):
            raise RuntimeError(f"HP goruntusu kaydedilemedi: {image_path}")

        result = RecordResult(
            image_path=str(image_path),
            label=label,
            timestamp=timestamp,
            hp_region=hp_region,
        )
        self._append_metadata(result)
        logger.info("HP analysis sample saved: %s", image_path)
        return result

    def save_with_metadata(
        self,
        image_bgr: np.ndarray,
        label: str,
        text_regions: list[dict[str, Any]],
        bbox: tuple[int, int, int, int] | None = None,
        hp_region: tuple[int, int, int, int] | None = None,
    ) -> RecordResult:
        """Save an image and passive analysis regions for dataset labeling workflows."""

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Kaydedilecek goruntu bos")
        if not label.strip():
            raise ValueError("label bos olamaz")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_label = self._safe_name(label)
        image_path = self.images_dir / f"{timestamp}_{safe_label}_metadata.png"
        if not cv2.imwrite(str(image_path), image_bgr):
            raise RuntimeError(f"Goruntu kaydedilemedi: {image_path}")

        result = RecordResult(
            image_path=str(image_path),
            label=label,
            timestamp=timestamp,
            bbox=bbox,
            hp_region=hp_region,
            text_regions=text_regions,
        )
        self._append_metadata(result)
        logger.info("Image with analysis metadata saved: %s", image_path)
        return result

    def capture_with_analysis(
        self,
        label: str,
        analyzer: Any,
        region: dict[str, int] | None = None,
    ) -> RecordResult:
        """Capture a screenshot, run passive offline analysis, and store metadata."""

        image_bgr = self.capture_screen(region)
        detections = analyzer._find_text_regions(image_bgr) if hasattr(analyzer, "_find_text_regions") else []
        text_regions = [self._box_to_metadata(item) for item in detections]
        return self.save_metadata_with_boxes(image_bgr, label, text_regions)

    def capture_with_metadata(
        self,
        label: str,
        analyzer: Any | None = None,
        ocr_reader: Any | None = None,
        region: dict[str, int] | None = None,
    ) -> RecordResult:
        """Capture an image and persist passive analysis/OCR metadata when enabled."""

        image_bgr = self.capture_screen(region)
        text_regions: list[dict[str, Any]] = []

        if analyzer is not None:
            if hasattr(analyzer, "_find_text_regions"):
                boxes = analyzer._find_text_regions(image_bgr)
                text_regions = [self._box_to_metadata(item) for item in boxes]
            if getattr(analyzer, "last_metadata", None):
                text_regions = list(analyzer.last_metadata)

        if ocr_reader is not None and getattr(ocr_reader, "enabled", False):
            for item in text_regions:
                x, y, width, height = [int(value) for value in item.get("bbox", (0, 0, 0, 0))]
                roi = image_bgr[y : y + height, x : x + width]
                text = ocr_reader.read_text(roi)
                item["ocr_text"] = text
                item["ocr_keyword_match"] = ocr_reader.contains_keyword(text)

        return self.save_with_metadata(image_bgr, label, text_regions)

    def save_metadata_with_boxes(
        self,
        image_bgr: np.ndarray,
        label: str,
        boxes: list[Any],
    ) -> RecordResult:
        """Save image, detected boxes, and label metadata together."""

        text_regions = [self._box_to_metadata(item) for item in boxes]
        return self.save_with_metadata(image_bgr, label, text_regions)

    def capture_screen(self, region: dict[str, int] | None = None) -> np.ndarray:
        capture_region = self._normalize_region(region)
        left = capture_region["left"]
        top = capture_region["top"]
        width = capture_region["width"]
        height = capture_region["height"]

        hwnd_desktop = win32gui.GetDesktopWindow()
        desktop_dc = win32gui.GetWindowDC(hwnd_desktop)
        source_dc = win32ui.CreateDCFromHandle(desktop_dc)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source_dc, width, height)
        memory_dc.SelectObject(bitmap)

        try:
            memory_dc.BitBlt((0, 0), (width, height), source_dc, (left, top), win32con.SRCCOPY)
            bitmap_info = bitmap.GetInfo()
            bitmap_bits = bitmap.GetBitmapBits(True)
            image = np.frombuffer(bitmap_bits, dtype=np.uint8)
            image.shape = (bitmap_info["bmHeight"], bitmap_info["bmWidth"], 4)
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        finally:
            win32gui.DeleteObject(bitmap.GetHandle())
            memory_dc.DeleteDC()
            source_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd_desktop, desktop_dc)

    def _append_metadata(self, result: RecordResult) -> None:
        metadata: list[dict[str, Any]]
        if self.metadata_path.exists():
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = []
        else:
            metadata = []

        metadata.append(asdict(result))
        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_yolo_label(
        self,
        image_path: Path,
        image_shape: tuple[int, ...],
        bbox: tuple[int, int, int, int],
        class_index: int,
    ) -> None:
        height, width = image_shape[:2]
        x, y, box_width, box_height = bbox
        x_center = (x + box_width / 2) / width
        y_center = (y + box_height / 2) / height
        normalized_width = box_width / width
        normalized_height = box_height / height
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        label_path.write_text(
            f"{class_index} {x_center:.6f} {y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}\n",
            encoding="utf-8",
        )

    def _class_index(self, label: str) -> int:
        labels = []
        if self.metadata_path.exists():
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                labels = [str(item.get("label", "")).strip() for item in metadata if item.get("label")]
            except json.JSONDecodeError:
                labels = []

        labels.append(label.strip())
        unique_labels = sorted({item for item in labels if item})
        return unique_labels.index(label.strip())

    def _normalize_region(self, region: dict[str, int] | None) -> dict[str, int]:
        if region:
            return {
                "left": int(region["left"]),
                "top": int(region["top"]),
                "width": int(region["width"]),
                "height": int(region["height"]),
            }

        return {
            "left": win32gui.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN),
            "top": win32gui.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN),
            "width": win32gui.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN),
            "height": win32gui.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN),
        }

    def _safe_name(self, value: str) -> str:
        return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value).strip("_") or "label"

    def _box_to_metadata(self, item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            bbox = item.get("bbox")
            label = item.get("label", "offline_text_region")
            confidence = float(item.get("confidence", 0.0))
        else:
            bbox = getattr(item, "bbox", None)
            label = getattr(item, "label", "offline_text_region")
            confidence = float(getattr(item, "confidence", 0.0))

        if bbox is None:
            bbox = tuple(item) if isinstance(item, (list, tuple)) and len(item) == 4 else (0, 0, 0, 0)
        x, y, width, height = [int(value) for value in bbox]
        return {
            "label": label,
            "confidence": confidence,
            "bbox": (x, y, width, height),
            "center": (x + width // 2, y + height // 2),
        }
