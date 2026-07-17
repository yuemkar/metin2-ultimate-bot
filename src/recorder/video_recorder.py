from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32ui


class VideoRecorder:
    """Collect screen frames in memory and save them as training images."""

    def __init__(self, output_dir: str | Path = "training_data", region=None):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.metadata_path = self.output_dir / "video_recordings.json"
        self.region = region
        self.frames: list[tuple[str, np.ndarray]] = []
        self.is_recording = False
        self.active_label = ""
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def start_recording(self, label: str, duration: float | None = None, fps: float = 5) -> list[str]:
        if fps <= 0:
            raise ValueError("fps 0'dan buyuk olmali")
        if not label.strip():
            raise ValueError("label bos olamaz")

        self.frames = []
        self.active_label = label
        self.is_recording = True
        started = time.monotonic()
        interval = 1.0 / fps

        try:
            while self.is_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                self.frames.append((timestamp, self._capture_frame()))

                if duration is not None and time.monotonic() - started >= duration:
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.is_recording = False

        return self.stop_recording()

    def stop_recording(self) -> list[str]:
        self.is_recording = False
        saved_paths: list[str] = []
        if not self.frames:
            return saved_paths

        safe_label = self._safe_name(self.active_label)
        for index, (timestamp, frame_bgr) in enumerate(self.frames, start=1):
            filename = f"{safe_label}_{timestamp}_{index:05d}.png"
            filepath = self.images_dir / filename
            if cv2.imwrite(str(filepath), frame_bgr):
                saved_paths.append(str(filepath))

        self._append_session_metadata(saved_paths)
        self.frames = []
        return saved_paths

    def _capture_frame(self) -> np.ndarray:
        capture_region = self._normalize_region(self.region)
        hwnd_dc = win32gui.GetWindowDC(0)
        dc = win32ui.CreateDCFromHandle(hwnd_dc)
        create_dc = dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(dc, capture_region["width"], capture_region["height"])
        create_dc.SelectObject(bitmap)

        try:
            create_dc.BitBlt(
                (0, 0),
                (capture_region["width"], capture_region["height"]),
                dc,
                (capture_region["left"], capture_region["top"]),
                win32con.SRCCOPY,
            )
            bmpstr = bitmap.GetBitmapBits(True)
            image = np.frombuffer(bmpstr, dtype=np.uint8)
            image = image.reshape((capture_region["height"], capture_region["width"], 4))
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        finally:
            win32gui.DeleteObject(bitmap.GetHandle())
            create_dc.DeleteDC()
            dc.DeleteDC()
            win32gui.ReleaseDC(0, hwnd_dc)

    def _normalize_region(self, region):
        if region is not None:
            x, y, width, height = region
            return {"left": int(x), "top": int(y), "width": int(width), "height": int(height)}
        return {
            "left": win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN),
            "top": win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN),
            "width": win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN),
            "height": win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN),
        }

    def _append_session_metadata(self, saved_paths: list[str]) -> None:
        sessions = []
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as file:
                try:
                    sessions = json.load(file)
                except json.JSONDecodeError:
                    sessions = []

        sessions.append(
            {
                "label": self.active_label,
                "frame_count": len(saved_paths),
                "saved_paths": saved_paths,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(sessions, file, ensure_ascii=False, indent=2)

    def _safe_name(self, value: str) -> str:
        safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
        return safe.strip("_") or "recording"
