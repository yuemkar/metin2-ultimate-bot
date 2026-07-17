from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import win32con
import win32gui
import win32ui

from src.core.config import AppConfig


@dataclass(frozen=True)
class CaptureFrame:
    image_bgr: np.ndarray
    offset: tuple[int, int]

    def to_screen_point(self, local_point: tuple[int, int]) -> tuple[int, int]:
        return self.offset[0] + local_point[0], self.offset[1] + local_point[1]


class ScreenCapture:
    def __init__(self, config: AppConfig) -> None:
        self.region = config.get("capture.region", None)

    def grab(self) -> CaptureFrame:
        region = self._select_region()
        left, top, width, height = region["left"], region["top"], region["width"], region["height"]

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
            image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return CaptureFrame(image_bgr=image_bgr, offset=(left, top))
        finally:
            win32gui.DeleteObject(bitmap.GetHandle())
            memory_dc.DeleteDC()
            source_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd_desktop, desktop_dc)

    def _select_region(self) -> dict[str, int]:
        if self.region:
            return {
                "left": int(self.region["left"]),
                "top": int(self.region["top"]),
                "width": int(self.region["width"]),
                "height": int(self.region["height"]),
            }

        return {
            "left": win32gui.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN),
            "top": win32gui.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN),
            "width": win32gui.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN),
            "height": win32gui.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN),
        }
