from __future__ import annotations

import logging
from pathlib import Path
from time import monotonic, sleep

import cv2
import win32gui

from src.core.config import AppConfig
from src.core.state import BotState
from src.input.mouse import MouseController
from src.input.window import ScreenCapture
from src.vision.detector import Detector, draw_detections
from src.vision.tracker import TargetTracker

logger = logging.getLogger(__name__)


def find_game_window(title_keywords: tuple[str, ...] = ("Metin2", "Kraliyet2")) -> int | None:
    matches: list[int] = []

    def enum_handler(hwnd: int, _: object) -> None:
        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if any(keyword.lower() in title.lower() for keyword in title_keywords):
            matches.append(hwnd)

    win32gui.EnumWindows(enum_handler, None)
    return matches[0] if matches else None


def is_window_visible(hwnd: int | None) -> bool:
    return bool(hwnd and win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd))


def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    return win32gui.GetWindowRect(hwnd)


def bring_to_front(hwnd: int) -> bool:
    if not is_window_visible(hwnd):
        return False
    win32gui.SetForegroundWindow(hwnd)
    return True


class Bot:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state = BotState()
        self.capture = ScreenCapture(config)
        self.detector = Detector(config)
        self.tracker = TargetTracker()
        self.mouse = MouseController(config)
        self.tick_interval_ms = int(config.get("app.tick_interval_ms", 80))
        self.print_interval_sec = float(config.get("app.print_interval_sec", 1.0))
        self.save_debug_frame = bool(config.get("app.save_debug_frame", True))
        self.debug_frame_path = Path(config.root_dir) / str(config.get("app.debug_frame_path", "debug_frame.jpg"))
        self._last_print_at = 0.0
        self.window_hwnd: int | None = None

    def find_game_window(self) -> int | None:
        title = str(self.config.get("window.window_title", "Metin2"))
        return find_game_window((title, "Metin2", "Kraliyet2"))

    def is_window_visible(self, hwnd: int | None = None) -> bool:
        return is_window_visible(hwnd or self.window_hwnd)

    def get_window_rect(self, hwnd: int | None = None) -> tuple[int, int, int, int] | None:
        target_hwnd = hwnd or self.window_hwnd
        if not target_hwnd or not is_window_visible(target_hwnd):
            return None
        return get_window_rect(target_hwnd)

    def bring_to_front(self, hwnd: int | None = None) -> bool:
        target_hwnd = hwnd or self.window_hwnd
        return bring_to_front(target_hwnd) if target_hwnd else False

    def run(self) -> None:
        self.state.mark_started()
        logger.info("Bot started in terminal mode")
        print("Terminal modu basladi. Durdurmak icin Ctrl+C.")
        print(f"Tiklama aktif: {self.mouse.click_enabled}")

        while self.state.status.value == "running":
            started = monotonic()
            self.tick()
            elapsed = monotonic() - started
            sleep(max(0.0, (self.tick_interval_ms / 1000.0) - elapsed))

    def stop(self) -> None:
        self.state.mark_stopped()
        logger.info("Bot stopped")

    def tick(self) -> None:
        try:
            capture_frame = self.capture.grab()
            detections = self.detector.detect(capture_frame.image_bgr)
            target = self.tracker.select_target(detections)

            self.state.increment("frames")
            if detections:
                self.state.increment("detections", len(detections))
            if target:
                self.state.selected_target = capture_frame.to_screen_point(target.center)
                self.mouse.click_target(self.state.selected_target, self.state)
            else:
                self.state.selected_target = None

            debug_frame = draw_detections(capture_frame.image_bgr, detections)
            if self.save_debug_frame:
                cv2.imwrite(str(self.debug_frame_path), debug_frame)
            self._print_status()
        except Exception as exc:  # noqa: BLE001
            self.state.last_error = str(exc)
            self.state.increment("errors")
            self._print_status(force=True)
            logger.exception("Bot tick failed")

    def _print_status(self, force: bool = False) -> None:
        now = monotonic()
        if not force and now - self._last_print_at < self.print_interval_sec:
            return
        self._last_print_at = now
        target = self.state.selected_target
        target_text = f"{target[0]},{target[1]}" if target else "yok"
        print(
            " | ".join(
                [
                    f"fps~{self.state.counters.get('frames', 0) / max(self.state.uptime_sec, 0.001):.1f}",
                    f"frame={self.state.counters.get('frames', 0)}",
                    f"tespit={self.state.counters.get('detections', 0)}",
                    f"tiklama={self.state.counters.get('clicks', 0)}",
                    f"hedef={target_text}",
                    f"hata={self.state.last_error or '-'}",
                ]
            )
        )
