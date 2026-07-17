from __future__ import annotations

from time import monotonic

import win32api
import win32con
import win32gui

from src.core.config import AppConfig
from src.core.state import BotState


class MouseController:
    def __init__(self, config: AppConfig) -> None:
        self.click_enabled = bool(config.get("input.click_enabled", False))
        self.move_before_click = bool(config.get("input.move_before_click", True))
        self.click_cooldown_sec = float(config.get("detection.click_cooldown_ms", 450)) / 1000.0
        self.window_hwnd: int | None = None

    def set_window_context(self, hwnd: int | None) -> None:
        self.window_hwnd = hwnd

    def click_at(self, x: int, y: int) -> bool:
        screen_point = self._resolve_window_point(x, y)
        if screen_point is None:
            return False

        win32api.SetCursorPos(screen_point)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_point[0], screen_point[1], 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_point[0], screen_point[1], 0, 0)
        return True

    def click_target(self, screen_point: tuple[int, int], state: BotState) -> bool:
        now = monotonic()
        if not self.click_enabled:
            return False
        if now - state.last_click_at < self.click_cooldown_sec:
            return False

        resolved_point = self._resolve_screen_point(screen_point[0], screen_point[1])
        if resolved_point is None:
            return False

        if self.move_before_click:
            win32api.SetCursorPos(resolved_point)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, resolved_point[0], resolved_point[1], 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, resolved_point[0], resolved_point[1], 0, 0)
        state.last_click_at = now
        state.increment("clicks")
        return True

    def _resolve_screen_point(self, x: int, y: int) -> tuple[int, int] | None:
        if not self.window_hwnd:
            return int(x), int(y)
        if not win32gui.IsWindow(self.window_hwnd) or not win32gui.IsWindowVisible(self.window_hwnd):
            return None

        left, top, right, bottom = win32gui.GetWindowRect(self.window_hwnd)
        screen_x = int(x)
        screen_y = int(y)
        if screen_x < left or screen_x >= right or screen_y < top or screen_y >= bottom:
            return None
        return screen_x, screen_y

    def _resolve_window_point(self, x: int, y: int) -> tuple[int, int] | None:
        if not self.window_hwnd:
            return int(x), int(y)
        if not win32gui.IsWindow(self.window_hwnd) or not win32gui.IsWindowVisible(self.window_hwnd):
            return None

        left, top, right, bottom = win32gui.GetWindowRect(self.window_hwnd)
        screen_x = left + int(x)
        screen_y = top + int(y)
        if screen_x < left or screen_x >= right or screen_y < top or screen_y >= bottom:
            return None
        return screen_x, screen_y
