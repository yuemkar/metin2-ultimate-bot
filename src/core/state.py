from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic


class BotStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class BotState:
    status: BotStatus = BotStatus.IDLE
    started_at: float | None = None
    last_click_at: float = 0.0
    last_error: str | None = None
    selected_target: tuple[int, int] | None = None
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "frames": 0,
            "detections": 0,
            "clicks": 0,
            "errors": 0,
        }
    )

    def mark_started(self) -> None:
        self.status = BotStatus.RUNNING
        self.started_at = monotonic()
        self.last_error = None

    def mark_stopped(self) -> None:
        self.status = BotStatus.STOPPED
        self.selected_target = None

    def increment(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    @property
    def uptime_sec(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, monotonic() - self.started_at)
