from __future__ import annotations

from dataclasses import dataclass

from src.core.state import BotState


@dataclass(frozen=True)
class StatsSnapshot:
    uptime_sec: float
    frames: int
    detections: int
    clicks: int
    errors: int

    @property
    def fps(self) -> float:
        if self.uptime_sec <= 0:
            return 0.0
        return self.frames / self.uptime_sec


def snapshot_stats(state: BotState) -> StatsSnapshot:
    return StatsSnapshot(
        uptime_sec=state.uptime_sec,
        frames=state.counters.get("frames", 0),
        detections=state.counters.get("detections", 0),
        clicks=state.counters.get("clicks", 0),
        errors=state.counters.get("errors", 0),
    )
