from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque


@dataclass(frozen=True)
class QueuedTarget:
    detection: Any
    priority: float


class TargetQueue:
    """Passive target-candidate queue for offline debugging and dataset review."""

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max(1, int(max_size))
        self._items: Deque[QueuedTarget] = deque(maxlen=self.max_size)

    def add_candidate(self, detection: Any) -> None:
        priority = float(getattr(detection, "confidence", 0.0))
        self._items.append(QueuedTarget(detection=detection, priority=priority))
        ordered = sorted(self._items, key=lambda item: item.priority, reverse=True)
        self._items = deque(ordered[: self.max_size], maxlen=self.max_size)

    def get_queue(self) -> list[Any]:
        return [item.detection for item in self._items]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
