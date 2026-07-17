from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


@dataclass(frozen=True)
class PerformanceSample:
    name: str
    latency_ms: float
    fps: float


class PerformanceMonitor:
    """Passive latency and FPS monitor for offline detector runs."""

    def __init__(self, output_path: str | Path = "logs/performance.json") -> None:
        self.output_path = Path(output_path)
        self.samples: list[PerformanceSample] = []
        self._started_at: float | None = None

    def start_measurement(self) -> None:
        self._started_at = perf_counter()

    def end_measurement(self, name: str = "detect") -> PerformanceSample:
        if self._started_at is None:
            raise RuntimeError("start_measurement() cagrilmadan end_measurement() kullanilamaz")
        elapsed = perf_counter() - self._started_at
        self._started_at = None
        return self.record(name, elapsed)

    def measure(self, name: str, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        result = callback(*args, **kwargs)
        elapsed = perf_counter() - started
        self.record(name, elapsed)
        return result

    def record(self, name: str, latency_seconds: float) -> PerformanceSample:
        latency_ms = latency_seconds * 1000.0
        fps = 1.0 / latency_seconds if latency_seconds > 0 else 0.0
        sample = PerformanceSample(name=name, latency_ms=latency_ms, fps=fps)
        self.samples.append(sample)
        return sample

    @property
    def average_latency_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.latency_ms for sample in self.samples) / len(self.samples)

    @property
    def average_fps(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.fps for sample in self.samples) / len(self.samples)

    def log_performance(self) -> dict[str, object]:
        payload = {
            "sample_count": len(self.samples),
            "average_latency_ms": self.average_latency_ms,
            "average_fps": self.average_fps,
            "samples": [asdict(sample) for sample in self.samples],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
