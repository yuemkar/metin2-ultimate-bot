from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    root_dir: Path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        value: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    return AppConfig(raw=raw, root_dir=config_path.parent)
