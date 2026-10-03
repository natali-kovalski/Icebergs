"""Load and access `config.yaml`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("config.yaml")

REQUIRED_SECTIONS = ("paths", "search", "hyp3", "preprocess", "cfar", "detections")


@dataclass(frozen=True)
class Config:
    """Parsed configuration. Paths are resolved against the config file's directory."""

    root: Path
    raw: dict[str, Any] = field(repr=False)

    def section(self, name: str) -> dict[str, Any]:
        if name not in self.raw:
            raise KeyError(f"Config section '{name}' not found")
        return self.raw[name]

    def path(self, key: str) -> Path:
        """Absolute path for an entry in the `paths` section."""
        paths = self.section("paths")
        if key not in paths:
            raise KeyError(f"Path '{key}' not found in config 'paths' section")
        return (self.root / paths[key]).resolve()

    @property
    def data_dirs(self) -> list[Path]:
        return [self.path(k) for k in ("raw", "interim", "outputs")]


def metres_to_px(metres: float, pixel_m: float, minimum: int = 1) -> int:
    """Config lengths are in metres so they hold at any resolution; round to whole pixels."""
    return max(minimum, round(metres / pixel_m))


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(f"Config is missing sections: {', '.join(missing)}")
    return Config(root=config_path.parent, raw=raw)
