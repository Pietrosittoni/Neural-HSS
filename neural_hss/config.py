"""Minimal YAML config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, values: dict[str, Any]) -> None:
        self.__dict__.update(values)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, "r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return cls(values)
