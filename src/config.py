"""Lightweight config loader.

Loads a YAML config into nested dataclasses with attribute access, so the rest
of the codebase can do `cfg.model.base_model` instead of dict indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class _AttrDict(dict):
    """Dict that also supports attribute access (cfg.model.lr)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _AttrDict(value) if isinstance(value, dict) else value


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        raw = object.__getattribute__(self, "raw")
        if name in raw:
            value = raw[name]
            return _AttrDict(value) if isinstance(value, dict) else value
        raise AttributeError(name)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path, e.g. cfg.get('model.lora.r')."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(raw=data)
