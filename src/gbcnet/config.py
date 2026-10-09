"""YAML config loading with `_base_` inheritance and command-line overrides."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, resolving a `_base_` key relative to the file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("_base_", None)
    if base_name:
        base = load_yaml(path.parent / base_name)
        cfg = _deep_merge(base, cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply `section.key=value` overrides; values are parsed as YAML scalars/lists."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set {dotted!r}: {k!r} is not a mapping")
        node[keys[-1]] = value
    return cfg


def add_config_args(parser: argparse.ArgumentParser, default_config: Optional[str] = None) -> None:
    parser.add_argument("--config", default=default_config, required=default_config is None, help="Path to YAML config")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set train.epochs=2 cv.n_splits=2",
    )


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return apply_overrides(load_yaml(args.config), args.set)


def save_yaml(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
