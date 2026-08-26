"""Configuration loading.

A thin wrapper over ``config/default.yaml`` so that every script reads its
paths, seeds and year blocks from one place. Paths in the config are resolved
relative to the repository root, which is inferred from this file's location.

This module is not in the layout given in the project brief; it was added
because six scripts need the same three lines of YAML loading and path
resolution, and duplicating that is how configs drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "REPO_ROOT",
    "load_config",
    "require_year_block",
    "resolve_path",
]

# src/dalec/config.py -> src/dalec -> src -> repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path
        Path to the YAML file. Defaults to ``config/default.yaml`` at the
        repository root.

    Returns
    -------
    dict
        The parsed configuration.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration file did not parse to a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path against the repository root if it is relative."""
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def require_year_block(config: dict[str, Any], key: str) -> tuple[int, int]:
    """Return an inclusive ``(start_year, end_year)`` block from ``config['years']``.

    Raises rather than defaulting: the calibration and prediction blocks are a
    decision that comes out of the Phase 1 coverage table, and silently guessing
    them would invalidate the calibration/prediction split.
    """
    block = (config.get("years") or {}).get(key)
    if block is None:
        raise ValueError(
            f"years.{key} is not set in {config.get('_config_path', 'the config')}. "
            "Run scripts/00_qc_coverage.py and fill it in as an inclusive [start, end] pair."
        )
    if len(block) != 2:
        raise ValueError(f"years.{key} must be an inclusive [start, end] pair, got {block!r}")
    start, end = int(block[0]), int(block[1])
    if end < start:
        raise ValueError(f"years.{key} has end before start: {block!r}")
    return start, end
