"""Tests for configuration loading and the year-block guard."""

from __future__ import annotations

import pytest
import yaml

from dalec.config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    load_config,
    require_year_block,
    resolve_path,
)


def test_default_config_loads() -> None:
    config = load_config()
    assert config["site"]["code"] == "FI-Hyy"
    assert 0.0 <= config["data"]["qc_threshold"] <= 1.0
    assert config["_config_path"] == str(DEFAULT_CONFIG_PATH)


def test_default_config_leaves_year_blocks_unset() -> None:
    """The split comes out of the Phase 1 coverage table, not out of a default."""
    config = load_config()
    assert config["years"]["calibration"] is None
    assert config["years"]["evaluation"] is None


def test_default_config_has_no_out_of_scope_knobs() -> None:
    """Guard against the locked design decisions drifting back in as options."""
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").lower()
    for forbidden in ("vpd", "edc", "nighttime_only", "daytime_only", "season_filter"):
        assert forbidden not in text, f"config mentions out-of-scope option: {forbidden}"


def test_relative_paths_resolve_against_repo_root() -> None:
    assert resolve_path("data/raw") == REPO_ROOT / "data" / "raw"


def test_absolute_paths_pass_through(tmp_path) -> None:
    assert resolve_path(tmp_path) == tmp_path


def test_require_year_block_raises_when_unset() -> None:
    with pytest.raises(ValueError, match="00_qc_coverage"):
        require_year_block({"years": {"calibration": None}}, "calibration")


def test_require_year_block_returns_inclusive_pair() -> None:
    assert require_year_block({"years": {"calibration": [2000, 2006]}}, "calibration") == (
        2000,
        2006,
    )


def test_require_year_block_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="end before start"):
        require_year_block({"years": {"calibration": [2006, 2000]}}, "calibration")


def test_missing_config_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_config_is_valid_yaml_mapping(tmp_path) -> None:
    path = tmp_path / "scalar.yaml"
    path.write_text("just-a-string\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
    # and the real one parses to a mapping
    assert isinstance(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")), dict)
