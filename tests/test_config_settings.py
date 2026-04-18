"""Tests for config save/load/validate and settings system."""
from pathlib import Path
import json
import pytest

from brain_mcp.config import (
    BrainConfig, DEFAULT_MODEL, VALID_LOG_LEVELS,
    load_config, save_config, validate_config,
)


@pytest.fixture
def tmp_config_dir(tmp_path):
    return tmp_path / "brain-data"


def test_save_load_roundtrip(tmp_config_dir):
    config = BrainConfig(
        vault_path=Path("/fake/vault"),
        model_name="test-model",
        auto_index=False,
        index_on_startup=False,
        folder_to_region={"docs": 3, "code": 0},
        log_level="DEBUG",
        data_dir=tmp_config_dir,
    )
    save_config(config)
    loaded = load_config(tmp_config_dir)
    assert loaded.vault_path == Path("/fake/vault")
    assert loaded.model_name == "test-model"
    assert loaded.auto_index is False
    assert loaded.index_on_startup is False
    assert loaded.folder_to_region == {"docs": 3, "code": 0}
    assert loaded.log_level == "DEBUG"


def test_save_strips_legacy_keys(tmp_config_dir):
    tmp_config_dir.mkdir(parents=True, exist_ok=True)
    legacy = {
        "vault_path": "/fake",
        "graph_path": "should/be/stripped",
        "obsidian_api_key": "secret",
        "model_name": DEFAULT_MODEL,
    }
    (tmp_config_dir / "config.json").write_text(json.dumps(legacy))
    config = load_config(tmp_config_dir)
    save_config(config)
    data = json.loads((tmp_config_dir / "config.json").read_text())
    assert "graph_path" not in data
    assert "obsidian_api_key" not in data
    assert "vault_path" in data


def test_validate_bad_vault_path():
    config = BrainConfig(vault_path=Path("/nonexistent/path"))
    errors = validate_config(config)
    assert any("vault_path" in e for e in errors)


def test_validate_bad_log_level():
    config = BrainConfig(log_level="VERBOSE")
    errors = validate_config(config)
    assert any("log_level" in e for e in errors)


def test_validate_bad_region_index():
    config = BrainConfig(folder_to_region={"bad": 99})
    errors = validate_config(config)
    assert any("folder_to_region" in e for e in errors)


def test_validate_good_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    config = BrainConfig(vault_path=vault, log_level="INFO")
    errors = validate_config(config)
    assert errors == []


def test_save_atomic_no_corrupt(tmp_config_dir):
    """Verify config file is valid JSON after save."""
    config = BrainConfig(data_dir=tmp_config_dir)
    save_config(config)
    data = json.loads((tmp_config_dir / "config.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "auto_index" in data
