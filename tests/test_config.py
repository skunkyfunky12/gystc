# tests/test_config.py
import json
from pathlib import Path
from brain_mcp.config import load_config

def test_load_config_from_file(tmp_path):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({
        "vault_path": "/my/vault",
        "model_name": "test-model",
    }))
    cfg = load_config(config_dir=config_dir)
    assert cfg.vault_path == Path("/my/vault")
    assert cfg.model_name == "test-model"
    assert cfg.embedding_backend == "sentence-transformers"

def test_env_var_overrides_config(tmp_path, monkeypatch):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({"vault_path": "/from/file"}))
    monkeypatch.setenv("BRAIN_VAULT_PATH", "/from/env")
    cfg = load_config(config_dir=config_dir)
    assert cfg.vault_path == Path("/from/env")

def test_default_config_when_no_file(tmp_path):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    cfg = load_config(config_dir=config_dir)
    assert cfg.vault_path is None
    assert cfg.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
    assert cfg.data_dir == config_dir

def test_data_dir_env_override(tmp_path, monkeypatch):
    alt_dir = tmp_path / "alt"
    alt_dir.mkdir()
    monkeypatch.setenv("BRAIN_DATA_DIR", str(alt_dir))
    cfg = load_config(config_dir=tmp_path / "ignored")
    assert cfg.data_dir == alt_dir

def test_folder_to_region_defaults(tmp_path):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    cfg = load_config(config_dir=config_dir)
    assert cfg.folder_to_region == {}

def test_folder_to_region_from_config(tmp_path):
    config_dir = tmp_path / ".neural-brain"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({
        "folder_to_region": {"Projects": 0, "Notes": 3}
    }))
    cfg = load_config(config_dir=config_dir)
    assert cfg.folder_to_region == {"Projects": 0, "Notes": 3}
