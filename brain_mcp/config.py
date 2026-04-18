# brain_mcp/config.py
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATA_DIR = Path.home() / ".neural-brain"


@dataclass
class BrainConfig:
    vault_path: Path | None = None
    model_name: str = DEFAULT_MODEL
    embedding_backend: str = "sentence-transformers"
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    auto_index: bool = True
    index_on_startup: bool = True
    folder_to_region: dict[str, int] = field(default_factory=dict)
    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "brain.db"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.faiss"

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.json"


def load_config(config_dir: Path | None = None) -> BrainConfig:
    data_dir_env = os.environ.get("BRAIN_DATA_DIR")
    if data_dir_env:
        data_dir = Path(data_dir_env)
    elif config_dir:
        data_dir = config_dir
    else:
        data_dir = DEFAULT_DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    config_file = data_dir / "config.json"

    file_data: dict = {}
    if config_file.exists():
        try:
            file_data = json.loads(config_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: Failed to read config {config_file}: {e}", file=sys.stderr)
            print("Falling back to defaults and environment variables.", file=sys.stderr)

    vault_path_str = os.environ.get("BRAIN_VAULT_PATH") or file_data.get("vault_path")
    vault_path = Path(vault_path_str) if vault_path_str else None

    return BrainConfig(
        vault_path=vault_path,
        model_name=os.environ.get("BRAIN_MODEL_NAME") or file_data.get("model_name", DEFAULT_MODEL),
        embedding_backend=file_data.get("embedding_backend", "sentence-transformers"),
        data_dir=data_dir,
        auto_index=file_data.get("auto_index", True),
        index_on_startup=file_data.get("index_on_startup", True),
        folder_to_region=file_data.get("folder_to_region", {}),
        log_level=os.environ.get("BRAIN_LOG_LEVEL") or file_data.get("log_level", "INFO"),
    )
