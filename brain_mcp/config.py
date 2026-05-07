# brain_mcp/config.py
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATA_DIR = Path.home() / ".neural-brain"

KNOWN_KEYS = {
    "vault_path", "model_name", "embedding_backend", "auto_index",
    "index_on_startup", "folder_to_region", "log_level", "auto_context",
    "reranker",
}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


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
    auto_context: bool = True
    reranker: str | None = "cross-encoder"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "brain.db"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.faiss"

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.json"


def validate_config(config: BrainConfig) -> list[str]:
    errors: list[str] = []
    if config.vault_path is not None and not config.vault_path.is_dir():
        errors.append(f"vault_path is not a directory: {config.vault_path}")
    if not config.model_name:
        errors.append("model_name must not be empty")
    if config.log_level not in VALID_LOG_LEVELS:
        errors.append(f"log_level must be one of {sorted(VALID_LOG_LEVELS)}, got: {config.log_level}")
    for folder, idx in config.folder_to_region.items():
        if not isinstance(idx, int) or not (0 <= idx <= 11):
            errors.append(f"folder_to_region[{folder!r}] must be 0-11, got: {idx}")
    return errors


def save_config(config: BrainConfig, path: Path | None = None) -> Path:
    errors = validate_config(config)
    if errors:
        print(f"WARNING: Config has validation issues: {'; '.join(errors)}", file=sys.stderr)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config_file = path or config.config_path
    data = {
        "vault_path": str(config.vault_path) if config.vault_path else None,
        "model_name": config.model_name,
        "embedding_backend": config.embedding_backend,
        "auto_index": config.auto_index,
        "index_on_startup": config.index_on_startup,
        "folder_to_region": config.folder_to_region,
        "log_level": config.log_level,
        "auto_context": config.auto_context,
        "reranker": config.reranker,
    }
    fd, tmp = tempfile.mkstemp(dir=config.data_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("\n")
        Path(tmp).replace(config_file)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return config_file


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
        auto_context=file_data.get("auto_context", True),
        reranker=file_data.get("reranker", None),
    )
