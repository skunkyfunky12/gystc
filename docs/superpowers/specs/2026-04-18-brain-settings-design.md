# Neural Brain Settings System

## Goal

Add a CLI-based configuration system with an interactive setup wizard, a `brain_reindex` MCP tool, and remove the Obsidian MCP dependency. Neural Brain becomes the sole brain interface for Claude.

## Architecture

The settings system adds three things to the existing codebase:

1. **`config` CLI subcommand** — `init`, `show`, `set`, `reset` actions
2. **`save_config()` in config.py** — write-back with validation
3. **`brain_reindex` MCP tool** — on-demand re-indexing from within Claude

No new dependencies. No new files except tests. Changes touch `config.py`, `__main__.py`, `server.py`, and `.mcp.json`.

## Settings Schema

All settings live in `~/.neural-brain/config.json`. Environment variables override file values (existing behavior, unchanged).

| Key | Type | Default | ENV Override | Description |
|-----|------|---------|-------------|-------------|
| `vault_path` | `string \| null` | `null` | `BRAIN_VAULT_PATH` | Absolute path to Obsidian vault root |
| `model_name` | `string` | `paraphrase-multilingual-MiniLM-L12-v2` | `BRAIN_MODEL_NAME` | sentence-transformers model ID |
| `auto_index` | `bool` | `true` | — | Enable watchdog file watcher |
| `index_on_startup` | `bool` | `true` | — | Scan vault on server start |
| `folder_to_region` | `dict[str, int]` | `{}` | — | Top-level folder name to region index (0-11) |
| `log_level` | `string` | `INFO` | `BRAIN_LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR |

Legacy keys (`graph_path`, `obsidian_api_key`) are silently ignored on read and stripped on next write.

## CLI Commands

### `python -m brain_mcp config init`

Interactive wizard for first-time setup. Steps:

1. **Vault path** — prompt for path, validate it exists and contains `.md` files. Show count: "Found N .md files."
2. **Auto-index** — "Enable live-sync when files change? [Y/n]"
3. **Folder-to-region mapping** — detect top-level folders, propose default mapping based on known folder names, ask to confirm or skip.
4. **Build index** — "Build index now? [Y/n]" — runs `index_vault()` with progress output.

If `config.json` already exists, warn: "Config exists. Overwrite? [y/N]"

The wizard uses `input()` for prompts. No third-party TUI library.

### `python -m brain_mcp config show`

Print current config as formatted table:

```
Neural Brain Configuration (~/.neural-brain/config.json)
─────────────────────────────────────────────────────────
vault_path        C:\Users\lucas\...\Claude Brain
model_name        paraphrase-multilingual-MiniLM-L12-v2
auto_index        true
index_on_startup  true
folder_to_region  00 Index → Thalamus (8), 01 Lucas → Hippocampus (3), ...
log_level         INFO
─────────────────────────────────────────────────────────
DB: ~/.neural-brain/brain.db (exists, 1101 notes)
Index: ~/.neural-brain/index.faiss (exists, 1101 vectors)
```

Shows derived info (DB note count, index vector count) below the settings.

### `python -m brain_mcp config set <key> <value>`

Set a single config value. Validates:
- `vault_path`: must be an existing directory
- `model_name`: must be a non-empty string
- `auto_index`, `index_on_startup`: must be `true` or `false`
- `log_level`: must be one of `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `folder_to_region`: accepts `folder=index` format, e.g. `"02 Projekte"=0`

Prints confirmation: "Set vault_path = C:\...\Claude Brain"

### `python -m brain_mcp config reset`

Reset to defaults. Prompts: "Reset all settings to defaults? This will clear your vault path. [y/N]"

## brain_reindex MCP Tool

```python
@mcp.tool()
def brain_reindex(force: bool = False) -> dict:
    """Re-index the vault. Use after bulk edits, graphify exports, or when results seem stale.

    Args:
        force: Re-embed all notes even if unchanged (default: false)
    """
```

Implementation: calls `index_vault()` from `pipeline.py` (already exists). Returns:

```json
{
    "indexed": 42,
    "total": 1101,
    "elapsed_seconds": 2.3
}
```

If vault_path is not configured or not a directory, returns `{"error": "..."}`.

## Remove Obsidian MCP

Remove the `obsidian` entry from `~/.claude/.mcp.json`. Neural Brain replaces it entirely:

- Note reading: `brain_retrieve` (semantic) replaces `obsidian_get_file` (path-based)
- Note writing: `brain_store` replaces `obsidian_put_file`
- Searching: `brain_retrieve` (semantic + FTS5) replaces `obsidian_simple_search` (keyword)
- No Obsidian app needs to be running

## Changes to Existing Files

### `config.py`

Add:
- `save_config(config: BrainConfig, path: Path | None = None)` — serialize to JSON, strip legacy keys, write atomically (write to `.tmp`, rename)
- `validate_config(config: BrainConfig) -> list[str]` — return list of validation errors (empty = valid)
- `KNOWN_KEYS` set for identifying legacy keys to strip
- `REGION_NAMES` list for display in wizard (already exists in `tools/recent.py`, import from there)

### `__main__.py`

Add `config` subcommand with sub-actions:
- `config init` — wizard function
- `config show` — display function
- `config set <key> <value>` — setter with validation
- `config reset` — reset with confirmation

The wizard function (`cmd_config_init`) handles all interactive I/O via `input()`.

### `server.py`

Add `brain_reindex` tool that calls `index_vault()` and returns stats.

### `~/.claude/.mcp.json`

Remove `obsidian` server entry. Keep only `neural-brain`.

## Testing

- `test_config_save_load_roundtrip` — save, load, verify values match
- `test_config_strips_legacy_keys` — save with `graph_path`, reload, verify absent
- `test_config_validate_vault_path` — invalid path returns error
- `test_config_validate_log_level` — invalid level returns error
- `test_config_set_folder_to_region` — parse `"folder=idx"` format
- `test_brain_reindex_tool` — mock embedder, verify count returned

Wizard is not unit-tested (interactive I/O). Manual test during implementation.

## Out of Scope

- Web UI for settings
- Obsidian REST API integration
- graphify direct graph.json import (already integrated via vault notes)
- Migration tool from old config format (legacy keys are silently stripped)
