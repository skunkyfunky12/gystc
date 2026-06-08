import argparse
import os
import sys


def cmd_index(args):
    from pathlib import Path
    from brain_mcp.config import load_config
    from brain_mcp.storage.database import BrainDB
    from brain_mcp.indexer.vector_store import VectorStore
    from brain_mcp.indexer.embedder import SentenceTransformerBackend
    from brain_mcp.indexer.pipeline import index_vault

    config = load_config()
    if args.vault:
        config.vault_path = Path(args.vault)
    if config.vault_path is None or not config.vault_path.is_dir():
        print(f"ERROR: vault_path not set or not a directory: {config.vault_path}", file=sys.stderr)
        sys.exit(1)

    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = BrainDB(config.db_path)
    embedder = SentenceTransformerBackend(config.model_name)
    vectors = VectorStore(dimension=embedder.dimension) if args.force else VectorStore.load(config.index_path, dimension=embedder.dimension)

    print(f"Scanning vault: {config.vault_path}", file=sys.stderr)
    try:
        count = index_vault(db, vectors, embedder, config.vault_path,
                            config.folder_to_region, force=args.force)
        if count == 0:
            print("No new/changed notes to embed.", file=sys.stderr)
        vectors.save(config.index_path)
        print(f"Done. Index: {config.index_path} | DB: {config.db_path}", file=sys.stderr)
    finally:
        db.close()


def cmd_serve(args):
    from brain_mcp.server import mcp
    mcp.run(transport="stdio")
    # mcp.run() returned -> stdin EOF -> the client disconnected. Force a
    # prompt process exit so a worker thread that is mid-embed can't keep this
    # server (and its brain.db handle) alive as an orphan. Lifespan cleanup
    # has already run inside mcp.run().
    sys.stderr.flush()
    os._exit(0)


def cmd_config(args):
    from pathlib import Path
    from brain_mcp.config import (
        BrainConfig, DEFAULT_MODEL, KNOWN_KEYS, VALID_LOG_LEVELS,
        load_config, save_config, validate_config,
    )
    from brain_mcp.tools.recent import REGION_NAMES

    action = args.config_action

    if action == "show":
        config = load_config()
        print(f"GYSTC Configuration ({config.config_path})")
        print("-" * 60)
        print(f"  vault_path        {config.vault_path or '(not set)'}")
        print(f"  model_name        {config.model_name}")
        print(f"  auto_index        {str(config.auto_index).lower()}")
        print(f"  index_on_startup  {str(config.index_on_startup).lower()}")
        if config.folder_to_region:
            mappings = ", ".join(
                f"{f} -> {REGION_NAMES[i] if 0 <= i < 12 else i}"
                for f, i in config.folder_to_region.items()
            )
            print(f"  folder_to_region  {mappings}")
        else:
            print("  folder_to_region  (empty)")
        print(f"  log_level         {config.log_level}")
        print("-" * 60)
        if config.db_path.exists():
            try:
                from brain_mcp.storage.database import BrainDB
                db = BrainDB(config.db_path)
                notes = db.get_all_notes()
                db.close()
                print(f"  DB: {config.db_path} ({len(notes)} notes)")
            except Exception as e:
                print(f"  DB: {config.db_path} (error reading: {e})")
        else:
            print(f"  DB: {config.db_path} (not created)")
        if config.index_path.exists():
            try:
                from brain_mcp.indexer.vector_store import VectorStore
                vs = VectorStore.load(config.index_path, dimension=384)
                print(f"  Index: {config.index_path} ({vs.size} vectors)")
            except Exception as e:
                print(f"  Index: {config.index_path} (error reading: {e})")
        else:
            print(f"  Index: {config.index_path} (not created)")

    elif action == "set":
        if not args.key:
            print("ERROR: usage: brain_mcp config set <key> <value>", file=sys.stderr)
            sys.exit(1)
        config = load_config()
        key, value = args.key, args.value
        if value is None:
            print(f"ERROR: Missing value. Usage: brain_mcp config set {key} <value>", file=sys.stderr)
            sys.exit(1)

        if key == "vault_path":
            p = Path(value)
            if not p.is_dir():
                print(f"ERROR: Not a directory: {value}", file=sys.stderr)
                sys.exit(1)
            config.vault_path = p
        elif key == "model_name":
            if not value or not value.strip():
                print("ERROR: model_name must not be empty", file=sys.stderr)
                sys.exit(1)
            config.model_name = value.strip()
        elif key in ("auto_index", "index_on_startup"):
            if value.lower() not in ("true", "false"):
                print(f"ERROR: {key} must be true or false", file=sys.stderr)
                sys.exit(1)
            setattr(config, key, value.lower() == "true")
        elif key == "log_level":
            if value.upper() not in VALID_LOG_LEVELS:
                print(f"ERROR: log_level must be one of {sorted(VALID_LOG_LEVELS)}", file=sys.stderr)
                sys.exit(1)
            config.log_level = value.upper()
        elif key == "folder_to_region":
            if "=" not in value:
                print('ERROR: format: folder_to_region "folder name"=INDEX', file=sys.stderr)
                sys.exit(1)
            folder, idx_str = value.rsplit("=", 1)
            folder = folder.strip().strip('"').strip("'")
            try:
                idx = int(idx_str.strip())
                if not (0 <= idx <= 11):
                    raise ValueError
            except ValueError:
                print(f"ERROR: region index must be 0-11, got: {idx_str}", file=sys.stderr)
                sys.exit(1)
            config.folder_to_region[folder] = idx
        else:
            print(f"ERROR: Unknown key: {key}", file=sys.stderr)
            print(f"Valid keys: {', '.join(sorted(KNOWN_KEYS))}", file=sys.stderr)
            sys.exit(1)

        save_config(config)
        print(f"Set {key} = {getattr(config, key) if key != 'folder_to_region' else config.folder_to_region}")

    elif action == "reset":
        config = load_config()
        answer = input("Reset all settings to defaults? This will clear your vault path. [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return
        new_config = BrainConfig(data_dir=config.data_dir)
        save_config(new_config)
        print("Config reset to defaults.")

    elif action == "init":
        config = load_config()
        if config.config_path.exists() and config.vault_path is not None:
            answer = input(f"Config exists at {config.config_path}. Overwrite? [y/N] ")
            if answer.strip().lower() != "y":
                print("Aborted.")
                return

        print("\nGYSTC Setup")
        print("=" * 40)

        # 1. Vault path
        default_vault = str(config.vault_path) if config.vault_path else ""
        prompt = f"\n1. Vault path"
        if default_vault:
            prompt += f" [{default_vault}]"
        prompt += ": "
        vault_input = input(prompt).strip() or default_vault
        if not vault_input:
            print("ERROR: vault_path is required.", file=sys.stderr)
            sys.exit(1)
        vault_path = Path(vault_input)
        if not vault_path.is_dir():
            print(f"ERROR: Not a directory: {vault_path}", file=sys.stderr)
            sys.exit(1)
        md_count = sum(1 for _ in vault_path.rglob("*.md") if ".obsidian" not in _.parts)
        print(f"   Found {md_count} .md files.")
        config.vault_path = vault_path

        # 2. Auto-index
        auto = input("\n2. Enable live-sync (watchdog)? [Y/n] ").strip().lower()
        config.auto_index = auto != "n"
        print(f"   auto_index = {config.auto_index}")

        # 3. Folder-to-region mapping
        top_folders = sorted(
            d.name for d in vault_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if top_folders:
            print(f"\n3. Folder-to-region mapping")
            print(f"   Detected folders: {', '.join(top_folders)}")
            default_mapping = {
                "00 Index": 8, "01 Lucas": 3, "02 Projekte": 0,
                "03 Agenten": 6, "04 Sessions": 3, "05 Referenzen": 2,
            }
            proposed = {f: default_mapping.get(f, 9) for f in top_folders}
            print("   Proposed mapping:")
            for folder, idx in proposed.items():
                region_name = REGION_NAMES[idx] if 0 <= idx < 12 else "Stammhirn"
                print(f"     {folder} -> {region_name} ({idx})")
            accept = input("   Accept? [Y/n] ").strip().lower()
            if accept != "n":
                config.folder_to_region = proposed
                print("   Mapping accepted.")
            else:
                print("   Skipped. You can set mappings later with: config set folder_to_region \"folder\"=INDEX")

        save_config(config)
        print(f"\nConfig saved: {config.config_path}")

        # 4. Build index
        build = input("\n4. Build index now? [Y/n] ").strip().lower()
        if build != "n":
            from brain_mcp.storage.database import BrainDB
            from brain_mcp.indexer.vector_store import VectorStore
            from brain_mcp.indexer.embedder import SentenceTransformerBackend
            from brain_mcp.indexer.pipeline import index_vault
            import time

            db = BrainDB(config.db_path)
            embedder = SentenceTransformerBackend(config.model_name)
            vectors = VectorStore.load(config.index_path, dimension=embedder.dimension)
            t0 = time.time()
            try:
                count = index_vault(db, vectors, embedder, config.vault_path, config.folder_to_region)
                vectors.save(config.index_path)
                elapsed = time.time() - t0
                print(f"   Indexed {count} notes in {elapsed:.1f}s.")
            finally:
                db.close()

        print("\nDone! GYSTC is ready.")
    else:
        print(f"ERROR: Unknown config action: {action}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="brain_mcp", description="GYSTC MCP Server")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start MCP server (default)")
    serve_p.add_argument("--direct", action="store_true",
                         help="Run the in-process stdio server instead of the shared daemon proxy")

    index_p = sub.add_parser("index", help="Build/update vault index")
    index_p.add_argument("--vault", type=str, help="Vault directory path")
    index_p.add_argument("--force", action="store_true", help="Re-embed all notes")

    config_p = sub.add_parser("config", help="Manage configuration")
    config_p.add_argument("config_action", choices=["init", "show", "set", "reset"],
                          help="init | show | set | reset")
    config_p.add_argument("key", nargs="?", help="Config key (for set)")
    config_p.add_argument("value", nargs="?", help="Config value (for set)")

    daemon_p = sub.add_parser("daemon", help="Run the shared background daemon (HTTP)")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "daemon":
        from brain_mcp.daemon.server import run_daemon
        run_daemon()
    else:  # serve
        if getattr(args, "direct", False):
            cmd_serve(args)            # legacy in-process stdio server
        else:
            from brain_mcp.daemon.proxy import run_proxy
            run_proxy()


if __name__ == "__main__":
    main()
