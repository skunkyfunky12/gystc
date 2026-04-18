import argparse
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


def main():
    parser = argparse.ArgumentParser(prog="brain_mcp", description="Neural Brain MCP Server")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="Start MCP server (default)")
    index_p = sub.add_parser("index", help="Build/update vault index")
    index_p.add_argument("--vault", type=str, help="Vault directory path")
    index_p.add_argument("--force", action="store_true", help="Re-embed all notes")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    else:
        cmd_serve(args)


if __name__ == "__main__":
    main()
