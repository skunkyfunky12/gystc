import argparse
import sys
import time


def cmd_index(args):
    from pathlib import Path
    from brain_mcp.config import load_config
    from brain_mcp.storage.database import BrainDB
    from brain_mcp.indexer.vector_store import VectorStore
    from brain_mcp.indexer.embedder import SentenceTransformerBackend
    from brain_mcp.indexer.scanner import scan_vault

    config = load_config()
    if args.vault:
        config.vault_path = Path(args.vault)
    if config.vault_path is None or not config.vault_path.is_dir():
        print(f"ERROR: vault_path not set or not a directory: {config.vault_path}", file=sys.stderr)
        sys.exit(1)

    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = BrainDB(config.db_path)
    vectors = VectorStore(dimension=384) if args.force else VectorStore.load(config.index_path, dimension=384)
    embedder = SentenceTransformerBackend(config.model_name)

    print(f"Scanning vault: {config.vault_path}", file=sys.stderr)
    notes = scan_vault(config.vault_path, config.folder_to_region)
    print(f"Found {len(notes)} notes.", file=sys.stderr)

    title_to_id: dict[str, int] = {}
    to_embed: list[tuple[int, str]] = []
    for note in notes:
        if not args.force:
            existing_hash = db.get_content_hash(note["path"])
            if existing_hash == note["content_hash"]:
                row = db.get_note_by_path(note["path"])
                if row:
                    title_to_id[note["title"]] = row["id"]
                continue
        note_id = db.upsert_note(
            path=note["path"], title=note["title"], content=note["content"],
            content_hash=note["content_hash"], region_idx=note["region_idx"],
            tags=note["tags"], word_count=note["word_count"],
            created_at=note["created_at"], modified_at=note["modified_at"],
        )
        title_to_id[note["title"]] = note_id
        to_embed.append((note_id, note["content"]))

    if to_embed:
        print(f"Embedding {len(to_embed)} notes...", file=sys.stderr)
        t0 = time.time()
        texts = [t for _, t in to_embed]
        vecs = embedder.embed(texts)
        faiss_ids = vectors.add(vecs)
        for (note_id, _), fid in zip(to_embed, faiss_ids):
            db.set_faiss_idx(note_id, fid)
        elapsed = time.time() - t0
        print(f"Embedded in {elapsed:.1f}s", file=sys.stderr)
    else:
        print("No new/changed notes to embed.", file=sys.stderr)

    for note in notes:
        src_id = title_to_id.get(note["title"])
        if src_id is None:
            continue
        for bl_title in note.get("backlink_titles", []):
            tgt_id = title_to_id.get(bl_title)
            if tgt_id and src_id != tgt_id:
                db.upsert_edge(src_id, tgt_id, link_text=bl_title)

    vectors.save(config.index_path)
    db.close()
    print(f"Done. Index: {config.index_path} | DB: {config.db_path}", file=sys.stderr)


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
