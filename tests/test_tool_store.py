# tests/test_tool_store.py
from pathlib import Path
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.store import handle_brain_store, sanitize_title


def test_sanitize_title_strips_traversal():
    assert sanitize_title("../../etc/passwd") == "etcpasswd"
    assert sanitize_title("normal title") == "normal title"
    assert sanitize_title("a" * 300) == "a" * 200


def test_sanitize_title_strips_special_chars():
    assert sanitize_title('file:name*bad?"yes"') == "filenamebadyes"


def test_store_creates_file(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="My Note", content="Hello world content", region="Hippocampus",
        tags=["#test"], folder="",
        watcher=None,
    )
    assert result["path"] == "My Note.md"
    assert (vault / "My Note.md").exists()
    assert "Hello world content" in (vault / "My Note.md").read_text(encoding="utf-8")
    db.close()


def test_store_creates_subfolder(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="Sub Note", content="In a folder", region=None,
        tags=[], folder="Projects",
        watcher=None,
    )
    assert result["path"] == "Projects/Sub Note.md"
    assert (vault / "Projects" / "Sub Note.md").exists()
    db.close()


def test_store_indexes_note(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="Indexed", content="Some searchable content", region=None,
        tags=[], folder="", watcher=None,
    )
    assert vectors.size == 1
    row = db.get_note_by_title("Indexed")
    assert row is not None
    assert row["faiss_idx"] == 0
    db.close()


def test_store_adds_brain_tag(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="Tagged", content="Content here", region="Hippocampus",
        tags=[], folder="", watcher=None,
    )
    text = (vault / "Tagged.md").read_text(encoding="utf-8")
    assert "#brain/hippocampus" in text
    db.close()


def test_store_sets_pending_write(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    # Use a real BrainWatcher to test pending_writes integration
    from brain_mcp.indexer.watcher import BrainWatcher
    watcher = BrainWatcher(vault, lambda p, e: None)
    handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="Pending", content="Content", region=None,
        tags=[], folder="", watcher=watcher,
    )
    assert len(watcher._pending_writes) == 1
    db.close()


def test_store_invalid_region_returns_error(tmp_path, mock_embedder):
    """REVIEW FIX: Invalid region should error, not silently default."""
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="Bad Region", content="Content", region="Nonexistent",
        tags=[], folder="", watcher=None,
    )
    assert "error" in result
    db.close()


def test_store_strips_existing_brain_tag(tmp_path, mock_embedder):
    """REVIEW FIX: Changing region should strip old #brain/ tag."""
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="ReTag", content="Old content\n\n#brain/stammhirn\n",
        region="Hippocampus",
        tags=[], folder="", watcher=None,
    )
    text = (vault / "ReTag.md").read_text(encoding="utf-8")
    assert "#brain/hippocampus" in text
    assert "#brain/stammhirn" not in text
    db.close()


def test_store_sanitized_title_warning(tmp_path, mock_embedder):
    """REVIEW FIX: Warn when title was sanitized."""
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="bad:title*here", content="Content",
        region=None, tags=[], folder="", watcher=None,
    )
    assert result.get("title_sanitized") is True
    db.close()
