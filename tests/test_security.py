# tests/test_security.py
import json
from brain_mcp.tools.store import handle_brain_store
from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore


def test_path_traversal_blocked(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="../../etc/passwd", content="bad", region=None,
        tags=[], folder="", watcher=None,
    )
    assert not (tmp_path / "etc" / "passwd.md").exists()
    db.close()


def test_path_traversal_via_folder(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="test", content="bad", region=None,
        tags=[], folder="../../outside", watcher=None,
    )
    assert "error" in result
    db.close()


def test_content_size_limit(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    huge_content = "x" * (1024 * 1024 + 1)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="big", content=huge_content, region=None,
        tags=[], folder="", watcher=None,
    )
    assert "error" in result
    db.close()


def test_too_many_tags(tmp_path, mock_embedder):
    vault = tmp_path / "vault"
    vault.mkdir()
    db = BrainDB(tmp_path / "test.db")
    vectors = VectorStore(dimension=384)
    result = handle_brain_store(
        db, vectors, mock_embedder, vault,
        title="tagged", content="c", region=None,
        tags=["#t" + str(i) for i in range(25)], folder="", watcher=None,
    )
    assert result.get("path") is not None
    row = db.get_note_by_title("tagged")
    assert len(json.loads(row["tags"])) <= 20
    db.close()
