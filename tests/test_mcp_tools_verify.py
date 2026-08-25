"""Verify all 8 MCP tools function correctly with timeout enforcement."""
import time

import pytest

from brain_mcp.storage.database import BrainDB
from brain_mcp.indexer.vector_store import VectorStore
from brain_mcp.tools.retrieve import handle_brain_retrieve
from brain_mcp.tools.store import handle_brain_store
from brain_mcp.tools.related import handle_brain_related
from brain_mcp.tools.recent import handle_brain_recent
from brain_mcp.tools.regions import handle_brain_regions
from brain_mcp.tools.classify_tool import handle_brain_classify
from brain_mcp.tools.versioning import handle_brain_history, handle_brain_diff, handle_brain_rollback
from tests.conftest import MockEmbedder

TOOL_TIMEOUT = 10  # seconds per tool


@pytest.fixture
def brain_env(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note1.md").write_text(
        "# Routing\nExpress routing handles HTTP requests.\n[[note2]]\n#brain/hippocampus",
        encoding="utf-8",
    )
    (vault / "note2.md").write_text(
        "# Auth\nJWT authentication protects endpoints.\n[[note1]]\n#brain/amygdala",
        encoding="utf-8",
    )

    db = BrainDB(tmp_path / "brain.db")
    embedder = MockEmbedder()
    vectors = VectorStore(dimension=384)

    for name, title, content, region_idx in [
        ("note1.md", "Routing", "Express routing handles HTTP requests.", 3),
        ("note2.md", "Auth", "JWT authentication protects endpoints.", 11),
    ]:
        nid = db.upsert_note(
            path=name, title=title, content=content, content_hash=f"h_{name}",
            region_idx=region_idx, tags=["#test"], word_count=len(content.split()),
            created_at="2026-01-01", modified_at="2026-01-01",
        )
        vec = embedder.embed([content])
        fids = vectors.add(vec)
        db.set_faiss_idx(nid, fids[0])

    db.upsert_edge(1, 2, edge_type="backlink")
    db.upsert_edge(2, 1, edge_type="backlink")

    yield {
        "db": db, "vectors": vectors, "embedder": embedder,
        "vault": vault, "tmp_path": tmp_path,
    }
    db.close()


# --- Tool 1: brain_status ---

def test_brain_status(brain_env):
    db = brain_env["db"]
    t0 = time.time()
    count = db.get_note_count()
    region_counts = db.get_region_note_counts()
    edge_types = db.get_edge_type_counts()
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert count == 2
    assert sum(region_counts.values()) == 2
    assert sum(edge_types.values()) == 2


# --- Tool 2: brain_retrieve ---

def test_brain_retrieve_semantic(brain_env):
    t0 = time.time()
    results = handle_brain_retrieve(
        brain_env["db"], brain_env["vectors"], brain_env["embedder"],
        query="routing HTTP", limit=10, threshold=-1.0,
    )
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert isinstance(results, list)
    assert len(results) >= 1
    titles = [r["title"] for r in results]
    assert "Routing" in titles


def test_brain_retrieve_fts_only(brain_env):
    t0 = time.time()
    results = handle_brain_retrieve(
        brain_env["db"], brain_env["vectors"], brain_env["embedder"],
        query="authentication", limit=10, threshold=-1.0, fts_only=True,
    )
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert isinstance(results, list)
    assert len(results) >= 1


def test_brain_retrieve_graph_mode(brain_env):
    t0 = time.time()
    results = handle_brain_retrieve(
        brain_env["db"], brain_env["vectors"], brain_env["embedder"],
        file_paths=["note1.md"], limit=10, threshold=-1.0,
    )
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert isinstance(results, list)
    assert len(results) >= 1
    assert any(r["title"] == "Auth" for r in results)


def test_brain_retrieve_empty_query(brain_env):
    results = handle_brain_retrieve(
        brain_env["db"], brain_env["vectors"], brain_env["embedder"],
        query=None, file_paths=None,
    )
    assert isinstance(results, list)
    assert results[0].get("error")


# --- Tool 3: brain_store ---

def test_brain_store(brain_env):
    t0 = time.time()
    result = handle_brain_store(
        brain_env["db"], brain_env["vectors"], brain_env["embedder"],
        vault_root=brain_env["vault"],
        title="New Note", content="This is a test note about pipelines.",
        region="Basalganglien", tags=["#test"], folder="",
        watcher=None,
    )
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert result.get("path") is not None
    assert "error" not in result
    assert (brain_env["vault"] / "New Note.md").exists()


def test_brain_store_update(brain_env):
    handle_brain_store(
        brain_env["db"], brain_env["vectors"], brain_env["embedder"],
        vault_root=brain_env["vault"],
        title="Update Me", content="Version 1",
        tags=[], folder="", watcher=None,
    )
    result = handle_brain_store(
        brain_env["db"], brain_env["vectors"], brain_env["embedder"],
        vault_root=brain_env["vault"],
        title="Update Me", content="Version 2",
        tags=[], folder="", watcher=None,
    )
    assert result.get("path") is not None
    assert "error" not in result
    content = (brain_env["vault"] / "Update Me.md").read_text(encoding="utf-8")
    assert "Version 2" in content


# --- Tool 4: brain_related ---

def test_brain_related(brain_env):
    t0 = time.time()
    results = handle_brain_related(
        brain_env["db"], brain_env["vectors"], brain_env["embedder"],
        title="Routing", limit=10,
    )
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert isinstance(results, list)
    assert len(results) >= 1


def test_brain_related_not_found(brain_env):
    result = handle_brain_related(
        brain_env["db"], brain_env["vectors"], brain_env["embedder"],
        title="Nonexistent Note XYZ",
    )
    assert isinstance(result, dict) and "error" in result


# --- Tool 5: brain_recent ---

def test_brain_recent(brain_env):
    t0 = time.time()
    results = handle_brain_recent(brain_env["db"], days=30, limit=10)
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert isinstance(results, list)


def test_brain_recent_with_region(brain_env):
    results = handle_brain_recent(brain_env["db"], days=30, region="Hippocampus", limit=10)
    assert isinstance(results, list)
    for r in results:
        assert r.get("region") == "Hippocampus"


# --- Tool 6: brain_regions ---

def test_brain_regions_list(brain_env):
    t0 = time.time()
    results = handle_brain_regions(brain_env["db"], action="list")
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert isinstance(results, list)
    assert len(results) == 12


def test_brain_regions_describe(brain_env):
    result = handle_brain_regions(brain_env["db"], action="describe", region="Hippocampus")
    assert isinstance(result, dict)
    assert "name" in result or "description" in result or "note_count" in result


# --- Tool 7: brain_classify ---

def test_brain_classify_single(brain_env):
    t0 = time.time()
    result = handle_brain_classify(
        brain_env["db"], brain_env["vault"],
        title="Express Routing", content="Express routing handles HTTP requests",
        batch=False, apply=False,
    )
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert isinstance(result, dict)
    assert "region_idx" in result


def test_brain_classify_reclassify_dryrun(brain_env):
    result = handle_brain_classify(
        brain_env["db"], brain_env["vault"],
        batch=True, apply=False,
    )
    assert isinstance(result, dict)
    assert "batch" in result


# --- Tool 8: brain_versions ---

def test_brain_versions_history(brain_env):
    brain_env["db"].upsert_note(
        path="note1.md", title="Routing", content="Updated routing content",
        content_hash="h_updated", region_idx=3, tags=["#test"],
        word_count=3, created_at="2026-01-01", modified_at="2026-01-02",
    )
    t0 = time.time()
    result = handle_brain_history(brain_env["db"], path="note1.md")
    elapsed = time.time() - t0

    assert elapsed < TOOL_TIMEOUT
    assert isinstance(result, list)
    assert len(result) >= 1


def test_brain_versions_diff(brain_env):
    brain_env["db"].upsert_note(
        path="note1.md", title="Routing", content="Changed content",
        content_hash="h_changed", region_idx=3, tags=["#test"],
        word_count=2, created_at="2026-01-01", modified_at="2026-01-02",
    )
    note = brain_env["db"].get_note_by_path("note1.md")
    versions = brain_env["db"].get_versions(note["id"])
    assert len(versions) >= 1, "Expected at least one version after content change"
    result = handle_brain_diff(brain_env["db"], path="note1.md", version_id=versions[0]["id"])
    assert isinstance(result, dict)
    assert "diff" in result


def test_brain_versions_rollback(brain_env):
    brain_env["db"].upsert_note(
        path="note1.md", title="Routing", content="v2 content",
        content_hash="h_v2", region_idx=3, tags=["#test"],
        word_count=2, created_at="2026-01-01", modified_at="2026-01-02",
    )
    note = brain_env["db"].get_note_by_path("note1.md")
    versions = brain_env["db"].get_versions(note["id"])
    assert len(versions) >= 1, "Expected at least one version after content change"
    result = handle_brain_rollback(
        brain_env["db"], brain_env["vault"],
        path="note1.md", version_id=versions[0]["id"],
        watcher=None,
    )
    assert result.get("rolled_back") is True


def test_brain_versions_not_found(brain_env):
    result = handle_brain_history(brain_env["db"], path="nonexistent.md")
    assert isinstance(result, dict) and "error" in result
