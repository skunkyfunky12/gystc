"""Regression tests for ox-alpha review 2026-08-23, finding 4 (severity raised).

`VectorStore.load()` DELETES an index file whose dimension differs from the one
passed in -- correct for the writer paths, which rebuild afterwards, fatal for a
display-only caller that has to guess. Two such callers passed a hardcoded 384,
so with any non-384 embedding model configured (`BRAIN_MODEL_NAME`, e.g.
all-mpnet-base-v2 at 768) they destroyed the index just by being run:

- `brain/web_widget.py` /api/stats  -> the dashboard fetches it on page load
  (covered in tests/test_fix_dashboard_web.py)
- `brain_mcp config show`           -> covered here; this is the command a user
  runs precisely *because* search stopped working

Read-only callers must use `read_index_stats()`, which never writes and never
guesses a dimension.
"""
import argparse

import numpy as np
import pytest

from brain_mcp.indexer.vector_store import VectorStore, read_index_stats


@pytest.fixture
def index_768(tmp_path, monkeypatch):
    """Data dir holding an index built with a non-default (768) dimension."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("BRAIN_DATA_DIR", str(data_dir))
    monkeypatch.delenv("BRAIN_VAULT_PATH", raising=False)
    path = data_dir / "index.faiss"
    store = VectorStore(dimension=768)
    store.add(np.ones((4, 768), dtype=np.float32))
    store.save(path)
    return path


def test_config_show_does_not_delete_non_384_index(index_768, capsys):
    from brain_mcp.__main__ import cmd_config

    cmd_config(argparse.Namespace(config_action="show", key=None, value=None))

    assert index_768.exists(), "`config show` must not delete the index it reports on"


def test_config_show_reports_vector_count_of_non_384_index(index_768, capsys):
    from brain_mcp.__main__ import cmd_config

    cmd_config(argparse.Namespace(config_action="show", key=None, value=None))

    assert "4 vectors" in capsys.readouterr().out


def test_config_show_reports_index_dimension(index_768, capsys):
    """The dimension is what a broken search hinges on: an index built by a
    different model than the configured one is exactly the case this command is
    run to diagnose, so it must be printed rather than left to be guessed."""
    from brain_mcp.__main__ import cmd_config

    cmd_config(argparse.Namespace(config_action="show", key=None, value=None))

    assert "768" in capsys.readouterr().out


# ------------------------------------------------------- read_index_stats itself

def test_read_index_stats_returns_count_and_dimension(index_768):
    assert read_index_stats(index_768) == (4, 768)


def test_read_index_stats_returns_none_for_missing_file(tmp_path):
    assert read_index_stats(tmp_path / "nope.faiss") is None


def test_read_index_stats_leaves_damaged_file_in_place(tmp_path):
    damaged = tmp_path / "index.faiss"
    damaged.write_bytes(b"not a faiss index")

    assert read_index_stats(damaged) is None
    assert damaged.exists(), "a read-only helper must never delete the file it fails on"
