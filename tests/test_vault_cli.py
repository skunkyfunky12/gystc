"""CLI wiring for `brain_mcp curate`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain_mcp.curation.cli import main


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """cmd_analyze's load_config() reads (and mkdirs!) the real ~/.gystc
    otherwise -- these tests only passed because the developer's real
    exclude_dirs happened not to match the tmp vault."""
    monkeypatch.setenv("BRAIN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("BRAIN_VAULT_PATH", raising=False)


def _vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    (v / "a.md").write_text("# A\n[[ghost]]\n", encoding="utf-8")
    (v / ".gitignore").write_text(".obsidian/\n", encoding="utf-8")
    return v


def test_init_then_analyze(tmp_path, capsys):
    v = _vault(tmp_path)
    main(["init", "--vault", str(v)])
    assert (v / ".git").is_dir()
    out = tmp_path / "proposal.json"
    main(["analyze", "--vault", str(v), "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["stats"]["notes_scanned"] == 1
    assert any(c["kind"] == "dead_link" for c in data["candidates"])


def test_apply_dry_run_then_real(tmp_path):
    v = _vault(tmp_path)
    main(["init", "--vault", str(v)])
    (v / "junk.md").write_text("# Junk\n", encoding="utf-8")
    # commit junk so it's part of the tracked tree (mirrors real use)
    from brain_mcp.curation.vault_git import commit_run
    commit_run(v, "add junk")
    prop = tmp_path / "actions.json"
    prop.write_text(json.dumps({"actions": [{"op": "archive", "file": "junk.md"}]}), encoding="utf-8")

    main(["apply", "--vault", str(v), "--proposals", str(prop)])          # dry run
    assert (v / "junk.md").exists(), "dry run must not change anything"

    main(["apply", "--vault", str(v), "--proposals", str(prop), "--yes"])  # real
    assert not (v / "junk.md").exists()
    assert (v / "99 Archiv" / "junk.md").exists()
