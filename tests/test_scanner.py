from brain_mcp.indexer.scanner import scan_vault, compute_content_hash


def test_scan_vault_finds_md_files(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    assert len(notes) == 3
    titles = {n["title"] for n in notes}
    assert titles == {"note1", "note2", "project1"}


def test_scan_vault_extracts_brain_tags(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert by_title["note1"]["region_idx"] == 3  # hippocampus
    assert by_title["note2"]["region_idx"] == 0  # praefrontaler-cortex


def test_scan_vault_folder_mapping(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={"Projects": 10})
    by_title = {n["title"]: n for n in notes}
    assert by_title["project1"]["region_idx"] == 10


def test_scan_vault_default_region(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert by_title["project1"]["region_idx"] == 9


def test_scan_vault_backlinks(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert "note2" in by_title["note1"]["backlink_titles"]
    assert "note1" in by_title["note2"]["backlink_titles"]


def test_scan_vault_content_hash(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    assert all("content_hash" in n for n in notes)
    assert all(len(n["content_hash"]) == 64 for n in notes)


def test_scan_vault_content_included(tmp_vault):
    notes = scan_vault(tmp_vault, folder_to_region={})
    by_title = {n["title"]: n for n in notes}
    assert "routing" in by_title["note1"]["content"].lower()


def test_compute_content_hash_deterministic():
    h1 = compute_content_hash("hello world")
    h2 = compute_content_hash("hello world")
    h3 = compute_content_hash("different")
    assert h1 == h2
    assert h1 != h3


def test_scan_vault_skips_obsidian_dir(tmp_vault):
    obs = tmp_vault / ".obsidian"
    obs.mkdir()
    (obs / "config.md").write_text("# Config", encoding="utf-8")
    notes = scan_vault(tmp_vault, folder_to_region={})
    titles = {n["title"] for n in notes}
    assert "config" not in titles
