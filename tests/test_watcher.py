import time
from brain_mcp.indexer.watcher import BrainWatcher


def test_watcher_starts_and_stops(tmp_path):
    watcher = BrainWatcher(tmp_path, on_change=lambda p, ev: None)
    watcher.start()
    assert watcher.is_running
    watcher.stop()
    assert not watcher.is_running


def test_watcher_detects_file_create(tmp_path):
    events = []
    watcher = BrainWatcher(tmp_path, on_change=lambda p, ev: events.append((p, ev)))
    watcher.start()
    time.sleep(0.3)
    (tmp_path / "new.md").write_text("# New", encoding="utf-8")
    time.sleep(1.0)
    watcher.stop()
    md_events = [(p, e) for p, e in events if p.endswith(".md")]
    assert len(md_events) >= 1


def test_watcher_ignores_non_md(tmp_path):
    events = []
    watcher = BrainWatcher(tmp_path, on_change=lambda p, ev: events.append((p, ev)))
    watcher.start()
    time.sleep(0.3)
    (tmp_path / "ignore.txt").write_text("not markdown", encoding="utf-8")
    time.sleep(1.0)
    watcher.stop()
    md_events = [(p, e) for p, e in events if p.endswith(".md")]
    assert len(md_events) == 0


def test_watcher_skips_pending_writes(tmp_path):
    events = []
    watcher = BrainWatcher(tmp_path, on_change=lambda p, ev: events.append((p, ev)))
    target = tmp_path / "self.md"
    resolved = str(target.resolve())
    watcher.add_pending_write(resolved)
    watcher.start()
    time.sleep(0.3)
    target.write_text("# Self-written", encoding="utf-8")
    time.sleep(1.0)
    watcher.stop()
    md_events = [(p, e) for p, e in events if "self.md" in p]
    assert len(md_events) == 0


def test_watcher_callback_error_does_not_crash(tmp_path):
    """Verify watcher keeps running when callback raises."""
    call_count = []

    def exploding_callback(path: str, event_type: str) -> None:
        call_count.append(1)
        raise RuntimeError("boom")

    watcher = BrainWatcher(tmp_path, on_change=exploding_callback)
    watcher.start()
    time.sleep(0.3)
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    time.sleep(1.0)
    assert watcher.is_running
    watcher.stop()
