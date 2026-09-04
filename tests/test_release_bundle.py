"""Guards for the packaged release.

Regression suite for the 2026-09-03 finding: the release workflow installed a
hand-written list of GUI packages instead of the project, so the shipped binary
contained no faiss, mcp, sentence-transformers, watchdog, psutil or httpx. Every
search in the released dashboard answered HTTP 500 -- and CI stayed green,
because the only verification was ``ls dist/``.

These tests fail at pull-request time, long before a tag exists.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from brain_mcp.indexer import bundled_model

ROOT = Path(__file__).resolve().parents[1]
BUILD_WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"
SPEC = ROOT / "gystc.spec"
PYPROJECT = ROOT / "pyproject.toml"

# Distribution name on PyPI -> module name you import. Only where they differ.
_DIST_TO_MODULE = {
    "faiss-cpu": "faiss",
    "sentence-transformers": "sentence_transformers",
    "PyQt6-WebEngine": "PyQt6.QtWebEngineWidgets",
}


MODEL = "some/model"


def _write_model_dir(path: Path, complete: bool = True,
                     model_name: str | None = MODEL) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    if complete:
        (path / "modules.json").write_text("[]", encoding="utf-8")
    if model_name is not None:
        bundled_model.write_marker(path, model_name)
    return path


# --------------------------------------------------------------------------
# Finding the bundled model
# --------------------------------------------------------------------------

def test_no_bundled_model_falls_back_to_the_hub_name(tmp_path, monkeypatch):
    monkeypatch.delenv(bundled_model.ENV_VAR, raising=False)
    monkeypatch.setattr(bundled_model, "candidate_model_dirs",
                        lambda: [tmp_path / "nope"])
    assert bundled_model.bundled_model_dir(MODEL) is None
    assert bundled_model.resolve_model_source(MODEL) == MODEL


def test_bundled_model_dir_wins_over_the_hub_name(tmp_path, monkeypatch):
    model = _write_model_dir(tmp_path / "model")
    monkeypatch.delenv(bundled_model.ENV_VAR, raising=False)
    monkeypatch.setattr(bundled_model, "candidate_model_dirs", lambda: [model])
    assert bundled_model.resolve_model_source(MODEL) == str(model)


def test_a_different_model_is_never_silently_substituted(tmp_path, monkeypatch):
    """The configured model decides -- a bundled default must not override it.

    Regression guard: swapping in the bundled model would embed the vault with a
    model nobody chose, and a dimension mismatch makes VectorStore.load delete
    the existing index.
    """
    model = _write_model_dir(tmp_path / "model", model_name="bundled/default")
    monkeypatch.delenv(bundled_model.ENV_VAR, raising=False)
    monkeypatch.setattr(bundled_model, "candidate_model_dirs", lambda: [model])

    assert bundled_model.bundled_model_dir("user/choice") is None
    assert bundled_model.resolve_model_source("user/choice") == "user/choice"


def test_unmarked_model_dir_is_not_trusted(tmp_path, monkeypatch):
    """Files alone do not say which model they are, so they cannot be assumed."""
    model = _write_model_dir(tmp_path / "model", model_name=None)
    monkeypatch.delenv(bundled_model.ENV_VAR, raising=False)
    monkeypatch.setattr(bundled_model, "candidate_model_dirs", lambda: [model])
    assert bundled_model.bundled_model_dir(MODEL) is None


def test_marker_roundtrip(tmp_path):
    path = _write_model_dir(tmp_path / "model", model_name="a/b")
    assert bundled_model.read_marker(path) == "a/b"
    (path / bundled_model.MARKER_FILE).write_text("not json", encoding="utf-8")
    assert bundled_model.read_marker(path) is None


def test_incomplete_model_dir_is_rejected(tmp_path, monkeypatch):
    """A half-written directory must fall back, not fail deep inside torch."""
    partial = _write_model_dir(tmp_path / "partial", complete=False)
    monkeypatch.delenv(bundled_model.ENV_VAR, raising=False)
    monkeypatch.setattr(bundled_model, "candidate_model_dirs", lambda: [partial])
    assert bundled_model.bundled_model_dir(MODEL) is None


def test_env_override_wins_regardless_of_the_marker(tmp_path, monkeypatch):
    """Pointing at a directory is an explicit choice, so it needs no marker."""
    model = _write_model_dir(tmp_path / "override", model_name=None)
    monkeypatch.setenv(bundled_model.ENV_VAR, str(model))
    assert bundled_model.bundled_model_dir("anything") == model


def test_env_override_pointing_nowhere_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv(bundled_model.ENV_VAR, str(tmp_path / "missing"))
    assert bundled_model.resolve_model_source(MODEL) == MODEL


def test_frozen_bundle_looks_next_to_the_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    try:
        assert tmp_path / "assets" / "model" in bundled_model.candidate_model_dirs()
    finally:
        monkeypatch.undo()


def test_embedder_loads_the_bundled_directory(tmp_path, monkeypatch):
    """The backend must hand the resolved path to SentenceTransformer."""
    from brain_mcp.indexer import embedder as embedder_mod

    model = _write_model_dir(tmp_path / "model", model_name="hub/name")
    monkeypatch.delenv(bundled_model.ENV_VAR, raising=False)
    monkeypatch.setattr(bundled_model, "candidate_model_dirs", lambda: [model])

    seen: list[str] = []

    class _FakeST:
        def __init__(self, source):
            seen.append(source)

        def get_embedding_dimension(self):
            return 384

    fake = type(sys)("sentence_transformers")
    fake.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    backend = embedder_mod.SentenceTransformerBackend("hub/name")
    backend._load()
    assert seen == [str(model)]


# --------------------------------------------------------------------------
# The self-check itself
# --------------------------------------------------------------------------

def test_selfcheck_fails_and_reports_when_the_model_is_missing(tmp_path, monkeypatch):
    from brain import selfcheck

    monkeypatch.setattr(bundled_model, "bundled_model_dir", lambda *_: None)
    report = tmp_path / "selfcheck.json"

    assert selfcheck.run_selfcheck(report) == 1

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    names = {step["name"] for step in payload["steps"]}
    assert {"runtime_packages", "brain_mcp_modules", "embedding_model"} <= names
    model_step = next(s for s in payload["steps"] if s["name"] == "embedding_model")
    assert model_step["ok"] is False
    assert "HF_HUB_OFFLINE" in model_step["detail"]


def test_selfcheck_survives_a_windowed_build_without_stdout(tmp_path, monkeypatch):
    """console=False leaves sys.stdout/stderr as None -- printing must not crash."""
    from brain import selfcheck

    monkeypatch.setattr(bundled_model, "bundled_model_dir", lambda *_: None)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    assert selfcheck.run_selfcheck(tmp_path / "r.json") == 1


def test_selfcheck_blocks_the_hub_before_running_any_check(tmp_path, monkeypatch):
    """Offline must be forced before the first import, or it is not forced.

    huggingface_hub freezes its offline flag into a constant at import time, so
    a check that imports first and sets the variable afterwards would happily
    download on a networked build machine what the artefact is missing.
    """
    from brain import selfcheck

    seen: dict[str, str | None] = {}

    def _capture():
        seen["offline"] = os.environ.get("HF_HUB_OFFLINE")
        seen["transformers"] = os.environ.get("TRANSFORMERS_OFFLINE")
        home = os.environ.get("HF_HOME")
        seen["home"] = home
        # Read it here: run_selfcheck removes the directory when it returns.
        seen["home_entries"] = os.listdir(home) if home else None
        return [{"name": "stub", "ok": True, "detail": ""}]

    monkeypatch.setattr(selfcheck, "_run_steps", _capture)
    selfcheck.run_selfcheck(tmp_path / "r.json")

    assert seen["offline"] == "1"
    assert seen["transformers"] == "1"
    # A cache the build just warmed would hide a missing file in the bundle.
    assert seen["home"] and seen["home_entries"] == []


def test_bundle_model_verifies_in_a_hub_less_subprocess(tmp_path, monkeypatch):
    """The verification must not inherit this process's already-online HF stack."""
    spec = importlib.util.spec_from_file_location(
        "bundle_model", ROOT / "scripts" / "bundle_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: list[dict] = []

    class _Result:
        returncode = 0
        stdout = "384\n"
        stderr = ""

    def _fake_run(cmd, env=None, **kwargs):
        calls.append({"cmd": cmd, "env": env})
        return _Result()

    monkeypatch.setenv(bundled_model.ENV_VAR, str(tmp_path / "hijack"))
    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    target = tmp_path / "model"
    assert module.verify_offline(target) == 384

    call = calls[0]
    assert str(target) in call["cmd"], "must verify the directory it was given"
    assert call["env"]["HF_HUB_OFFLINE"] == "1"
    assert call["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert bundled_model.ENV_VAR not in call["env"], (
        "an override in the environment must not decide what was verified"
    )


def _dist_names(block: str) -> set[str]:
    return set(re.findall(r'"([A-Za-z0-9_.-]+)', block))


def _declared_dependencies() -> set[str]:
    """Base dependencies plus the ``dashboard`` extra the release installs.

    The extra matters: ``build.yml`` installs ``.[dashboard]``, so a new
    dashboard-only dependency ships inside the binary. Checking only the base
    list would leave exactly that blind spot open.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    base = _dist_names(text.split("dependencies = [", 1)[1].split("]", 1)[0])
    dashboard = _dist_names(text.split("dashboard = [", 1)[1].split("]", 1)[0])
    return {
        _DIST_TO_MODULE.get(name, name.replace("-", "_"))
        for name in base | dashboard
    }


def test_every_declared_dependency_is_either_checked_or_explained():
    """A new dependency must not slip past the smoke test unnoticed."""
    from brain import selfcheck

    accounted = set(selfcheck.RUNTIME_PACKAGES) | set(selfcheck.NOT_BUNDLED)
    unaccounted = _declared_dependencies() - accounted
    assert not unaccounted, (
        f"{sorted(unaccounted)} are runtime dependencies the packaged self-check "
        "neither imports nor excuses -- add them to RUNTIME_PACKAGES, or to "
        "NOT_BUNDLED with the reason the dashboard does not need them"
    )


def test_exclusions_are_declared_dependencies_with_a_reason():
    """NOT_BUNDLED must not turn into a place to hide a real gap."""
    from brain import selfcheck

    declared = _declared_dependencies()
    for name, reason in selfcheck.NOT_BUNDLED.items():
        assert name in declared, f"{name} is excused but not a declared dependency"
        assert name not in selfcheck.RUNTIME_PACKAGES, f"{name} is in both lists"
        assert len(reason) > 40, f"{name} needs a real reason, not '{reason}'"


# --------------------------------------------------------------------------
# The workflow and the spec
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def build_workflow() -> str:
    return BUILD_WORKFLOW.read_text(encoding="utf-8")


def test_build_installs_the_project_not_a_hand_written_list(build_workflow):
    assert 'pip install ".[dashboard]"' in build_workflow, (
        "the build must install the project so every runtime dependency ships"
    )


def test_build_does_not_reintroduce_the_partial_install(build_workflow):
    assert "pip install PyQt6 PyQt6-WebEngine numpy scipy requests" not in build_workflow


def test_build_bundles_the_model_before_packaging(build_workflow):
    assert "scripts/bundle_model.py" in build_workflow
    assert build_workflow.index("scripts/bundle_model.py") < build_workflow.index(
        "pyinstaller gystc.spec"
    ), "the model must be in assets/ before PyInstaller collects it"


def test_build_runs_the_binary_before_shipping_it(build_workflow):
    assert "--selfcheck" in build_workflow
    smoke = build_workflow.index("--selfcheck")
    for packaging in ("Create Windows ZIP", "Create macOS DMG", "Upload artifact"):
        assert smoke < build_workflow.index(packaging), (
            f"the self-check must run before '{packaging}'"
        )


def test_spec_bundles_the_selfcheck_and_model_resolver():
    spec = SPEC.read_text(encoding="utf-8")
    for name in ("brain.selfcheck", "brain_mcp.indexer.bundled_model",
                 "brain_mcp.indexer.pipeline", "brain_mcp.storage.file_lock"):
        assert f"'{name}'" in spec, f"{name} is imported at runtime but not bundled"
    assert "collect_all('sentence_transformers')" in spec, (
        "sentence-transformers loads its modules dynamically; static analysis misses them"
    )


def test_bundled_model_is_not_committed():
    """465 MB of weights belong in the build, never in git.

    Asks git, not .gitignore: an ignore rule does not untrack a path that was
    already added, so only the index can answer this.
    """
    assert "assets/model/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
    tracked = subprocess.run(
        ["git", "ls-files", "--", "assets/model"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        pytest.skip("git not available")
    assert tracked.stdout.strip() == "", (
        f"model weights are tracked in git:\n{tracked.stdout}"
    )
