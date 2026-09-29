"""resolve_within: the one answer to "does this path stay inside that folder?".

Every CLI entry point that takes a path from outside -- a proposals file, argv,
a flag -- funnels it through here before touching the disk. The cases below are
the classic ways out: `..`, an absolute path, a sibling folder that merely
shares a name prefix, and a symlink that points elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from brain_mcp.pathguard import resolve_within


@pytest.fixture
def base(tmp_path: Path) -> Path:
    b = tmp_path / "vault"
    (b / "sub").mkdir(parents=True)
    (b / "sub" / "note.md").write_text("x", encoding="utf-8")
    return b


def test_relative_path_inside_resolves_to_an_absolute_path(base):
    got = resolve_within("sub/note.md", base)
    assert got == (base / "sub" / "note.md").resolve()
    assert got.is_absolute()


def test_a_path_that_does_not_exist_yet_is_still_allowed(base):
    # apply_create and --out targets are new files: existence is not the test.
    assert resolve_within("sub/new/deeper.md", base) == (base / "sub/new/deeper.md").resolve()


def test_the_base_itself_counts_as_inside(base):
    assert resolve_within(".", base) == base.resolve()


def test_absolute_path_inside_is_allowed(base):
    inside = base / "sub" / "note.md"
    assert resolve_within(inside, base) == inside.resolve()


@pytest.mark.parametrize("escape", ["../outside.md", "sub/../../outside.md", "../../.."])
def test_dotdot_escapes_are_refused(base, escape):
    with pytest.raises(ValueError, match="escapes"):
        resolve_within(escape, base)


def test_absolute_path_outside_is_refused(base, tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        resolve_within(tmp_path / "elsewhere.json", base)


def test_sibling_with_the_same_name_prefix_is_refused(base, tmp_path):
    """A bare startswith(base) would let 'vault-evil' pass as inside 'vault'."""
    evil = tmp_path / "vault-evil"
    evil.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        resolve_within(evil / "x.md", base)


def test_symlink_pointing_outside_is_refused(base, tmp_path):
    outside = tmp_path / "secret"
    outside.mkdir()
    link = base / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks needs developer mode / privileges here")
    with pytest.raises(ValueError, match="escapes"):
        resolve_within("link/key.txt", base)


def test_error_names_the_offending_input(base):
    with pytest.raises(ValueError) as info:
        resolve_within("../outside.md", base)
    assert "../outside.md" in str(info.value)
