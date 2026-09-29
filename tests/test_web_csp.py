"""The dashboard's Content-Security-Policy allows no inline script by default.

brain/web/index.html runs next to the local API token (window.__apiToken). With
'unsafe-inline' in script-src, any markup that slipped through escaping -- an
<img onerror=...> in a note title -- would run with that token. Without it, the
only inline script allowed is the import map, pinned by its SHA-256.

That pin is brittle by design: edit the import map and the browser refuses it,
three.js never loads, and the dashboard stays black without an error anywhere
a user would look. These tests turn that silent failure into a red suite.
"""
from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "brain" / "web"
INDEX = WEB / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    # Universal newlines, like the HTML parser: CRLF from a Windows checkout is
    # folded to LF before the browser computes the hash.
    return INDEX.read_text(encoding="utf-8")


def _directive(html: str, name: str) -> str:
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html)
    assert csp, "index.html lost its CSP meta tag"
    for part in csp.group(1).split(";"):
        tokens = part.split()
        if tokens and tokens[0] == name:
            return part
    raise AssertionError(f"CSP has no {name} directive")


def _sha256(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode() + "'"


def test_script_src_does_not_allow_inline_code(html):
    assert "'unsafe-inline'" not in _directive(html, "script-src")


def test_every_inline_script_is_pinned_by_its_hash(html):
    script_src = _directive(html, "script-src")
    inline = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
    assert inline, "expected at least the import map inline"
    for body in inline:
        assert _sha256(body) in script_src, (
            "an inline <script> changed (or was added) without updating its hash in "
            f"the CSP -- the browser will refuse it. Expected {_sha256(body)}"
        )


@pytest.mark.parametrize("name", ["index.html", "brain-app.js"])
def test_no_inline_event_handler_attributes(name):
    """onclick=/onerror= in markup is inline script: the CSP now blocks it, so
    it would silently do nothing. Wire handlers with addEventListener."""
    text = (WEB / name).read_text(encoding="utf-8")
    assert not re.search(r"<[a-zA-Z][^>]*\son[a-z]+\s*=", text)


def test_scripts_the_page_loads_exist():
    html = INDEX.read_text(encoding="utf-8")
    for src in re.findall(r'<script[^>]*\bsrc="\./([^"]+)"', html):
        assert (WEB / src).is_file(), f"index.html loads ./{src}, which does not exist"
