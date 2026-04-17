"""Tests for integrations/obsidian.py."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# build_open_url
# ---------------------------------------------------------------------------

def test_build_open_url_encodes_spaces():
    from integrations.obsidian import build_open_url

    url = build_open_url("02 Projekte/My Note.md")
    assert "02%20Projekte" in url
    assert "My%20Note.md" in url
    assert url.startswith("http://127.0.0.1:27123/open/")


def test_build_open_url_strips_vault_prefix():
    from integrations.obsidian import build_open_url

    url = build_open_url(
        "%USERPROFILE%/Desktop/REDACTED/Claude Stuff/Claude Brain Vault/Claude Brain/02 Projekte/test.md",
        vault_path="%USERPROFILE%/Desktop/REDACTED/Claude Stuff/Claude Brain Vault/Claude Brain",
    )
    assert "02%20Projekte/test.md" in url
    assert "C:" not in url


# ---------------------------------------------------------------------------
# open_in_obsidian
# ---------------------------------------------------------------------------

def test_open_in_obsidian_returns_true_on_success():
    from integrations.obsidian import open_in_obsidian

    mock_resp = MagicMock()
    mock_resp.ok = True
    with patch("integrations.obsidian.requests.post", return_value=mock_resp) as mock_post:
        result = open_in_obsidian("some/note.md", api_key="test-key")
    assert result is True
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert kwargs["timeout"] == 3


def test_open_in_obsidian_returns_false_on_http_error():
    from integrations.obsidian import open_in_obsidian

    mock_resp = MagicMock()
    mock_resp.ok = False
    with patch("integrations.obsidian.requests.post", return_value=mock_resp):
        result = open_in_obsidian("some/note.md", api_key="test-key")
    assert result is False


def test_open_in_obsidian_returns_false_on_connection_error():
    import requests as _requests
    from integrations.obsidian import open_in_obsidian

    with patch("integrations.obsidian.requests.post", side_effect=_requests.exceptions.ConnectionError):
        result = open_in_obsidian("some/note.md", api_key="test-key")
    assert result is False


def test_open_in_obsidian_returns_false_on_timeout():
    import requests as _requests
    from integrations.obsidian import open_in_obsidian

    with patch("integrations.obsidian.requests.post", side_effect=_requests.exceptions.Timeout):
        result = open_in_obsidian("some/note.md", api_key="test-key")
    assert result is False


# ---------------------------------------------------------------------------
# open_node_in_obsidian
# ---------------------------------------------------------------------------

def test_open_node_in_obsidian_calls_open_in_obsidian():
    from integrations.obsidian import open_node_in_obsidian

    node = {"id": "abc", "source_file": "03 Areas/note.md"}
    with patch("integrations.obsidian.open_in_obsidian", return_value=True) as mock_open:
        result = open_node_in_obsidian(node, api_key="key", vault_path="/vault")
    assert result is True
    mock_open.assert_called_once_with("03 Areas/note.md", "key", vault_path="/vault")


def test_open_node_in_obsidian_returns_false_when_no_source_file():
    from integrations.obsidian import open_node_in_obsidian

    node = {"id": "abc"}
    result = open_node_in_obsidian(node, api_key="key")
    assert result is False
