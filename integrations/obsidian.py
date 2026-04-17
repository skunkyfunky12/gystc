"""Obsidian REST API integration for opening notes on click."""

import urllib.parse

import requests

OBSIDIAN_HOST = "http://127.0.0.1:27123"


def build_open_url(note_path: str, vault_path: str | None = None) -> str:
    """Build the Obsidian Local REST API URL to open a note.

    Args:
        note_path: Relative vault path or absolute filesystem path to the note.
        vault_path: Absolute path to the vault root. If provided and note_path
                    starts with it, the vault prefix is stripped first.

    Returns:
        Full URL string like ``http://127.0.0.1:27123/open/<encoded-path>``.
    """
    path = note_path

    # Strip vault prefix when an absolute path is passed
    if vault_path is not None:
        # Normalise separators so comparison is reliable
        norm_path = path.replace("\\", "/")
        norm_vault = vault_path.rstrip("/\\").replace("\\", "/")
        if norm_path.startswith(norm_vault):
            path = norm_path[len(norm_vault):]

    # Remove any leading slash or backslash
    path = path.lstrip("/\\")

    encoded = urllib.parse.quote(path, safe="/")
    return f"{OBSIDIAN_HOST}/open/{encoded}"


def open_in_obsidian(
    note_path: str,
    api_key: str,
    vault_path: str | None = None,
) -> bool:
    """Open a note in Obsidian via its Local REST API.

    Args:
        note_path: Relative vault path or absolute filesystem path to the note.
        api_key: Bearer token for the Obsidian Local REST API plugin.
        vault_path: Optional absolute vault root path (forwarded to build_open_url).

    Returns:
        True if Obsidian accepted the request (HTTP 2xx), False otherwise.
    """
    url = build_open_url(note_path, vault_path=vault_path)
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(url, headers=headers, timeout=3)
        return response.ok
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def open_node_in_obsidian(
    node: dict,
    api_key: str,
    vault_path: str | None = None,
) -> bool:
    """Open the source note of a graph node in Obsidian.

    Args:
        node: Graph node dict that may contain a ``source_file`` key.
        api_key: Bearer token for the Obsidian Local REST API plugin.
        vault_path: Optional absolute vault root path.

    Returns:
        True if the note was opened successfully, False if no source_file is
        present or the request failed.
    """
    source_file = node.get("source_file")
    if not source_file:
        return False
    return open_in_obsidian(source_file, api_key, vault_path=vault_path)
