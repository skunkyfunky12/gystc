#!/usr/bin/env python3
"""Secret guard — blocks API keys / tokens / private keys from being committed or pushed.

Used by .githooks/pre-commit (scans staged files) and .githooks/pre-push (scans all
tracked files). Run manually:  python scripts/check_secrets.py [path ...]
Exit code 1 if anything suspicious is found.
"""
from __future__ import annotations

import re
import subprocess
import sys

# (label, compiled pattern) — high-signal credential shapes.
PATTERNS = [
    ("Anthropic key",        re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI key",           re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}\b")),
    ("AWS access key id",    re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token",         re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("Google API key",       re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("Slack token",          re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("Stripe key",           re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{20,}\b")),
    ("Private key block",    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("JWT / Supabase token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{6,}")),
    ("Secret assignment",    re.compile(
        r"""(?i)(api[_-]?key|secret|token|passwd|password|access[_-]?key|private[_-]?key|encryption[_-]?key)"""
        r"""\s*[:=]\s*['"][A-Za-z0-9_\-/+=.]{16,}['"]""")),
]

# Lines that are clearly NOT real secrets (placeholders, env-var indirection, key NAMES).
ALLOW = re.compile(
    r"""(?i)(example|placeholder|your[_-]?key|dummy|changeme|redacted|xxxx|<[^>]+>|"""
    r"""\$\{?[A-Z_]+\}?|process\.env|os\.environ|getenv|import\.meta\.env|"""
    r"""\.get\(|\.text\(\)|input\(|=\s*['"]['"])"""
)


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True,
    ).stdout
    return [f for f in out.splitlines() if f.strip()]


def scan(paths: list[str]) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    for path in paths:
        if path.endswith((".png", ".jpg", ".jpeg", ".ico", ".icns", ".svg",
                          ".faiss", ".db", ".zip", ".dmg", ".mp4", ".woff", ".woff2")):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if ALLOW.search(line):
                        continue
                    for name, pat in PATTERNS:
                        m = pat.search(line)
                        if m:
                            tok = m.group(0)
                            red = (tok[:4] + "…" + tok[-2:]) if len(tok) > 10 else "…"
                            findings.append((path, i, name, red))
                            break
        except (OSError, IsADirectoryError):
            continue
    return findings


def main() -> int:
    paths = sys.argv[1:] or staged_files()
    findings = scan(paths)
    if findings:
        sys.stderr.write("\n⛔ Potential secret(s) detected — commit/push blocked:\n\n")
        for path, line, name, red in findings:
            sys.stderr.write(f"  {path}:{line}  [{name}]  {red}\n")
        sys.stderr.write(
            "\nMove the value into an untracked file / env var. "
            "If it's a genuine false positive, re-run with --no-verify after checking.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
