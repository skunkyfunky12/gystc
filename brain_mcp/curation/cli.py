"""`brain_mcp curate <init|analyze|apply>` — the safe executor for vault curation.

  init     -> Step 0: make the vault a local git repo + snapshot (reversibility).
  analyze  -> read-only: scan + emit a reviewable candidate list (writes nothing).
  apply    -> execute a CONFIRMED action set; dry-run unless --yes; one commit/run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from brain_mcp.config import load_config
from brain_mcp.curation.analyze import analyze
from brain_mcp.curation.apply import apply_actions, unified_diff
from brain_mcp.curation.vault_git import ensure_vault_repo, is_repo


def _vault_path(args) -> Path:
    if args.vault:
        return Path(args.vault)
    vp = load_config().vault_path
    if vp is None:
        print("ERROR: no vault path — pass --vault or configure gystc.", file=sys.stderr)
        sys.exit(1)
    return Path(vp)


def cmd_init(args) -> None:
    res = ensure_vault_repo(_vault_path(args))
    sha = (res["commit"] or "-")[:8]
    print(f"Vault git: {res['status']} (commit {sha}). Local-only, no remote.")


def cmd_analyze(args) -> None:
    v = _vault_path(args)
    if not is_repo(v):
        print("WARNING: vault is not a git repo yet — run `curate init` before any apply.",
              file=sys.stderr)
    # curation and indexer must agree about vault scope: honor exclude_dirs
    result = analyze(v, max_age_days=args.max_age_days,
                     extra_skip_dirs=load_config().exclude_dirs)
    s = result["stats"]
    print(f"Scanned {s['notes_scanned']} notes: {s['dead_link']} dead links · "
          f"{s['exact_duplicate']} exact-dupe groups · {s['orphan']} orphans · "
          f"{s['stale']} stale (>{int(args.max_age_days)}d).")
    if s["read_errors"]:
        print(f"WARNING: {s['read_errors']} note(s) could not be read — "
              "excluded from proposals (see stderr above).", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Reviewable proposal written: {args.out}")


def _preview_actions(vault: Path, actions: list[dict]) -> None:
    """The human gate must SHOW what it approves: a real diff for edits, the full
    target list for archives, the content for creates — never just 'op file'."""
    for a in actions:
        op, rel = a["op"], a["file"]
        target = vault / rel
        print(f"\n== {op} {rel}")
        if op == "edit":
            try:
                old = target.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                print(f"  (cannot read current content: {e})", file=sys.stderr)
                old = ""
            diff = unified_diff(old, a.get("new_content", ""), rel)
            print(diff if diff.strip() else "  (no content change)")
        elif op == "archive":
            if target.is_dir():
                files = sorted(p.relative_to(vault).as_posix()
                               for p in target.rglob("*") if p.is_file())
                print(f"  archives a DIRECTORY with {len(files)} file(s):")
                for f in files:
                    print(f"    {f}")
            else:
                print(f"  archives 1 file: {rel}")
        elif op == "create":
            content = a.get("new_content", "")
            print(f"  creates a new note ({len(content)} chars):")
            for line in content.splitlines():
                print(f"    {line}")


def cmd_apply(args) -> None:
    v = _vault_path(args)
    data = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
    actions = data.get("actions", []) if isinstance(data, dict) else data
    # carry the analyze-time fingerprints onto the actions: apply refuses (skips)
    # any note that changed in the review window instead of clobbering it
    fingerprints = data.get("fingerprints", {}) if isinstance(data, dict) else {}
    for a in actions:
        if "base_hash" not in a and fingerprints.get(a.get("file")):
            a["base_hash"] = fingerprints[a["file"]]
    _preview_actions(v, actions)  # --yes sees the same preview as the dry run
    if not args.yes:
        print(f"\nDRY RUN — {len(actions)} action(s) (pass --yes to apply):")
        for a in actions:
            print(f"  {a['op']:7} {a['file']}")
        return
    res = apply_actions(v, actions, run_message=args.message or "vault curation run")
    print(f"Applied {res['applied']} action(s); commit {(res['commit'] or '-')[:8]} "
          f"(revert with: git -C <vault> revert {(res['commit'] or '')[:8]}).")
    if res["skipped"]:
        print(f"WARNING: {len(res['skipped'])} action(s) SKIPPED — note changed since "
              "analyze (re-run analyze to refresh the proposal):", file=sys.stderr)
        for skip in res["skipped"]:
            print(f"  {skip['file']}: {skip['reason']}", file=sys.stderr)
        sys.exit(1)  # nonzero summary: the run did not fully apply


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="brain_mcp curate")
    sub = p.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="git-init the vault + snapshot (reversibility)")
    init_p.add_argument("--vault", default=None)
    init_p.set_defaults(func=cmd_init)

    an_p = sub.add_parser("analyze", help="read-only candidate analysis")
    an_p.add_argument("--vault", default=None)
    an_p.add_argument("--max-age-days", type=float, default=365.0, dest="max_age_days")
    an_p.add_argument("--out", default=None, help="write the proposal JSON here")
    an_p.set_defaults(func=cmd_analyze)

    ap_p = sub.add_parser("apply", help="apply a confirmed action set (dry-run unless --yes)")
    ap_p.add_argument("--vault", default=None)
    ap_p.add_argument("--proposals", required=True, help="JSON file with confirmed actions")
    ap_p.add_argument("--yes", action="store_true", help="actually apply (else dry-run)")
    ap_p.add_argument("--message", default=None, help="git commit message for this run")
    ap_p.set_defaults(func=cmd_apply)

    args = p.parse_args(argv)
    args.func(args)
