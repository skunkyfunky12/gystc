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
from brain_mcp.curation.apply import apply_actions
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
    result = analyze(v, max_age_days=args.max_age_days)
    s = result["stats"]
    print(f"Scanned {s['notes_scanned']} notes: {s['dead_link']} dead links · "
          f"{s['exact_duplicate']} exact-dupe groups · {s['orphan']} orphans · "
          f"{s['stale']} stale (>{int(args.max_age_days)}d).")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Reviewable proposal written: {args.out}")


def cmd_apply(args) -> None:
    v = _vault_path(args)
    data = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
    actions = data.get("actions", []) if isinstance(data, dict) else data
    if not args.yes:
        print(f"DRY RUN — {len(actions)} action(s) (pass --yes to apply):")
        for a in actions:
            print(f"  {a['op']:7} {a['file']}")
        return
    res = apply_actions(v, actions, run_message=args.message or "vault curation run")
    print(f"Applied {res['applied']} action(s); commit {(res['commit'] or '-')[:8]} "
          f"(revert with: git -C <vault> revert {(res['commit'] or '')[:8]}).")


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
