"""Download the embedding model into ``assets/model`` so the release can ship it.

The packaged app runs with ``HF_HUB_OFFLINE=1`` and must never reach the network,
which means the model has to travel inside the bundle. This script is the one
place that is allowed online, and it runs on the build machine only.

    python scripts/bundle_model.py [model-name] [--out assets/model]

It does not just download. It records which model the directory holds (so a
bundled default can never silently replace a model the user configured), then
re-loads exactly that directory in a **fresh process** with the hub blocked and
the cache redirected. The subprocess matters: ``huggingface_hub`` freezes its
offline flag into a constant at import time, so a check in this process -- which
has already imported the stack online -- would prove nothing. A download that
cannot be used offline afterwards is not a successful download.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain_mcp.config import DEFAULT_MODEL  # noqa: E402  (needs sys.path first)
from brain_mcp.indexer.bundled_model import (  # noqa: E402
    ENV_VAR,
    read_marker,
    write_marker,
)


def _fetch(model_name: str, out_dir: Path) -> None:
    # Explicitly online: a stray HF_HUB_OFFLINE in the environment would turn
    # this into a confusing "model not found" instead of a download.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from sentence_transformers import SentenceTransformer

    print(f"Downloading {model_name} ...", flush=True)
    model = SentenceTransformer(model_name)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    write_marker(out_dir, model_name)
    print(f"Saved to {out_dir}", flush=True)


_VERIFY_SNIPPET = """
import sys
from pathlib import Path
target = Path(sys.argv[1])
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(str(target))
vector = model.encode(["offline probe"], normalize_embeddings=True)
print(int(vector.shape[1]))
"""


def verify_offline(out_dir: Path) -> int:
    """Load *out_dir* in a fresh, hub-less process. Returns the dimension."""
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env.pop(ENV_VAR, None)  # never let an override decide what we verified
    with tempfile.TemporaryDirectory(prefix="gystc-verify-hf-") as empty_home:
        # An empty HF_HOME proves the files come from out_dir and not from the
        # cache this script just filled.
        env["HF_HOME"] = empty_home
        result = subprocess.run(
            [sys.executable, "-c", _VERIFY_SNIPPET, str(out_dir)],
            env=env, capture_output=True, text=True, cwd=str(ROOT),
        )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(
            f"{out_dir} does not load with the hub blocked -- it is not shippable"
        )
    return int(result.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=str(ROOT / "assets" / "model"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    _fetch(args.model, out_dir)

    recorded = read_marker(out_dir)
    if recorded != args.model:
        raise SystemExit(f"marker in {out_dir} says {recorded!r}, expected {args.model!r}")

    dim = verify_offline(out_dir)
    total_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"Offline load OK: {out_dir} -> {args.model}, dimension {dim}", flush=True)
    print(f"Bundled model size: {total_mb:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
