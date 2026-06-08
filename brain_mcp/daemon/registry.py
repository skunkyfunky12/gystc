from __future__ import annotations
import json, os, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class DaemonInfo:
    port: int
    token: str
    pid: int

def write_registry(path: Path, info: DaemonInfo) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(info), f)
        try:
            os.chmod(tmp, 0o600)  # no-op on Windows, best-effort on POSIX
        except OSError:
            pass
        os.replace(tmp, str(path))
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

def read_registry(path: Path) -> DaemonInfo | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return DaemonInfo(port=int(d["port"]), token=str(d["token"]), pid=int(d["pid"]))
    except (OSError, ValueError, KeyError):
        return None
