"""What made a build: the code (commit, dirty flag, and a hash of the code files that holds
whether or not the work was committed) and the environment (interpreter, main packages).
The spoke standard's provenance (~/Dropbox/portal_pivot_db/docs/SPOKES.md §4)."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: the code that determines kenpom's outputs
CODE = ("models/*.py", "scripts/rolling_ratings.py", "scripts/build_team_ratings.py",
        "scripts/build_intl_team_ratings.py", "update_ratings.sh")


def tree_sha256(repo: Path = REPO) -> str:
    """sha256 over (path, sha256 of contents) of every code file, in path order."""
    h = hashlib.sha256()
    for f in sorted({p for pattern in CODE for p in repo.glob(pattern) if p.is_file()}):
        h.update(f"{f.relative_to(repo).as_posix()}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def code_state(repo: Path = REPO) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD") or None
        dirty = bool(git("status", "--porcelain", "--", "models", "scripts", "update_ratings.sh"))
    except OSError:
        commit, dirty = None, None
    return {"tree_sha256": tree_sha256(repo), "commit": commit, "dirty": dirty}


def environment() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "pandas", "pyarrow"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            pass
    return out
