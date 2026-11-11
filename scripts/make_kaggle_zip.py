"""Package this project for Kaggle without git: dist/infer-server.zip.

    python scripts/make_kaggle_zip.py

Upload the zip as a private Kaggle dataset and attach it to notebooks/kaggle_benchmarks.ipynb.
Everything sits under one top-level infer-server/ folder. A SOURCE_DIGEST file (sha256 over every
packaged file) stands in for the git commit, so each benchmark result records exactly which code
produced it.
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "infer-server.zip"
SKIP_DIRS = {".git", ".github", "dist", "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis",
             ".ipynb_checkpoints", ".venv", "logs", "_unzipped"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".png", ".gif", ".mp4"}   # media: README and docs only
SKIP_PREFIXES = ("bench/results/", "docs/")             # benchmark outputs and docs, not inputs


def project_files() -> list[Path]:
    files = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if (p.is_file() and not (set(rel.parts) & SKIP_DIRS) and p.suffix not in SKIP_SUFFIXES
                and not rel.as_posix().startswith(SKIP_PREFIXES)):
            files.append(p)
    return files


def main() -> None:
    files = project_files()
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes())
    stamp = f"sha256:{digest.hexdigest()} built {time.strftime('%Y-%m-%d %H:%M:%S')} ({len(files)} files)\n"
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, "infer-server/" + p.relative_to(ROOT).as_posix())
        z.writestr("infer-server/SOURCE_DIGEST", stamp)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB, {len(files)} files)")
    print(stamp.strip())


if __name__ == "__main__":
    main()
