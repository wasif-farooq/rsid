"""Small shared helper for scripts that default to "most recent matching file"
so you don't have to pass exact parquet filenames (which embed a date range)
on every invocation.
"""

from pathlib import Path


def find_latest(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")
    return candidates[-1]
