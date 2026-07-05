"""Crash-safe JSON state persistence for the paper-trading engine.

Only the open position and last-processed bar second are persisted --
RSI/pivot/divergence detector internal state is deliberately NOT saved (see
scripts/run_paper_trading.py); on restart those cold-start and re-warm over
their normal warm-up window, which is cheap. What must never silently
disappear is an open position's TP/SL levels, which this store protects with
an atomic write plus a rotated backup copy.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class StateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")
        self.tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.replace(self.backup_path)
        with open(self.tmp_path, "w") as f:
            json.dump(state, f, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self.tmp_path, self.path)

    def load(self) -> dict | None:
        for candidate in (self.path, self.backup_path):
            if not candidate.exists():
                continue
            try:
                with open(candidate) as f:
                    state = json.load(f)
                if candidate == self.backup_path:
                    logger.warning("state_store: primary state file unusable, loaded backup %s", candidate)
                return state
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("state_store: failed to load %s: %s", candidate, exc)
        return None
