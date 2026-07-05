"""StateStore must round-trip state atomically and fall back to the rotated
.bak copy when the primary file is missing or corrupt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsid.state_store import StateStore


def test_load_returns_none_when_no_file_exists(tmp_path):
    store = StateStore(tmp_path / "state.json")
    assert store.load() is None


def test_save_then_load_round_trips(tmp_path):
    store = StateStore(tmp_path / "state.json")
    state = {"last_processed_bar_second": 123, "position": {"entry_price": 42.0}}
    store.save(state)

    loaded = store.load()
    assert loaded == state


def test_second_save_rotates_previous_to_backup(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.save({"last_processed_bar_second": 1, "position": None})
    store.save({"last_processed_bar_second": 2, "position": None})

    assert store.path.exists()
    assert store.backup_path.exists()
    assert store.load()["last_processed_bar_second"] == 2

    import json

    with open(store.backup_path) as f:
        backup = json.load(f)
    assert backup["last_processed_bar_second"] == 1


def test_corrupt_primary_falls_back_to_backup(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.save({"last_processed_bar_second": 1, "position": None})
    store.save({"last_processed_bar_second": 2, "position": None})

    store.path.write_text("{not valid json")

    loaded = store.load()
    assert loaded == {"last_processed_bar_second": 1, "position": None}


def test_missing_primary_falls_back_to_backup(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.save({"last_processed_bar_second": 1, "position": None})
    store.save({"last_processed_bar_second": 2, "position": None})

    store.path.unlink()

    loaded = store.load()
    assert loaded == {"last_processed_bar_second": 1, "position": None}
