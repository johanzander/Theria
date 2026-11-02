import json

import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.radiators.settings import GuestRoom, RadiatorPrefs
from core.theria.radiators.store import PrefsStore

DEFAULTS = RadiatorPrefs(guest_rooms=(GuestRoom("climate.bedroom", 18.0),))


def test_current_is_the_defaults_without_a_file(tmp_path):
    store = PrefsStore(DEFAULTS, str(tmp_path / "prefs.json"))
    assert store.current == DEFAULTS
    assert store.is_default()


def test_update_persists_and_a_new_store_reads_it(tmp_path):
    path = str(tmp_path / "prefs.json")
    PrefsStore(DEFAULTS, path).update({"salon_open_target_c": 19.5})
    again = PrefsStore(DEFAULTS, path)
    assert again.current.salon_open_target_c == 19.5
    assert not again.is_default()


def test_a_partial_update_keeps_the_other_fields(tmp_path):
    store = PrefsStore(DEFAULTS, str(tmp_path / "prefs.json"))
    store.update({"salon_open_target_c": 19.5})
    assert store.current.guest_rooms == DEFAULTS.guest_rooms
    assert store.current.salon_schedule == DEFAULTS.salon_schedule


def test_invalid_update_changes_nothing_and_raises(tmp_path):
    path = tmp_path / "prefs.json"
    store = PrefsStore(DEFAULTS, str(path))
    with pytest.raises(ConfigurationError):
        store.update({"salon_preheat_c": 10.0})
    assert store.current == DEFAULTS
    assert not path.exists()


def test_reset_returns_defaults_and_removes_the_override(tmp_path):
    path = tmp_path / "prefs.json"
    store = PrefsStore(DEFAULTS, str(path))
    store.update({"salon_open_target_c": 19.5})
    assert store.reset() == DEFAULTS
    assert store.is_default()
    assert not path.exists()


def test_a_damaged_file_is_ignored(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text("{not json")
    assert PrefsStore(DEFAULTS, str(path)).current == DEFAULTS


def test_a_saved_file_that_is_no_longer_valid_is_ignored(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({"mode": "active"}))
    assert PrefsStore(DEFAULTS, str(path)).current == DEFAULTS
