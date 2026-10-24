import json

import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.settings import PlanSettings
from core.theria.plan.settings_manager import EDITABLE, SettingsManager


def manager(settings, tmp_path, overrides=None):
    path = tmp_path / "overrides.json"
    if overrides is not None:
        path.write_text(json.dumps(overrides))
    return SettingsManager(settings, str(path)), path


def test_every_editable_setting_is_a_real_setting_with_a_description(settings):
    fields = set(PlanSettings.__dataclass_fields__)
    for key, info in EDITABLE.items():
        assert key in fields, key
        assert info.label and info.description and info.group, key


def test_the_catalog_shows_value_default_and_where_the_value_comes_from(settings, tmp_path):
    m, _ = manager(settings, tmp_path)
    entry = {e["key"]: e for e in m.catalog()}["comfort_cost_sek_per_c_hour"]
    assert entry["value"] == entry["default"] == settings.comfort_cost_sek_per_c_hour
    assert entry["source"] == "default" and entry["kind"] == "float"


def test_a_change_is_applied_saved_and_marked_as_changed(settings, tmp_path):
    m, path = manager(settings, tmp_path)
    new = m.update({"comfort_cost_sek_per_c_hour": 0.4, "ceiling_c": 21.5})
    assert new.comfort_cost_sek_per_c_hour == 0.4 and new.ceiling_c == 21.5
    assert json.loads(path.read_text()) == {"comfort_cost_sek_per_c_hour": 0.4, "ceiling_c": 21.5}
    entry = {e["key"]: e for e in m.catalog()}["ceiling_c"]
    assert entry["source"] == "override" and entry["default"] == settings.ceiling_c


def test_a_saved_change_is_there_after_a_restart(settings, tmp_path):
    m, path = manager(settings, tmp_path)
    m.update({"floor_mix_max_c": 41.0})
    again = SettingsManager(settings, str(path))
    assert again.current.floor_mix_max_c == 41.0


def test_resetting_goes_back_to_the_configured_value(settings, tmp_path):
    m, path = manager(settings, tmp_path)
    m.update({"floor_mix_max_c": 41.0})
    assert m.reset("floor_mix_max_c").floor_mix_max_c == settings.floor_mix_max_c
    assert json.loads(path.read_text()) == {}


def test_an_unknown_setting_is_refused(settings, tmp_path):
    m, _ = manager(settings, tmp_path)
    with pytest.raises(ConfigurationError, match="not editable"):
        m.update({"outdoor_entity": "sensor.other"})


def test_a_value_out_of_range_or_of_the_wrong_kind_is_refused_and_nothing_is_saved(settings, tmp_path):
    m, path = manager(settings, tmp_path)
    with pytest.raises(ConfigurationError):
        m.update({"comfort_cost_sek_per_c_hour": -1})
    with pytest.raises(ConfigurationError):
        m.update({"regulator_active": "yes"})
    with pytest.raises(ConfigurationError):
        m.update({"ceiling_c": 50})
    assert not path.exists() or json.loads(path.read_text()) == {}


def test_a_combination_the_plan_cannot_use_is_refused(settings, tmp_path):
    m, _ = manager(settings, tmp_path)
    with pytest.raises(ConfigurationError):
        m.update({"floor_c": 21.0})  # above the target


def test_an_invalid_saved_override_is_ignored_at_start(settings, tmp_path):
    m, _ = manager(settings, tmp_path, overrides={"floor_c": 99.0, "ceiling_c": 21.0})
    assert m.current.floor_c == settings.floor_c and m.current.ceiling_c == 21.0


def test_a_whole_numbers_setting_takes_whole_numbers(settings, tmp_path):
    m, _ = manager(settings, tmp_path)
    assert m.update({"night_end_hour": 7}).night_end_hour == 7
    with pytest.raises(ConfigurationError):
        m.update({"night_end_hour": 6.5})
