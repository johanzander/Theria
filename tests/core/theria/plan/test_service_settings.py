from dataclasses import replace

import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.settings import RoomControl
from core.theria.plan.settings_manager import SettingsManager

from .test_service_active import EVENING, make, run_cycle


def with_manager(settings, tmp_path, **kwargs):
    manager = SettingsManager(settings, str(tmp_path / "overrides.json"))
    service, ha, *rest = make(settings, settings_manager=manager, **kwargs)
    return service, ha, manager


def test_a_changed_setting_is_in_use_at_once(settings, tmp_path):
    service, _, _ = with_manager(settings, tmp_path)
    service.update_settings({"comfort_cost_sek_per_c_hour": 0.3})
    assert service._settings.comfort_cost_sek_per_c_hour == 0.3
    entry = {e["key"]: e for e in service.settings_catalog()}["comfort_cost_sek_per_c_hour"]
    assert entry["value"] == 0.3 and entry["source"] == "override"


def test_the_executors_use_the_new_range(settings, tmp_path):
    service, ha, _ = with_manager(settings, tmp_path, setpoint=18.0)
    service.update_settings({"rts_max_c": 19.0, "rts_min_c": 17.0})
    assert service._executor._s.rts_max_c == 19.0
    assert service._lk._s.lk_limit_c == settings.lk_limit_c


def test_a_new_limit_reaches_the_regulator_and_its_learned_rise_is_kept(settings, tmp_path):
    control = RoomControl("bibliotek", "climate.lk_bibliotek_thermostat", ("sensor.lk_bibliotek_temperature",))
    manager = SettingsManager(replace(settings, room_controls=(control,)), str(tmp_path / "o.json"))
    service, *_ = make(replace(settings, room_controls=(control,)), settings_manager=manager)
    service._regulator._rooms["bibliotek"].rise_c = 0.6
    service.update_settings({"lk_limit_c": 21.0, "ceiling_c": 21.0})
    room = service._regulator._rooms["bibliotek"]
    assert room.limit_c == 21.0 and room.rise_c == 0.6


def test_an_invalid_change_leaves_everything_as_it_was(settings, tmp_path):
    service, _, _ = with_manager(settings, tmp_path)
    with pytest.raises(ConfigurationError):
        service.update_settings({"floor_c": 21.0})
    assert service._settings.floor_c == settings.floor_c


def test_a_reset_goes_back_to_the_configured_value(settings, tmp_path):
    service, _, _ = with_manager(settings, tmp_path)
    service.update_settings({"ceiling_c": 21.0})
    service.reset_setting("ceiling_c")
    assert service._settings.ceiling_c == settings.ceiling_c


def test_a_plan_after_a_change_uses_it(settings, tmp_path):
    service, _, _ = with_manager(settings, tmp_path, clock=lambda: EVENING)
    service.update_settings({"rts_min_c": 17.5, "rts_max_c": 18.5})
    run_cycle(service)
    assert all(17.5 <= h.rts_c <= 18.5 for h in service.latest().plan.hours)
