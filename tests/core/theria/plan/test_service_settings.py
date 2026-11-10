
import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.settings_manager import SettingsManager

from .test_service_active import EVENING, make, run_cycle


def with_manager(settings, tmp_path, **kwargs):
    manager = SettingsManager(settings, str(tmp_path / "overrides.json"))
    service, ha, *rest = make(settings, settings_manager=manager, **kwargs)
    return service, ha, manager


def test_a_changed_setting_is_in_use_at_once(settings, tmp_path):
    service, _, _ = with_manager(settings, tmp_path)
    service.update_settings({"shift_min_gain_sek_per_kwh": 0.3})
    assert service._settings.shift_min_gain_sek_per_kwh == 0.3
    entry = {e["key"]: e for e in service.settings_catalog()}["shift_min_gain_sek_per_kwh"]
    assert entry["value"] == 0.3 and entry["source"] == "override"


def test_the_executors_use_the_new_range(settings, tmp_path):
    service, ha, _ = with_manager(settings, tmp_path, setpoint=18.0)
    service.update_settings({"rts_max_c": 19.0, "rts_min_c": 17.0})
    assert service._executor._s.rts_max_c == 19.0
    assert service._lk._s.rts_max_c == 19.0


def test_an_invalid_change_leaves_everything_as_it_was(settings, tmp_path):
    service, _, _ = with_manager(settings, tmp_path)
    with pytest.raises(ConfigurationError):
        service.update_settings({"floor_c": 21.0})
    assert service._settings.floor_c == settings.floor_c


def test_a_reset_goes_back_to_the_configured_value(settings, tmp_path):
    service, _, _ = with_manager(settings, tmp_path)
    service.update_settings({"lk_upper_c": 21.0})
    service.reset_setting("lk_upper_c")
    assert service._settings.lk_upper_c == settings.lk_upper_c


def test_a_plan_after_a_change_uses_it(settings, tmp_path):
    service, _, _ = with_manager(settings, tmp_path, clock=lambda: EVENING)
    service.update_settings({"rts_min_c": 17.5, "rts_max_c": 18.5})
    run_cycle(service)
    assert all(17.5 <= h.rts_c <= 18.5 for h in service.latest().plan.hours)
