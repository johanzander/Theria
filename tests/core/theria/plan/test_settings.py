import json

import pytest
import yaml

from core.theria.exceptions import ConfigurationError
from core.theria.plan.settings import PlanSettings, load_plan_settings

BASE = {
    "price_config_entry": "entry",
    "outdoor_entity": "sensor.outdoor",
    "room_setpoint_entity": "climate.room_temp_setpoint",
    "flow_setpoint_entity": "sensor.heating_setpoint",
    "room_temp_entities": ["sensor.a", "sensor.b"],
    "room_thermostat_entities": ["climate.a"],
    "room_history_entities": ["sensor.old"],
    "heating_mode_entity": "binary_sensor.valve",
    "flow_measured_entity": "sensor.flow",
    "energy_counter_entity": "sensor.energy",
    "aux_energy_entity": "sensor.aux",
    "aux_status_entity": "sensor.aux_status",
    "latitude": 55.578,
    "longitude": 12.927,
}


def test_defaults_match_the_agreed_design():
    s = PlanSettings.from_dict(BASE)
    assert (s.floor_c, s.target_c, s.lk_base_c) == (18.5, 19.5, 19.5)
    assert s.mode == "shadow"
    assert s.room_temp_entities == ("sensor.a", "sensor.b")


@pytest.mark.parametrize(
    "override",
    [
        {"mode": "active"},
        {"floor_c": 19.9},  # above target
        {"rts_base_c": 30.0},  # outside range
        {"rts_step_c": 0.0},
        {"max_rts_step_per_hour_c": 0.1},
        {"room_temp_entities": []},
        {"room_thermostat_entities": []},
        {"shift_room_swing_c": 0.0},
        {"shift_rts_c": 0.0},
        {"lk_lower_day_c": 21.0},  # above the upper limit
        {"shift_closed_energy_share": 1.5},
        {"unknown_option": 1},
    ],
)
def test_invalid_settings_raise_configuration_error(override):
    with pytest.raises(ConfigurationError):
        PlanSettings.from_dict({**BASE, **override})


def test_missing_required_setting_raises_configuration_error():
    data = dict(BASE)
    del data["price_config_entry"]
    with pytest.raises(ConfigurationError):
        PlanSettings.from_dict(data)


def test_from_options_returns_none_without_a_plan_section():
    assert PlanSettings.from_options({"zones": []}) is None
    assert PlanSettings.from_options({"plan": BASE}).mode == "shadow"


def test_there_is_no_off_mode_a_plan_is_always_made():
    with pytest.raises(ConfigurationError):
        PlanSettings.from_options({"plan": {**BASE, "mode": "off"}})


def test_load_prefers_addon_options_json_over_config_yaml(tmp_path):
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"plan": {**BASE, "mode": "shadow"}}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"options": {"plan": {**BASE, "mode": "bogus"}}}))
    assert load_plan_settings(str(options), str(config)).mode == "shadow"


def test_load_falls_back_to_config_yaml_in_development(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"options": {"plan": BASE}}))
    settings = load_plan_settings(str(tmp_path / "missing.json"), str(config))
    assert settings.price_config_entry == "entry"


def test_load_returns_none_when_no_file_exists_as_in_the_container_image(tmp_path):
    assert (
        load_plan_settings(str(tmp_path / "a.json"), str(tmp_path / "b.yaml")) is None
    )


@pytest.mark.parametrize("text", ["{not json", "[1, 2]"])
def test_load_raises_configuration_error_for_unreadable_options(tmp_path, text):
    options = tmp_path / "options.json"
    options.write_text(text)
    with pytest.raises(ConfigurationError):
        load_plan_settings(str(options), str(tmp_path / "b.yaml"))


def test_load_raises_configuration_error_for_invalid_yaml(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("options: [unclosed")
    with pytest.raises(ConfigurationError):
        load_plan_settings(str(tmp_path / "a.json"), str(config))


def test_floor_loops_are_read_as_a_list_of_rooms():
    s = PlanSettings.from_dict(
        {**BASE, "room_actuators": [{"room": "bib", "entities": ["binary_sensor.a"]}]}
    )
    assert s.room_actuators == {"bib": ("binary_sensor.a",)}


def test_the_plan_starts_with_three_quarters_of_the_setpoint_range_and_the_minimum_can_be_changed(settings):
    from dataclasses import replace

    import pytest

    from core.theria.exceptions import ConfigurationError

    assert settings.min_authority == 0.75
    assert replace(settings, min_authority=1.0).min_authority == 1.0
    for bad in (0.0, 1.5):
        with pytest.raises(ConfigurationError):
            replace(settings, min_authority=bad)


def test_the_setpoint_shift_stays_inside_what_has_been_measured():
    with pytest.raises(ConfigurationError):
        PlanSettings.from_dict({**BASE, "shift_rts_c": 2.5})


def test_the_options_of_the_removed_optimizer_can_still_be_stored_in_home_assistant(
    settings,
):
    from dataclasses import asdict

    from core.theria.plan.settings import PlanSettings

    stored = {
        k: v
        for k, v in asdict(settings).items()
        if k not in ("room_actuators",)
    }
    loaded = PlanSettings.from_dict(
        {
            **stored,
            "radiator_entities": ["climate.a"],
            "plan_strategy": "optimizer",
            "ceiling_c": 22.0,
            "room_controls": [],
        }
    )
    assert loaded.outdoor_entity == settings.outdoor_entity
