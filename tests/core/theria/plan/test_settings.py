import json

import pytest
import yaml

from core.theria.exceptions import ConfigurationError
from core.theria.plan.settings import PlanSettings, load_plan_settings

BASE = {
    "price_entity": "sensor.price",
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
    assert (s.floor_c, s.extreme_floor_c, s.target_c, s.ceiling_c) == (
        18.0,
        17.0,
        19.5,
        21.0,
    )
    assert s.comfort_cost_sek_per_c_hour == 1.0
    assert s.mode == "off"
    assert s.room_temp_entities == ("sensor.a", "sensor.b")


@pytest.mark.parametrize(
    "override",
    [
        {"mode": "active"},
        {"floor_c": 19.9},  # above target
        {"extreme_floor_c": 18.5},  # above floor
        {"rts_base_c": 30.0},  # outside range
        {"rts_step_c": 0.0},
        {"max_rts_step_per_hour_c": 0.1},
        {"comfort_cost_sek_per_c_hour": -1.0},
        {"setpoint_deviation_cost_sek_per_c_hour": -0.1},
        {"extreme_price_percentile": 100.0},
        {"room_temp_entities": []},
        {"room_thermostat_entities": []},
        {"boost_flow_limit_c": 29.0},  # below any useful flow temperature
        {"boost_flow_limit_c": 46.0},  # above the floor limit
        {"max_flow_c": 29.0},  # below any useful flow temperature
        {"max_flow_c": 63.0},  # above the pump limit of 62 C
        {"unknown_option": 1},
    ],
)
def test_invalid_settings_raise_configuration_error(override):
    with pytest.raises(ConfigurationError):
        PlanSettings.from_dict({**BASE, **override})


def test_missing_required_setting_raises_configuration_error():
    data = dict(BASE)
    del data["price_entity"]
    with pytest.raises(ConfigurationError):
        PlanSettings.from_dict(data)


def test_from_options_returns_none_without_a_plan_section():
    assert PlanSettings.from_options({"zones": []}) is None
    assert PlanSettings.from_options({"plan": BASE}).mode == "off"


def test_load_prefers_addon_options_json_over_config_yaml(tmp_path):
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"plan": {**BASE, "mode": "shadow"}}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"options": {"plan": {**BASE, "mode": "off"}}}))
    assert load_plan_settings(str(options), str(config)).mode == "shadow"


def test_load_falls_back_to_config_yaml_in_development(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"options": {"plan": BASE}}))
    settings = load_plan_settings(str(tmp_path / "missing.json"), str(config))
    assert settings.price_entity == "sensor.price"


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


def test_max_flow_default_is_the_flow_temperature_lk_designed_for():
    assert PlanSettings.from_dict(BASE).max_flow_c == 45.0


def test_deviating_from_the_base_setpoint_has_a_default_cost_for_the_rooms_that_react_faster():
    assert PlanSettings.from_dict(BASE).setpoint_deviation_cost_sek_per_c_hour == 0.1
