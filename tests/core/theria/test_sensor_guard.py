import asyncio
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import yaml

from core.theria.exceptions import ConfigurationError, SensorError
from core.theria.sensor_guard import (
    EntityState,
    GuardSettings,
    SensorGuard,
    SensorSpec,
    evaluate_sensor,
    load_guard_settings,
    room_offsets,
)

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)

LK = SensorSpec(
    name="Vardagsrum LK",
    room="vardagsrum",
    temperature_entity="sensor.lk_v_temp",
    battery_entity="sensor.lk_v_batt",
    signal_entity="sensor.lk_v_rssi",
    reference=True,
    silent_warn_minutes=480,
    silent_fault_minutes=960,
)
SONOFF = SensorSpec(
    name="Vardagsrum Sonoff",
    room="vardagsrum",
    temperature_entity="sensor.son_v_temp",
    battery_entity="sensor.son_v_batt",
    silent_warn_minutes=360,
    silent_fault_minutes=720,
)
SETTINGS = GuardSettings(sensors=(LK, SONOFF))


def state(value: str, minutes_ago: float = 10) -> EntityState:
    return EntityState(value, NOW - timedelta(minutes=minutes_ago))


def codes(report):
    return {issue.code: issue.severity for issue in report.issues}


def test_a_fresh_sensor_with_good_battery_and_signal_is_ok():
    report = evaluate_sensor(
        LK, SETTINGS, state("20.3"), state("80"), state("-60"), NOW
    )
    assert report.status == "ok" and report.issues == ()
    assert (
        report.value_c == 20.3
        and report.battery_percent == 80
        and report.signal_dbm == -60
    )
    assert report.age_minutes == pytest.approx(10)


@pytest.mark.parametrize("value", ["unavailable", "unknown", "garbage"])
def test_unavailable_or_non_numeric_is_a_fault(value):
    report = evaluate_sensor(SONOFF, SETTINGS, state(value, 4000), None, None, NOW)
    assert report.status == "fault" and codes(report) == {"unavailable": "fault"}
    assert report.value_c is None
    assert (
        "HA har sett det" in report.issues[0].message
    )  # an age only HA knows about, not the real death


def test_silence_warns_then_faults_by_the_sensors_own_limits():
    warn = evaluate_sensor(SONOFF, SETTINGS, state("20.0", 400), None, None, NOW)
    fault = evaluate_sensor(SONOFF, SETTINGS, state("20.0", 800), None, None, NOW)
    assert codes(warn) == {"silent": "warning"} and warn.status == "warning"
    assert codes(fault) == {"silent": "fault"} and fault.status == "fault"
    lk_same_age = evaluate_sensor(LK, SETTINGS, state("20.0", 400), None, None, NOW)
    assert lk_same_age.status == "ok"  # LK is allowed to be quiet for longer


@pytest.mark.parametrize(
    ("battery", "severity"),
    [("25", "warning"), ("24", "warning"), ("10", "fault"), ("3", "fault")],
)
def test_low_battery_warns_at_25_and_faults_at_10_percent(battery, severity):
    report = evaluate_sensor(LK, SETTINGS, state("20.0"), state(battery), None, NOW)
    assert codes(report) == {"battery_low": severity}


def test_a_battery_just_above_the_limit_is_fine():
    report = evaluate_sensor(LK, SETTINGS, state("20.0"), state("26"), None, NOW)
    assert report.issues == ()


def test_weak_signal_warns():
    report = evaluate_sensor(LK, SETTINGS, state("20.0"), None, state("-88"), NOW)
    assert codes(report) == {"signal_weak": "warning"}


@pytest.mark.parametrize("value", ["-30", "80"])
def test_implausible_temperature_is_a_fault(value):
    report = evaluate_sensor(LK, SETTINGS, state(value), None, None, NOW)
    assert codes(report) == {"implausible": "fault"}


def test_the_worst_issue_decides_the_status():
    report = evaluate_sensor(
        SONOFF, SETTINGS, state("20.0", 400), state("5"), None, NOW
    )
    assert set(codes(report)) == {"silent", "battery_low"} and report.status == "fault"


def hourly(values):
    return np.array(values, dtype=float)


def test_offset_is_the_median_difference_to_the_reference_in_the_same_room():
    ref = hourly([20.0] * 100)
    other = hourly([21.7] * 100)
    offsets = room_offsets(
        SETTINGS, {LK.temperature_entity: ref, SONOFF.temperature_entity: other}
    )
    assert offsets[SONOFF.temperature_entity] == pytest.approx(1.7)
    assert offsets[LK.temperature_entity] == 0.0


def test_a_room_without_enough_overlap_has_no_offset():
    ref = hourly([20.0] * 100)
    other = hourly([np.nan] * 95 + [21.0] * 5)
    offsets = room_offsets(
        SETTINGS, {LK.temperature_entity: ref, SONOFF.temperature_entity: other}
    )
    assert SONOFF.temperature_entity not in offsets


class FakeHA:
    def __init__(self, states, history=None):
        self.states = dict(states)
        self.history = history or {}
        self.notified = []
        self.reads = []

    def get_state(self, entity_id):
        self.reads.append(entity_id)
        value, minutes_ago = self.states[entity_id]
        stamp = (NOW - timedelta(minutes=minutes_ago)).isoformat()
        return {"state": value, "last_changed": stamp, "attributes": {}}

    def get_numeric_history(self, entity_id, start, stop):
        return self.history.get(entity_id, [])


class FakeNotifier:
    def __init__(self):
        self.created, self.dismissed = [], []

    def notify(self, notification_id, title, message):
        self.created.append((notification_id, title, message))

    def dismiss(self, notification_id):
        self.dismissed.append(notification_id)


def make_guard(states, history=None, notifier=None, clock=lambda: NOW):
    ha = FakeHA(states, history)
    return SensorGuard(ha, SETTINGS, notifier=notifier, now=clock), ha


HEALTHY = {
    "sensor.lk_v_temp": ("20.0", 10),
    "sensor.lk_v_batt": ("80", 10),
    "sensor.lk_v_rssi": ("-60", 10),
    "sensor.son_v_temp": ("21.7", 30),
    "sensor.son_v_batt": ("90", 30),
}


def hourly_points(entity_value, hours=24 * 7):
    return [
        (NOW - timedelta(hours=h, minutes=30), entity_value)
        for h in range(hours, 0, -1)
    ]


def history_for(ref_value=20.0, other_value=21.7):
    return {
        "sensor.lk_v_temp": hourly_points(ref_value),
        "sensor.son_v_temp": hourly_points(other_value),
    }


def test_a_room_temperature_is_the_mean_of_healthy_sensors_corrected_to_the_reference():
    guard, _ = make_guard(HEALTHY, history_for())
    guard.check()
    assert guard.room_temperature() == pytest.approx(20.0)  # (20.0 + (21.7 - 1.7)) / 2


def test_a_faulty_sensor_is_left_out_of_the_room_temperature():
    states = {**HEALTHY, "sensor.son_v_temp": ("unavailable", 5000)}
    guard, _ = make_guard(states, history_for())
    guard.check()
    assert guard.room_temperature() == pytest.approx(20.0)
    assert {r.name: r.status for r in guard.reports()} == {
        "Vardagsrum LK": "ok",
        "Vardagsrum Sonoff": "fault",
    }


def test_with_no_usable_sensor_there_is_no_room_temperature():
    states = {
        **HEALTHY,
        "sensor.lk_v_temp": ("unavailable", 5000),
        "sensor.son_v_temp": ("unavailable", 5000),
    }
    guard, _ = make_guard(states, history_for())
    guard.check()
    with pytest.raises(SensorError):
        guard.room_temperature()


def test_a_non_reference_sensor_without_a_known_offset_is_not_trusted_alone():
    states = {**HEALTHY, "sensor.lk_v_temp": ("unavailable", 5000)}
    guard, _ = make_guard(states, history={})  # no history -> no offset
    guard.check()
    with pytest.raises(SensorError):
        guard.room_temperature()


def test_a_sensor_that_drifts_away_from_its_usual_difference_is_flagged():
    history = history_for()
    history["sensor.son_v_temp"] = hourly_points(21.7)[:-6] + hourly_points(
        23.9, hours=6
    )
    states = {**HEALTHY, "sensor.son_v_temp": ("23.9", 30)}
    guard, _ = make_guard(states, history)
    reports = {r.name: r for r in guard.check()}
    assert reports["Vardagsrum Sonoff"].status == "warning"
    assert "disagree" in {i.code for i in reports["Vardagsrum Sonoff"].issues}


def test_notification_is_sent_only_after_the_problem_persists_two_checks_and_once():
    notifier = FakeNotifier()
    states = {**HEALTHY, "sensor.son_v_temp": ("unavailable", 5000)}
    guard, _ = make_guard(states, history_for(), notifier)
    guard.scheduled_check()
    assert notifier.created == []  # first sighting: could be a restart
    guard.scheduled_check()
    assert len(notifier.created) == 1
    guard.scheduled_check()
    assert len(notifier.created) == 1  # not repeated
    notification_id, _, message = notifier.created[0]
    assert "son_v_temp" in notification_id and "Vardagsrum Sonoff" in message


def test_the_notification_is_dismissed_when_the_sensor_recovers():
    notifier = FakeNotifier()
    guard, ha = make_guard(
        {**HEALTHY, "sensor.son_v_temp": ("unavailable", 5000)}, history_for(), notifier
    )
    guard.scheduled_check()
    guard.scheduled_check()
    created_id = notifier.created[0][0]
    notifier.dismissed.clear()  # the start-up clean-up calls are tested elsewhere
    ha.states["sensor.son_v_temp"] = ("21.7", 5)
    guard.scheduled_check()
    assert notifier.dismissed == [created_id]


def test_a_problem_that_clears_before_being_confirmed_never_notifies():
    notifier = FakeNotifier()
    guard, ha = make_guard(
        {**HEALTHY, "sensor.son_v_temp": ("unavailable", 5000)}, history_for(), notifier
    )
    guard.scheduled_check()
    ha.states["sensor.son_v_temp"] = ("21.7", 5)
    guard.scheduled_check()
    assert notifier.created == []
    assert "theria_sensor_sensor_son_v_temp_unavailable" not in notifier.dismissed


def test_settings_reject_bad_input():
    with pytest.raises(ConfigurationError):
        GuardSettings.from_dict({"sensors": []})
    with pytest.raises(ConfigurationError):
        GuardSettings.from_dict({"sensors": [{"name": "x"}]})
    with pytest.raises(ConfigurationError):
        GuardSettings.from_dict(
            {
                "sensors": [
                    {"name": "a", "room": "r", "temperature_entity": "sensor.a"}
                ],
                "bogus": 1,
            }
        )


def test_settings_load_sensor_definitions_from_options():
    settings = GuardSettings.from_dict(
        {
            "notify": False,
            "sensors": [
                {
                    "name": "A",
                    "room": "r",
                    "temperature_entity": "sensor.a",
                    "battery_entity": "sensor.a_b",
                    "reference": True,
                    "silent_warn_minutes": 60,
                    "silent_fault_minutes": 120,
                }
            ],
        }
    )
    assert settings.notify is False and settings.sensors[0].reference is True
    assert settings.sensors[0].silent_fault_minutes == 120


SECTION = {
    "sensors": [
        {"name": "A", "room": "r", "temperature_entity": "sensor.a", "reference": True},
    ]
}


def test_guard_settings_load_from_addon_options_or_config_yaml(tmp_path):
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"sensor_guard": SECTION}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"options": {"sensor_guard": SECTION}}))
    assert load_guard_settings(str(options), str(config)).sensors[0].name == "A"
    assert (
        load_guard_settings(str(tmp_path / "none.json"), str(config)).sensors[0].name
        == "A"
    )
    assert (
        load_guard_settings(str(tmp_path / "none.json"), str(tmp_path / "none.yaml"))
        is None
    )


def test_guard_settings_are_none_without_a_section(tmp_path):
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"zones": []}))
    assert load_guard_settings(str(options), str(tmp_path / "none.yaml")) is None


def test_an_unexpected_error_during_a_scheduled_check_does_not_raise():
    guard, ha = make_guard(HEALTHY, history_for())
    ha.get_state = lambda entity_id: (_ for _ in ()).throw(KeyError(entity_id))
    asyncio.run(guard.cycle())  # supervisor boundary: must not raise


def test_a_silent_warning_is_shown_but_does_not_notify_while_a_silent_fault_does():
    notifier = FakeNotifier()
    states = {**HEALTHY, "sensor.son_v_temp": ("21.7", 400)}  # 6.7 h: warning only
    guard, ha = make_guard(states, history_for(), notifier)
    guard.scheduled_check()
    guard.scheduled_check()
    assert notifier.created == []
    ha.states["sensor.son_v_temp"] = ("21.7", 800)  # 13 h: fault
    guard.scheduled_check()
    guard.scheduled_check()
    assert len(notifier.created) == 1 and "silent" in notifier.created[0][0]


def test_only_scheduled_checks_count_towards_a_notification():
    notifier = FakeNotifier()
    guard, _ = make_guard(
        {**HEALTHY, "sensor.son_v_temp": ("unavailable", 5000)}, history_for(), notifier
    )
    for _ in range(3):
        guard.check()  # page loads, plan runs and manual rechecks
    assert notifier.created == []
    guard.scheduled_check()
    guard.scheduled_check()
    assert len(notifier.created) == 1


def test_a_worse_severity_of_the_same_problem_replaces_the_notification():
    notifier = FakeNotifier()
    states = {**HEALTHY, "sensor.lk_v_batt": ("20", 10)}  # battery warning
    guard, ha = make_guard(states, history_for(), notifier)
    guard.scheduled_check()
    guard.scheduled_check()
    ha.states["sensor.lk_v_batt"] = ("5", 10)  # battery fault
    guard.scheduled_check()
    guard.scheduled_check()
    assert len(notifier.created) == 2
    assert (
        notifier.created[0][0] == notifier.created[1][0]
    )  # same id: replaces, no pile-up
    assert "5 %" in notifier.created[1][2]


def test_notifications_left_by_a_previous_run_are_dismissed_on_the_first_scheduled_check():
    notifier = FakeNotifier()
    guard, _ = make_guard(HEALTHY, history_for(), notifier)
    guard.scheduled_check()
    assert "theria_sensor_sensor_son_v_temp_unavailable" in notifier.dismissed
    count = len(notifier.dismissed)
    guard.scheduled_check()
    assert len(notifier.dismissed) == count  # only once per start


def test_a_problem_present_at_start_is_not_dismissed():
    notifier = FakeNotifier()
    guard, _ = make_guard(
        {**HEALTHY, "sensor.son_v_temp": ("unavailable", 5000)}, history_for(), notifier
    )
    guard.scheduled_check()
    assert "theria_sensor_sensor_son_v_temp_unavailable" not in notifier.dismissed


def test_a_low_battery_is_a_maintenance_alert_not_a_reason_to_drop_a_valid_reading():
    states = {
        **HEALTHY,
        "sensor.lk_v_batt": ("8", 10),
        "sensor.son_v_temp": ("21.9", 30),  # corrected to 20.2
    }
    guard, _ = make_guard(states, history_for())
    guard.check()
    assert {r.name: r.status for r in guard.reports()}["Vardagsrum LK"] == "fault"
    assert guard.room_temperature() == pytest.approx(20.1)  # the LK (20.0) still counts


def test_rooms_count_equally_whatever_the_number_of_sensors_in_them():
    specs = (
        LK,
        SONOFF,
        SensorSpec("Bibliotek LK", "bibliotek", "sensor.lk_b_temp", reference=True),
        SensorSpec("Bibliotek Sonoff", "bibliotek", "sensor.son_b_temp"),
    )
    settings = GuardSettings(sensors=specs)
    states = {
        "sensor.lk_v_temp": ("20.0", 10),
        "sensor.lk_v_batt": ("80", 10),
        "sensor.lk_v_rssi": ("-60", 10),
        "sensor.son_v_temp": ("unavailable", 5000),
        "sensor.son_v_batt": ("unavailable", 5000),
        "sensor.lk_b_temp": ("22.0", 10),
        "sensor.son_b_temp": ("23.7", 10),
    }
    history = {
        **history_for(),
        "sensor.lk_b_temp": hourly_points(22.0),
        "sensor.son_b_temp": hourly_points(23.7),
    }
    guard = SensorGuard(FakeHA(states, history), settings, now=lambda: NOW)
    guard.check()
    # one healthy sensor in one room, two in the other: (20 + 22) / 2, not (20 + 22 + 22) / 3
    assert guard.room_temperature() == pytest.approx(21.0)


def test_every_room_needs_exactly_one_reference_sensor():
    lone = SensorSpec("Gäststuga", "gaststuga", "sensor.g_temp")  # no reference flag
    with pytest.raises(ConfigurationError):
        GuardSettings(sensors=(LK, SONOFF, lone))
    second = SensorSpec("Extra", "vardagsrum", "sensor.x_temp", reference=True)
    with pytest.raises(ConfigurationError):
        GuardSettings(sensors=(LK, second))
