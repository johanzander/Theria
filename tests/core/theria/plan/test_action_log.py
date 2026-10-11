from .test_service_active import make, run_cycle
from .test_service_lk import CHEAP_BEFORE_PEAK


def kinds(service):
    return [(e["kind"], e["action"]) for e in service.action_log()]


def test_writes_to_the_pump_and_the_thermostats_are_logged_with_the_time(settings):
    service, *_ = make(settings, clock=lambda: CHEAP_BEFORE_PEAK)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ("pump", "wrote") in kinds(service) and ("lk", "wrote") in kinds(service)
    entry = next(e for e in service.action_log() if e["kind"] == "pump")
    assert entry["at"] == CHEAP_BEFORE_PEAK and "18.5" in entry["detail"]


def test_cycles_that_change_nothing_are_not_logged(settings):
    service, *_ = make(settings, setpoint=18.0)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    first = len(service.action_log())
    run_cycle(service)  # same hour, nothing new to write
    assert len(service.action_log()) == first


def test_mode_changes_and_trouble_are_logged(settings):
    service, *_ = make(
        settings, clock=lambda: CHEAP_BEFORE_PEAK, trouble="stopped", setpoint=17.0
    )
    service.set_mode("active", confirm=True)
    run_cycle(service)
    service.set_mode("shadow")
    seen = kinds(service)
    assert ("mode", "active") in seen and ("mode", "shadow") in seen
    assert ("trouble", "stepped back") in seen


def test_the_log_is_newest_last_and_limited(settings):
    service, *_ = make(settings)
    for i in range(300):
        service._log("mode", "x", str(i))
    log = service.action_log()
    assert len(log) == 200 and log[-1]["detail"] == "299"
