"""The live plan of 2026-10-30 16:00 UTC, rebuilt from what the add-on's API showed: fitted models, prices,
the outdoor temperature behind the flow targets, the sun behind the swing, and the unmodelled drift that
reproduces the flat setpoint's room.

With the floor loops closable the plan bought heat in the last, dearest hours of the horizon (setpoint up to
20.5 C at 3.2 kr/kWh with the room at 19.2 C, comfort target 18.5 C): the end of the horizon was charged
against the room's current level, not against the comfort target. The room limit and the LK limit were
19.5 C that day, the room at 19.27 C.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.plan.planner import PlanInputs, plan
from core.theria.plan.pump_model import PumpParams
from core.theria.plan.room_model import RoomParams, RoomState, step
from core.theria.plan.settings import PlanSettings

PRICE = [3.5666,3.3984,3.19,2.9901,2.6413,2.4874,2.3755,2.5458,2.5006,2.4749,2.3977,2.413,2.4168,2.6353,2.7965,2.8339,2.6789,2.5354,2.4878,2.4372,2.5099,2.5555,2.7832,3.1653,3.2166,3.246,3.214,3.1585,2.938,2.7669,2.5954]
BASE_FLOW = [31.9867,32.0581,32.0581,32.2008,32.4148,32.6288,32.7715,32.6288,32.7715,32.9141,33.6275,34.0556,34.4123,34.2696,34.0556,33.9129,34.0556,33.5562,33.3422,32.9141,32.6288,32.5575,32.7715,33.3422,33.8416,34.2696,35.0543,35.411,35.8391,36.0531,36.1958]
BASE_ROOM = [19.2875,19.3056,19.3215,19.3339,19.3429,19.3487,19.3523,19.2981,19.2424,19.1872,19.1245,19.0618,19.0027,19.0339,19.0693,19.1013,19.1267,19.1504,19.1704,19.1891,19.2061,19.2206,19.2316,19.2376,19.2399,19.2395,19.2344,19.228,19.2201,19.2121,19.2046]
BASE_SLAB = [24.2189,24.5319,24.3621,24.0827,23.8273,23.6319,23.4916,23.3541,24.2298,25.5296,27.2301,29.1945,31.294,33.1222,33.0679,32.302,31.4949,30.4858,29.5936,28.7315,27.9813,27.3949,27.014,26.8827,26.8946,26.9979,27.3029,27.6345,28.0322,28.4192,28.7808]
SWING = [0.2187,0.1703,0.1327,0.1033,0.0805,0.0627,0.0488,0.038,0.0296,0.0231,0.018,0.014,0.0109,0.0085,0.0141,0.0574,0.1614,0.3173,0.5241,0.8108,1.0162,1.0579,0.9436,0.7439,0.5793,0.4512,0.3514,0.2737,0.2131,0.166,0.1293]
N = len(PRICE)
LK_LIMIT_C = 19.5
ROOM_C, SLAB_C, SUN_C = 19.269144711280806, 22.6345723556404, 41.50104511443773

PUMP = PumpParams(
    flow_intercept=40.55446185156523, flow_slope=-0.713396056407848, rts_ref=17.966123499142366,
    flow_per_rts=1.9, energy_intercept=-0.9567105114036295, energy_per_outdoor=-0.23629863316547184,
    energy_per_rts=0.2556666527581085, aux_onset_c=44.0, aux_per_flow_c=0.20613932898415657,
)
ROOM = RoomParams(
    tau_slab_h=1.0, k_slab=0.0035535882815427374, k_out=0.004352820687291827, k_sun=0.0,
    sun_lag_h=4.0, drift_c_per_h=0.0, sun_swing_c_per_wm2=0.006767426891172187,
)


def live_inputs() -> PlanInputs:
    t_out = np.array(
        [(PUMP.flow_intercept + PUMP.flow_per_rts * (18 - PUMP.rts_ref) - f) / -PUMP.flow_slope for f in BASE_FLOW]
    )
    k = 1 - np.exp(-1 / ROOM.sun_lag_h)
    sun_end = np.array(SWING) / ROOM.sun_swing_c_per_wm2
    sun_wm2 = np.empty(N)
    filtered = SUN_C
    for t in range(N):
        sun_wm2[t] = filtered + (sun_end[t] - filtered) / k
        filtered = sun_end[t]
    extra = np.empty(N)
    room, slab, filtered = ROOM_C, SLAB_C, SUN_C
    for t in range(N):
        room_next, _, filtered_next = step(
            ROOM, room, slab, filtered, min(BASE_FLOW[t], 40.0), t_out[t], sun_wm2[t], LK_LIMIT_C
        )
        extra[t] = BASE_ROOM[t] - float(room_next)
        room, slab, filtered = BASE_ROOM[t], BASE_SLAB[t], filtered_next
    start = datetime(2026, 10, 30, 16, tzinfo=UTC)
    return PlanInputs(
        hour_starts=[start + timedelta(hours=i) for i in range(N)],
        price_sek_per_kwh=np.array(PRICE), extreme=np.zeros(N, bool), t_out_c=t_out, sun_wm2=sun_wm2,
        initial=RoomState(ROOM_C, SLAB_C, SUN_C), previous_rts_c=16.0, extra_drift_c_per_h=extra,
    )


def live_settings(**changes) -> PlanSettings:
    return PlanSettings(**{
        "price_config_entry": "e", "outdoor_entity": "s.o", "room_setpoint_entity": "c.r",
        "flow_setpoint_entity": "s.f", "room_temp_entities": ("s.a",), "room_thermostat_entities": ("c.a",),
        "room_history_entities": (), "heating_mode_entity": "b.v", "flow_measured_entity": "s.fl",
        "energy_counter_entity": "s.e", "aux_energy_entity": "s.x", "aux_status_entity": "s.xs",
        "latitude": 55.5, "longitude": 12.9, "rts_min_c": 15.0, "rts_max_c": 22.0, "rts_base_c": 18.0,
        "floor_c": 18.5, "target_c": 18.5, "ceiling_c": 19.35, "extreme_floor_c": 17.0,
        "night_floor_c": 17.85, "night_target_c": 18.5, "lk_limit_c": LK_LIMIT_C, "lk_base_c": 19.5,
        "min_saving_sek": 0.2, "plan_closes_valves": True, "closed_energy_share": 0.15, **changes,
    })


def dearest_hours() -> np.ndarray:
    """The dear hours at the end of the horizon (the second day's evening peak)."""
    return (np.array(PRICE) >= 2.9) & (np.arange(N) >= 23)


@pytest.mark.parametrize("closing", [True, False])
def test_the_plan_does_not_buy_more_heat_than_the_flat_setpoint_in_the_dearest_hours(closing):
    result = plan(live_inputs(), live_settings(plan_closes_valves=closing), PUMP, ROOM)
    dear = dearest_hours()
    planned = sum(h.energy_kwh for h, d in zip(result.hours, dear, strict=True) if d)
    flat = sum(h.energy_kwh for h, d in zip(result.baseline_hours, dear, strict=True) if d)
    assert planned <= flat + 0.01


def test_the_plan_does_not_raise_the_setpoint_in_the_dearest_hours():
    result = plan(live_inputs(), live_settings(), PUMP, ROOM)
    rts = np.array([h.rts_c for h in result.hours])[dearest_hours()]
    assert rts.max() <= 18.0


def test_the_end_of_the_horizon_is_charged_against_the_comfort_target():
    result = plan(live_inputs(), live_settings(), PUMP, ROOM)
    assert result.end_reference_c == pytest.approx(18.5)
    assert all(h.room_c_end + h.sun_swing_c >= h.floor_c - 0.05 for h in result.hours)
