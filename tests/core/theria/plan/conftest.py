from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.plan.planner import PlanInputs
from core.theria.plan.room_model import RoomState
from core.theria.plan.settings import PlanSettings

HORIZON = 36
START = datetime(2026, 10, 12, 0, tzinfo=UTC)


@pytest.fixture
def settings() -> PlanSettings:
    return PlanSettings(
        price_entity="sensor.price",
        outdoor_entity="sensor.outdoor",
        room_setpoint_entity="climate.room_temp_setpoint",
        flow_setpoint_entity="sensor.heating_setpoint",
        room_temp_entities=("sensor.lk_bibliotek_temperature",),
        room_thermostat_entities=("climate.lk_bibliotek_thermostat",),
        room_history_entities=("sensor.sonoff_a480045bf5_temperature",),
        heating_mode_entity="binary_sensor.switch_valve_1",
        flow_measured_entity="sensor.radiator_forward",
        energy_counter_entity="sensor.compr_cons_heating",
        aux_energy_entity="sensor.aux_cons_heating",
        aux_status_entity="sensor.add_heat_status",
        min_saving_sek=0.0,  # the mechanics are tested; the threshold has its own tests
        latitude=55.578,
        longitude=12.927,
        night_end_hour=0,  # the night window is off unless a test turns it on
        rts_min_c=16.0,  # the planner's mechanics are tested on the old +/- 2 C range, whatever the production defaults are
        rts_max_c=20.0,
        floor_c=18.0,
    )


@pytest.fixture
def make_inputs():
    def _make(
        price,
        extreme=None,
        t_out=5.0,
        room_c=19.5,
        slab_c=28.0,
        previous_rts_c=18.0,
    ) -> PlanInputs:
        return PlanInputs(
            hour_starts=[START + timedelta(hours=i) for i in range(HORIZON)],
            price_sek_per_kwh=np.asarray(price, dtype=float),
            extreme=np.zeros(HORIZON, bool) if extreme is None else np.asarray(extreme),
            t_out_c=np.full(HORIZON, float(t_out)),
            sun_wm2=np.zeros(HORIZON),
            initial=RoomState(room_c, slab_c, 0.0),
            previous_rts_c=previous_rts_c,
        )

    return _make
