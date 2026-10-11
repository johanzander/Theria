from datetime import UTC, datetime, timedelta

import numpy as np

from core.theria.plan.planner import HourPlan, Plan
from core.theria.plan.room_model import RoomState
from core.theria.plan.scoring import ScoreBoard
from core.theria.plan.snapshot import PlanSnapshot

T0 = datetime(2026, 10, 12, 6, 0, tzinfo=UTC)


def snapshot(computed_at: datetime, predicted: list[float]) -> PlanSnapshot:
    hours = [
        HourPlan(
            start=T0 + timedelta(hours=i),
            rts_c=18.0,
            price_sek_per_kwh=1.0,
            extreme=False,
            floor_c=18.0,
            flow_target_c=36.0,
            energy_kwh=1.0,
            room_c_end=value,
            slab_c_end=28.0,
            energy_cost_sek=1.0,
            comfort_cost_sek=0.0,
            tail=False,
            reason={},
        )
        for i, value in enumerate(predicted)
    ]
    return PlanSnapshot(
        computed_at=computed_at,
        mode="shadow",
        plan=Plan(hours, hours, 0.0, 0.0, 0),
        extreme_threshold_sek_per_kwh=3.0,
        outdoor_bias_c=0.0,
        initial=RoomState(19.5, 28.0, 0.0),
        model_status={},
    )


def test_errors_are_grouped_by_lead_time_and_signed_as_predicted_minus_actual():
    # Plan made at 06:00 predicts the room at 07:00 (1 h) and 12:00 (6 h).
    plan = snapshot(T0, [19.8, 19.7, 19.6, 19.5, 19.4, 19.3])
    board = ScoreBoard()
    board.record(T0 + timedelta(hours=1, minutes=3), 19.5)  # 1 h: predicted 19.8
    board.record(T0 + timedelta(hours=6), 19.0)  # 6 h: predicted 19.3
    scores = {s.lead_h: s for s in board.scores([plan])}
    assert scores[1].n == 1 and np.isclose(scores[1].bias_c, 0.3)
    assert scores[6].n == 1 and np.isclose(scores[6].mae_c, 0.3)
    assert 24 not in scores


def test_predictions_without_a_nearby_reading_are_not_scored():
    plan = snapshot(T0, [19.8, 19.7])
    board = ScoreBoard()
    board.record(T0 + timedelta(hours=1, minutes=40), 19.0)
    assert board.scores([plan]) == []


def test_mean_absolute_error_ignores_sign():
    board = ScoreBoard()
    board.record(T0 + timedelta(hours=1), 19.0)
    plans = [snapshot(T0, [19.5]), snapshot(T0, [18.5])]
    (score,) = board.scores(plans)
    assert score.n == 2
    assert np.isclose(score.mae_c, 0.5) and np.isclose(score.bias_c, 0.0)
