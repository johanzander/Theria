"""Ready-made fit results for tests that do not fit anything."""

from core.theria.plan.daily_room import DailyRoomFit, SeasonFit


def daily_fit(gain: float = 0.39, rate: float = 0.21) -> DailyRoomFit:
    return DailyRoomFit(
        gain_c_per_c=gain,
        rate_per_day=rate,
        time_constant_days=1 / rate,
        seasons=(SeasonFit(2023, 67, 0.44, 0.46), SeasonFit(2025, 181, 0.25, 0.07)),
        windows=248,
        cv_rmse_c={1: 0.48, 2: 0.66, 3: 0.80},
        cv_persistence_rmse_c={1: 0.49, 2: 0.67, 3: 0.81},
    )
