"""A measured day for tests that need DayOutcome records."""

from datetime import date, timedelta

from core.theria.plan.outcomes import DayOutcome

START = date(2026, 11, 1)


def day(offset, saving=2.0, aux=0.0, room_min=19.7, coverage=1.0):
    return DayOutcome(
        day=START + timedelta(days=offset),
        heating_hours=int(24 * coverage),
        coverage=coverage,
        compressor_kwh=40.0,
        aux_kwh=aux,
        actual_cost_sek=100.0,
        baseline_cost_sek=100.0 + saving,
        saving_sek=saving,
        average_price_paid_sek_per_kwh=1.0,
        average_price_baseline_sek_per_kwh=1.0,
        mean_rts_c=18.0,
        room_min_c=room_min,
        room_mean_c=19.9,
    )
