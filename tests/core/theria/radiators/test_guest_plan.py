import math
from datetime import UTC, datetime, timedelta

from core.theria.radiators.guest_plan import plan_guest_room
from core.theria.radiators.settings import GuestRoom

START = datetime(2026, 11, 2, 5, tzinfo=UTC)
ROOM = GuestRoom("climate.hallway", 20.0)


def run(prices, room=ROOM, band=1.5, room_temp=None, **limits):
    hours = [START + timedelta(hours=k) for k in range(len(prices))]
    return plan_guest_room(hours, prices, room, band, room_temp, **limits)


def test_cheapest_hour_is_the_top_of_the_band_and_dearest_the_bottom():
    plan = run([0.5, 1.0, 2.0])
    assert plan[0].setpoint_c == 21.5
    assert plan[1].setpoint_c == 20.0
    assert plan[2].setpoint_c == 18.5


def test_setpoints_are_half_degree_steps():
    for h in run([0.3, 0.9, 0.4, 1.7, 2.2, 0.8, 1.1]):
        assert h.setpoint_c * 2 == int(h.setpoint_c * 2)


def test_equal_prices_give_t_room():
    assert {h.setpoint_c for h in run([1.0] * 6)} == {20.0}


def test_a_room_below_its_band_is_not_cut():
    plan = run([0.5, 1.0, 2.0], room_temp=18.0)
    assert plan[2].setpoint_c >= 20.0
    assert plan[0].setpoint_c == 21.5


def test_a_room_above_its_band_is_not_heated_further():
    plan = run([0.5, 1.0, 2.0], room_temp=22.0)
    assert plan[0].setpoint_c <= 20.0
    assert plan[2].setpoint_c == 18.5


def test_a_room_inside_its_band_follows_the_price():
    assert run([0.5, 2.0], room_temp=19.0)[1].setpoint_c == 18.5


def test_nan_price_gives_t_room():
    plan = run([0.5, math.nan, 2.0])
    assert plan[1].setpoint_c == 20.0
    assert "no price" in plan[1].reason


def test_limits_clamp():
    plan = run([0.5, 2.0], room=GuestRoom("climate.x", 24.5), band=1.5)
    assert max(h.setpoint_c for h in plan) == 25.0
    plan = run([0.5, 2.0], room=GuestRoom("climate.x", 5.5), band=1.5)
    assert min(h.setpoint_c for h in plan) == 5.0


def test_tied_prices_share_a_rank():
    plan = run([0.5, 1.0, 1.0, 2.0])
    assert plan[1].setpoint_c == plan[2].setpoint_c


def test_near_flat_prices_do_not_move_the_room():
    plan = run([0.50, 0.5001, 0.5002, 0.5003])
    assert {h.setpoint_c for h in plan} == {20.0}


def test_a_modest_spread_moves_the_room_less_than_the_whole_band():
    plan = run([0.50, 0.60])
    assert 20.0 < plan[0].setpoint_c < 21.5
