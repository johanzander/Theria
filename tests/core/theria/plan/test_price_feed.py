from datetime import UTC, datetime, timedelta

import pytest

from core.theria.exceptions import PriceDataError
from core.theria.plan.price_feed import NordpoolFeed, parse_prices_for_date

NOW = datetime(2026, 10, 14, 10, 0, tzinfo=UTC)  # 12:00 local


def answer(day_start, price_per_mwh=50.0, quarters=96):
    return {
        "SE4": [
            {
                "start": (day_start + timedelta(minutes=15 * i)).isoformat(),
                "end": (day_start + timedelta(minutes=15 * (i + 1))).isoformat(),
                "price": price_per_mwh + i,
            }
            for i in range(quarters)
        ]
    }


class FakeNordpool:
    def __init__(self, tomorrow_published=False):
        self.calls = []
        self.tomorrow_published = tomorrow_published

    def get_service_response(self, domain, service, data):
        self.calls.append((domain, service, data))
        if data["date"] == "2026-10-14":
            return answer(datetime(2026, 10, 13, 22, tzinfo=UTC))
        if self.tomorrow_published and data["date"] == "2026-10-15":
            return answer(datetime(2026, 10, 14, 22, tzinfo=UTC))
        raise RuntimeError("connection error")


def feed(client, clock=lambda: NOW):
    return NordpoolFeed(client, "entry", "SE4", "Europe/Stockholm", clock)


def test_prices_are_converted_from_sek_per_mwh_to_sek_per_kwh():
    periods = parse_prices_for_date(answer(NOW, 55.78), "SE4")
    assert periods[0].price_sek_per_kwh == pytest.approx(0.05578)


def test_an_answer_without_the_area_is_a_price_error():
    with pytest.raises(PriceDataError):
        parse_prices_for_date({"SE3": []}, "SE4")


def test_today_is_read_and_a_missing_tomorrow_is_not_an_error():
    periods = feed(FakeNordpool()).periods()
    assert len(periods) == 96


def test_tomorrow_is_added_once_it_is_published_and_asked_for_again_after_a_wait():
    client = FakeNordpool()
    now = [NOW]
    prices = feed(client, lambda: now[0])
    assert len(prices.periods()) == 96
    asked = len(client.calls)
    prices.periods()
    assert len(client.calls) == asked  # not asked again at once
    client.tomorrow_published = True
    now[0] += timedelta(minutes=11)
    assert len(prices.periods()) == 192


def test_a_day_that_was_read_is_not_read_again():
    client = FakeNordpool(tomorrow_published=True)
    prices = feed(client)
    prices.periods()
    calls = len(client.calls)
    prices.periods()
    assert len(client.calls) == calls


def test_the_local_date_not_the_utc_date_is_asked_for():
    client = FakeNordpool()
    late = datetime(2026, 10, 13, 23, 30, tzinfo=UTC)  # 01:30 local on the 14th
    feed(client, lambda: late).periods()
    assert client.calls[0][2]["date"] == "2026-10-14"


def test_when_today_cannot_be_read_the_plan_gets_a_price_error():
    class Down:
        def get_service_response(self, domain, service, data):
            raise RuntimeError("connection error")

    with pytest.raises(PriceDataError):
        feed(Down()).periods()
