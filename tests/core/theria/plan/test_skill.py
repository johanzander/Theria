from datetime import UTC, datetime, timedelta

import pytest

from core.theria.plan.disturbance import Innovation
from core.theria.plan.skill import (
    BAD_C_PER_H,
    GOOD_C_PER_H,
    MIN_AUTHORITY,
    SkillTracker,
)

NOW = datetime(2026, 10, 14, 0, 0, tzinfo=UTC)


def fill(tracker, rate, hours=48, noise=0.0, dark=True):
    for k in range(hours * 4):
        wobble = noise if k % 2 == 0 else -noise  # quantisation: zero-mean
        tracker.add(Innovation(NOW + timedelta(minutes=15 * k), rate + wobble, dark))
    return NOW + timedelta(hours=hours)


def test_without_evidence_the_authority_is_the_minimum():
    tracker = SkillTracker()
    assert tracker.authority(NOW) == MIN_AUTHORITY
    assert tracker.error(NOW) is None


def test_a_model_that_predicts_well_earns_full_authority():
    tracker = SkillTracker()
    end = fill(tracker, 0.0, noise=0.3)  # noisy readings, no systematic error
    assert tracker.error(end) < GOOD_C_PER_H
    assert tracker.authority(end) == 1.0


def test_noise_that_averages_out_is_not_held_against_the_model():
    tracker = SkillTracker()
    end = fill(tracker, 0.0, noise=0.4)
    assert tracker.authority(end) == 1.0


def test_a_systematic_error_takes_the_authority_back_to_the_minimum():
    tracker = SkillTracker()
    end = fill(tracker, BAD_C_PER_H + 0.02)
    assert tracker.authority(end) == MIN_AUTHORITY


def test_in_between_the_authority_is_in_between():
    tracker = SkillTracker()
    end = fill(tracker, (GOOD_C_PER_H + BAD_C_PER_H) / 2)
    assert MIN_AUTHORITY < tracker.authority(end) < 1.0


def test_sunny_samples_are_not_evidence():
    tracker = SkillTracker()
    end = fill(tracker, 0.5, dark=False)
    assert tracker.error(end) is None and tracker.authority(end) == MIN_AUTHORITY


def test_old_evidence_is_forgotten():
    tracker = SkillTracker()
    fill(tracker, 0.0)
    later = NOW + timedelta(hours=48 + 200)
    assert tracker.error(later) is None


def test_the_evidence_survives_a_restart():
    tracker = SkillTracker()
    end = fill(tracker, 0.0, noise=0.2)
    again = SkillTracker()
    again.load_state(tracker.export_state())
    assert again.authority(end) == tracker.authority(end) == 1.0
