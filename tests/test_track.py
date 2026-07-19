"""Tests for persistent radar track lifecycle state."""

from __future__ import annotations

import numpy as np
import pytest

from iscir.sensing.tracker import ConstantVelocityKalmanFilter
from iscir.sensing.tracker.track import Track, TrackLifecycleConfig, TrackStatus


def make_track(*, confirmation_hits: int = 3, maximum_missed_updates: int = 2) -> Track:
    return Track(
        7,
        ConstantVelocityKalmanFilter(5.0, 1.0),
        lifecycle=TrackLifecycleConfig(
            confirmation_hits=confirmation_hits,
            maximum_missed_updates=maximum_missed_updates,
        ),
    )


def test_track_starts_tentative_and_exposes_filter_state() -> None:
    track = make_track()

    assert track.track_id == 7
    assert track.status is TrackStatus.TENTATIVE
    assert not track.is_confirmed
    assert track.age == 1
    assert track.hits == 1
    assert track.missed_updates == 0
    assert np.allclose(track.state, [5.0, 1.0])


def test_predict_advances_range_and_age() -> None:
    track = make_track()

    predicted = track.predict(0.5)

    assert predicted[0] == pytest.approx(5.5)
    assert track.range_m == pytest.approx(5.5)
    assert track.age == 2
    assert track.hits == 1


def test_successful_updates_confirm_track_and_reset_misses() -> None:
    track = make_track(confirmation_hits=3)
    track.mark_missed()

    track.update(5.1, 1.0, detection_confidence=0.8)
    assert track.status is TrackStatus.TENTATIVE
    assert track.hits == 2
    assert track.missed_updates == 0

    track.update(5.2, 1.0)
    assert track.status is TrackStatus.CONFIRMED
    assert track.is_confirmed
    assert track.hits == 3


def test_track_is_deleted_after_miss_budget_is_exceeded() -> None:
    track = make_track(maximum_missed_updates=2)

    track.mark_missed()
    track.mark_missed()
    assert not track.is_deleted

    track.mark_missed()
    assert track.status is TrackStatus.DELETED
    assert track.is_deleted
    assert track.mahalanobis_distance_squared(5.0, 1.0) == float("inf")


def test_deleted_track_rejects_prediction_and_update() -> None:
    track = make_track(maximum_missed_updates=0)
    track.mark_missed()

    with pytest.raises(RuntimeError, match="deleted track"):
        track.predict(0.1)
    with pytest.raises(RuntimeError, match="deleted track"):
        track.update(5.0, 1.0)


def test_lifecycle_and_confidence_validation() -> None:
    with pytest.raises(ValueError, match="confirmation_hits"):
        TrackLifecycleConfig(confirmation_hits=0)
    with pytest.raises(ValueError, match="maximum_missed_updates"):
        TrackLifecycleConfig(maximum_missed_updates=-1)
    with pytest.raises(ValueError, match="track_id"):
        Track(-1, ConstantVelocityKalmanFilter(1.0, 0.0))
    with pytest.raises(ValueError, match="initial_confidence"):
        Track(1, ConstantVelocityKalmanFilter(1.0, 0.0), initial_confidence=1.1)
