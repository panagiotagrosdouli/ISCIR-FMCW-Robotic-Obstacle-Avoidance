from __future__ import annotations

import numpy as np
import pytest

from iscir.sensing.tracker.assignment import (
    RadarDetection,
    associate_nearest_neighbor,
    mahalanobis_cost_matrix,
)
from iscir.sensing.tracker.kalman import ConstantVelocityKalmanFilter
from iscir.sensing.tracker.track import Track


def make_track(track_id: int, range_m: float, velocity_mps: float) -> Track:
    return Track(
        track_id,
        ConstantVelocityKalmanFilter(
            range_m=range_m,
            radial_velocity_mps=velocity_mps,
            initial_range_variance_m2=0.04,
            initial_velocity_variance_m2ps2=0.04,
        ),
    )


def test_cost_matrix_has_expected_shape_and_ordering() -> None:
    tracks = [make_track(1, 5.0, 1.0), make_track(2, 10.0, -1.0)]
    detections = [
        RadarDetection(5.05, 0.95),
        RadarDetection(9.9, -1.1),
    ]

    costs = mahalanobis_cost_matrix(tracks, detections)

    assert costs.shape == (2, 2)
    assert costs[0, 0] < costs[0, 1]
    assert costs[1, 1] < costs[1, 0]
    assert np.all(costs >= 0.0)


def test_association_matches_nearest_gated_detections() -> None:
    tracks = [make_track(1, 5.0, 1.0), make_track(2, 10.0, -1.0)]
    detections = [
        RadarDetection(10.05, -1.05),
        RadarDetection(5.05, 0.95),
        RadarDetection(30.0, 0.0),
    ]

    result = associate_nearest_neighbor(tracks, detections)

    assert set(result.matches) == {(0, 1), (1, 0)}
    assert result.unmatched_track_indices == ()
    assert result.unmatched_detection_indices == (2,)


def test_gate_rejects_implausible_detection() -> None:
    tracks = [make_track(1, 5.0, 0.0)]
    detections = [RadarDetection(20.0, 5.0)]

    result = associate_nearest_neighbor(tracks, detections)

    assert result.matches == ()
    assert result.unmatched_track_indices == (0,)
    assert result.unmatched_detection_indices == (0,)


def test_each_track_and_detection_is_assigned_at_most_once() -> None:
    tracks = [make_track(1, 5.0, 0.0), make_track(2, 5.2, 0.0)]
    detections = [RadarDetection(5.05, 0.0)]

    result = associate_nearest_neighbor(
        tracks,
        detections,
        gate_threshold_squared=100.0,
    )

    assert len(result.matches) == 1
    assert len(result.unmatched_track_indices) == 1
    assert result.unmatched_detection_indices == ()


def test_empty_inputs_are_reported_as_unmatched() -> None:
    detection = RadarDetection(5.0, 0.0)
    track = make_track(1, 5.0, 0.0)

    no_tracks = associate_nearest_neighbor([], [detection])
    no_detections = associate_nearest_neighbor([track], [])

    assert no_tracks.unmatched_detection_indices == (0,)
    assert no_detections.unmatched_track_indices == (0,)


def test_invalid_detection_and_gate_are_rejected() -> None:
    with pytest.raises(ValueError):
        RadarDetection(-1.0, 0.0)
    with pytest.raises(ValueError):
        RadarDetection(1.0, 0.0, confidence=1.1)
    with pytest.raises(ValueError):
        associate_nearest_neighbor([], [], gate_threshold_squared=0.0)
