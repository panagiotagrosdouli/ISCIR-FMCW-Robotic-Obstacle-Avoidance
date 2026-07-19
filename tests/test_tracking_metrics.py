from math import isclose

import pytest

from iscir.evaluation import (
    TrackEstimate,
    TrackingMetricsAccumulator,
    TruthTarget,
    evaluate_tracking_sequence,
)


def test_tracking_metrics_perfect_sequence() -> None:
    truths = [
        [TruthTarget(1, 10.0, -1.0)],
        [TruthTarget(1, 9.0, -1.0)],
    ]
    estimates = [
        [TrackEstimate(7, 10.0, -1.0)],
        [TrackEstimate(7, 9.0, -1.0)],
    ]

    metrics = evaluate_tracking_sequence(truths, estimates)

    assert metrics.truth_count == 2
    assert metrics.matched_count == 2
    assert metrics.missed_count == 0
    assert metrics.false_track_count == 0
    assert metrics.id_switches == 0
    assert metrics.range_rmse_m == 0.0
    assert metrics.velocity_rmse_mps == 0.0
    assert metrics.track_continuity == 1.0
    assert metrics.detection_probability == 1.0


def test_tracking_metrics_count_errors_and_identity_switch() -> None:
    accumulator = TrackingMetricsAccumulator(
        maximum_range_error_m=1.0,
        maximum_velocity_error_mps=1.0,
    )
    accumulator.update(
        [TruthTarget(3, 5.0, -0.5)],
        [TrackEstimate(10, 5.5, -0.25), TrackEstimate(99, 20.0, 0.0)],
    )
    accumulator.update(
        [TruthTarget(3, 4.5, -0.5), TruthTarget(4, 8.0, 0.0)],
        [TrackEstimate(11, 4.0, -0.75)],
    )

    metrics = accumulator.result()

    assert metrics.truth_count == 3
    assert metrics.matched_count == 2
    assert metrics.missed_count == 1
    assert metrics.false_track_count == 1
    assert metrics.id_switches == 1
    assert isclose(metrics.range_rmse_m, 0.5)
    assert isclose(metrics.velocity_rmse_mps, 0.25)
    assert isclose(metrics.track_continuity, 2.0 / 3.0)


def test_tracking_metrics_reject_unequal_sequence_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        evaluate_tracking_sequence([[]], [[], []])


@pytest.mark.parametrize(
    "keyword",
    [
        {"maximum_range_error_m": 0.0},
        {"maximum_velocity_error_mps": -1.0},
    ],
)
def test_tracking_metrics_validate_gates(keyword: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TrackingMetricsAccumulator(**keyword)
