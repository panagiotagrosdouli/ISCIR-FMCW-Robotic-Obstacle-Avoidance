from __future__ import annotations

import numpy as np
import pytest

from iscir.sensing.tracker.kalman import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
)


def test_predict_uses_constant_velocity_model() -> None:
    tracker = ConstantVelocityKalmanFilter(range_m=10.0, radial_velocity_mps=-2.0)

    state = tracker.predict(0.5)

    assert state[0] == pytest.approx(9.0)
    assert state[1] == pytest.approx(-2.0)


def test_update_moves_prediction_toward_measurement() -> None:
    tracker = ConstantVelocityKalmanFilter(range_m=10.0, radial_velocity_mps=0.0)
    tracker.predict(1.0)
    before = tracker.state

    measured_range_m = 8.0
    measured_velocity_mps = -1.5
    after = tracker.update(measured_range_m, measured_velocity_mps)

    assert abs(after[0] - measured_range_m) < abs(before[0] - measured_range_m)
    assert abs(after[1] - measured_velocity_mps) < abs(
        before[1] - measured_velocity_mps
    )
    assert np.all(np.diag(tracker.covariance) > 0.0)
    assert np.all(np.diag(tracker.covariance) < np.diag(np.eye(2) * 3.0))


def test_repeated_measurements_converge_to_motion() -> None:
    tracker = ConstantVelocityKalmanFilter(range_m=12.0, radial_velocity_mps=-0.5)

    dt_s = 0.1
    true_range_m = 12.0
    true_velocity_mps = -1.0
    for _ in range(40):
        true_range_m += true_velocity_mps * dt_s
        tracker.predict(dt_s)
        tracker.update(true_range_m, true_velocity_mps)

    assert tracker.range_m == pytest.approx(true_range_m, abs=0.03)
    assert tracker.radial_velocity_mps == pytest.approx(true_velocity_mps, abs=0.03)


def test_mahalanobis_distance_supports_gating() -> None:
    config = KalmanFilterConfig(
        range_measurement_std_m=0.1,
        velocity_measurement_std_mps=0.1,
    )
    tracker = ConstantVelocityKalmanFilter(
        range_m=5.0,
        radial_velocity_mps=1.0,
        config=config,
        initial_range_variance_m2=0.01,
        initial_velocity_variance_m2ps2=0.01,
    )

    near = tracker.mahalanobis_distance_squared(5.05, 0.95)
    far = tracker.mahalanobis_distance_squared(8.0, -2.0)

    assert near < far
    assert near < 9.21
    assert far > 9.21


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        KalmanFilterConfig(acceleration_std_mps2=0.0)
    with pytest.raises(ValueError):
        ConstantVelocityKalmanFilter(range_m=-1.0, radial_velocity_mps=0.0)

    tracker = ConstantVelocityKalmanFilter(range_m=1.0, radial_velocity_mps=0.0)
    with pytest.raises(ValueError):
        tracker.predict(0.0)
    with pytest.raises(ValueError):
        tracker.update(-1.0, 0.0)
