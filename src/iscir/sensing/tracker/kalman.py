"""Kalman filtering for one-dimensional radar target motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class KalmanFilterConfig:
    """Noise parameters for a range/radial-velocity Kalman filter."""

    acceleration_std_mps2: float = 1.5
    range_measurement_std_m: float = 0.15
    velocity_measurement_std_mps: float = 0.25

    def __post_init__(self) -> None:
        if self.acceleration_std_mps2 <= 0:
            raise ValueError("acceleration_std_mps2 must be positive")
        if self.range_measurement_std_m <= 0:
            raise ValueError("range_measurement_std_m must be positive")
        if self.velocity_measurement_std_mps <= 0:
            raise ValueError("velocity_measurement_std_mps must be positive")


class ConstantVelocityKalmanFilter:
    """Estimate target range and radial velocity with a linear Kalman filter.

    The state vector is ``[range_m, radial_velocity_mps]``. Both state
    components are directly observed by the Range-Doppler detector.
    """

    def __init__(
        self,
        range_m: float,
        radial_velocity_mps: float,
        *,
        config: KalmanFilterConfig | None = None,
        initial_range_variance_m2: float = 1.0,
        initial_velocity_variance_m2ps2: float = 1.0,
    ) -> None:
        if range_m < 0:
            raise ValueError("range_m cannot be negative")
        if initial_range_variance_m2 <= 0 or initial_velocity_variance_m2ps2 <= 0:
            raise ValueError("initial variances must be positive")

        self.config = config or KalmanFilterConfig()
        self._state = np.array([range_m, radial_velocity_mps], dtype=float)
        self._covariance = np.diag(
            [initial_range_variance_m2, initial_velocity_variance_m2ps2]
        ).astype(float)

    @property
    def state(self) -> np.ndarray:
        """Return a defensive copy of the current state estimate."""

        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Return a defensive copy of the current state covariance."""

        return self._covariance.copy()

    @property
    def range_m(self) -> float:
        return float(self._state[0])

    @property
    def radial_velocity_mps(self) -> float:
        return float(self._state[1])

    def predict(self, dt_s: float) -> np.ndarray:
        """Advance the estimate by ``dt_s`` using a constant-velocity model."""

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")

        transition = np.array([[1.0, dt_s], [0.0, 1.0]], dtype=float)
        acceleration_variance = self.config.acceleration_std_mps2**2
        process_noise = acceleration_variance * np.array(
            [
                [dt_s**4 / 4.0, dt_s**3 / 2.0],
                [dt_s**3 / 2.0, dt_s**2],
            ],
            dtype=float,
        )

        self._state = transition @ self._state
        self._covariance = (
            transition @ self._covariance @ transition.T + process_noise
        )
        return self.state

    def innovation(
        self, measured_range_m: float, measured_radial_velocity_mps: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return measurement residual and residual covariance."""

        measurement = self._measurement_vector(
            measured_range_m, measured_radial_velocity_mps
        )
        measurement_noise = self._measurement_noise()
        residual = measurement - self._state
        residual_covariance = self._covariance + measurement_noise
        return residual, residual_covariance

    def update(
        self, measured_range_m: float, measured_radial_velocity_mps: float
    ) -> np.ndarray:
        """Correct the predicted state using a Range-Doppler measurement."""

        residual, residual_covariance = self.innovation(
            measured_range_m, measured_radial_velocity_mps
        )
        kalman_gain = np.linalg.solve(
            residual_covariance.T, self._covariance.T
        ).T
        self._state = self._state + kalman_gain @ residual

        identity = np.eye(2, dtype=float)
        measurement_noise = self._measurement_noise()
        correction = identity - kalman_gain
        self._covariance = (
            correction @ self._covariance @ correction.T
            + kalman_gain @ measurement_noise @ kalman_gain.T
        )
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        return self.state

    def mahalanobis_distance_squared(
        self, measured_range_m: float, measured_radial_velocity_mps: float
    ) -> float:
        """Return the squared Mahalanobis distance for association gating."""

        residual, residual_covariance = self.innovation(
            measured_range_m, measured_radial_velocity_mps
        )
        solved = np.linalg.solve(residual_covariance, residual)
        return float(residual @ solved)

    @staticmethod
    def _measurement_vector(
        measured_range_m: float, measured_radial_velocity_mps: float
    ) -> np.ndarray:
        if measured_range_m < 0:
            raise ValueError("measured_range_m cannot be negative")
        return np.array(
            [measured_range_m, measured_radial_velocity_mps], dtype=float
        )

    def _measurement_noise(self) -> np.ndarray:
        return np.diag(
            [
                self.config.range_measurement_std_m**2,
                self.config.velocity_measurement_std_mps**2,
            ]
        )
