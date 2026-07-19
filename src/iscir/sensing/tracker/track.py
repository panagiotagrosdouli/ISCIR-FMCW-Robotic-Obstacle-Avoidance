"""Lifecycle state for a persistent radar target track."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .kalman import ConstantVelocityKalmanFilter


class TrackStatus(str, Enum):
    """Lifecycle status of a radar track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class TrackLifecycleConfig:
    """Rules controlling confirmation and deletion of tracks."""

    confirmation_hits: int = 3
    maximum_missed_updates: int = 3

    def __post_init__(self) -> None:
        if self.confirmation_hits < 1:
            raise ValueError("confirmation_hits must be positive")
        if self.maximum_missed_updates < 0:
            raise ValueError("maximum_missed_updates cannot be negative")


class Track:
    """Persistent identity and lifecycle around a Kalman state estimate."""

    def __init__(
        self,
        track_id: int,
        state_filter: ConstantVelocityKalmanFilter,
        *,
        lifecycle: TrackLifecycleConfig | None = None,
        initial_confidence: float = 1.0,
    ) -> None:
        if track_id < 0:
            raise ValueError("track_id cannot be negative")
        if not 0.0 <= initial_confidence <= 1.0:
            raise ValueError("initial_confidence must be between 0 and 1")

        self.track_id = track_id
        self.filter = state_filter
        self.lifecycle = lifecycle or TrackLifecycleConfig()
        self.age = 1
        self.hits = 1
        self.missed_updates = 0
        self.confidence = float(initial_confidence)
        self.status = (
            TrackStatus.CONFIRMED
            if self.lifecycle.confirmation_hits <= 1
            else TrackStatus.TENTATIVE
        )

    @property
    def range_m(self) -> float:
        return self.filter.range_m

    @property
    def radial_velocity_mps(self) -> float:
        return self.filter.radial_velocity_mps

    @property
    def state(self) -> np.ndarray:
        return self.filter.state

    @property
    def covariance(self) -> np.ndarray:
        return self.filter.covariance

    @property
    def is_confirmed(self) -> bool:
        return self.status is TrackStatus.CONFIRMED

    @property
    def is_deleted(self) -> bool:
        return self.status is TrackStatus.DELETED

    def predict(self, dt_s: float) -> np.ndarray:
        """Advance the state estimate and increment the track age."""

        if self.is_deleted:
            raise RuntimeError("cannot predict a deleted track")
        state = self.filter.predict(dt_s)
        self.age += 1
        return state

    def update(
        self,
        measured_range_m: float,
        measured_radial_velocity_mps: float,
        *,
        detection_confidence: float = 1.0,
    ) -> np.ndarray:
        """Correct the state and record a successful association."""

        if self.is_deleted:
            raise RuntimeError("cannot update a deleted track")
        if not 0.0 <= detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be between 0 and 1")

        state = self.filter.update(measured_range_m, measured_radial_velocity_mps)
        self.hits += 1
        self.missed_updates = 0
        self.confidence = 0.7 * self.confidence + 0.3 * detection_confidence
        if self.hits >= self.lifecycle.confirmation_hits:
            self.status = TrackStatus.CONFIRMED
        return state

    def mark_missed(self) -> None:
        """Record an unmatched frame and delete stale tracks when required."""

        if self.is_deleted:
            return
        self.missed_updates += 1
        self.confidence *= 0.8
        if self.missed_updates > self.lifecycle.maximum_missed_updates:
            self.status = TrackStatus.DELETED

    def mahalanobis_distance_squared(
        self, measured_range_m: float, measured_radial_velocity_mps: float
    ) -> float:
        """Return the association distance from this track to a detection."""

        if self.is_deleted:
            return float("inf")
        return self.filter.mahalanobis_distance_squared(
            measured_range_m, measured_radial_velocity_mps
        )
