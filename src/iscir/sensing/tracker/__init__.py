"""Multi-target tracking primitives for radar detections."""

from .assignment import (
    AssociationResult,
    RadarDetection,
    associate_nearest_neighbor,
    mahalanobis_cost_matrix,
)
from .kalman import ConstantVelocityKalmanFilter, KalmanFilterConfig
from .track import Track, TrackLifecycleConfig, TrackStatus

__all__ = [
    "AssociationResult",
    "ConstantVelocityKalmanFilter",
    "KalmanFilterConfig",
    "RadarDetection",
    "Track",
    "TrackLifecycleConfig",
    "TrackStatus",
    "associate_nearest_neighbor",
    "mahalanobis_cost_matrix",
]
