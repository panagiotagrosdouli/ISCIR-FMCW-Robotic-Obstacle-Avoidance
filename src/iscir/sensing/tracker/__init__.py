"""Multi-target tracking primitives for radar detections."""

from .assignment import (
    AssociationResult,
    RadarDetection,
    associate_nearest_neighbor,
    mahalanobis_cost_matrix,
)
from .kalman import ConstantVelocityKalmanFilter, KalmanFilterConfig
from .manager import MultiTargetTracker, TrackerManagerConfig
from .track import Track, TrackLifecycleConfig, TrackStatus

__all__ = [
    "AssociationResult",
    "ConstantVelocityKalmanFilter",
    "KalmanFilterConfig",
    "MultiTargetTracker",
    "RadarDetection",
    "Track",
    "TrackLifecycleConfig",
    "TrackStatus",
    "TrackerManagerConfig",
    "associate_nearest_neighbor",
    "mahalanobis_cost_matrix",
]
