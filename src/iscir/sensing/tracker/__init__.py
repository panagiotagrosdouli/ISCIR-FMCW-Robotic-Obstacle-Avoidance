"""Multi-target tracking primitives for radar detections."""

from .kalman import ConstantVelocityKalmanFilter, KalmanFilterConfig

__all__ = ["ConstantVelocityKalmanFilter", "KalmanFilterConfig"]
