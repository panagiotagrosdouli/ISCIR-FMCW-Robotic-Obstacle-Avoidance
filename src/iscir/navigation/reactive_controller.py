"""Risk-aware conversion of collision warnings into robot velocity commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .collision import CollisionLevel, CollisionPrediction


@dataclass(frozen=True, slots=True)
class ReactiveControllerConfig:
    """Configuration for conservative one-dimensional velocity control."""

    cruise_speed_mps: float = 1.0
    caution_speed_fraction: float = 0.65
    danger_speed_fraction: float = 0.25
    emergency_speed_mps: float = 0.0

    def __post_init__(self) -> None:
        if self.cruise_speed_mps < 0.0:
            raise ValueError("cruise_speed_mps cannot be negative")
        if self.emergency_speed_mps < 0.0:
            raise ValueError("emergency_speed_mps cannot be negative")
        if self.emergency_speed_mps > self.cruise_speed_mps:
            raise ValueError("emergency_speed_mps cannot exceed cruise_speed_mps")
        if not 0.0 <= self.danger_speed_fraction <= 1.0:
            raise ValueError("danger_speed_fraction must be between 0 and 1")
        if not 0.0 <= self.caution_speed_fraction <= 1.0:
            raise ValueError("caution_speed_fraction must be between 0 and 1")
        if self.danger_speed_fraction > self.caution_speed_fraction:
            raise ValueError(
                "danger_speed_fraction cannot exceed caution_speed_fraction"
            )


@dataclass(frozen=True, slots=True)
class VelocityCommand:
    """Longitudinal robot command generated from the most urgent warning."""

    linear_speed_mps: float
    level: CollisionLevel
    source_track_id: int | None
    risk_score: float
    emergency_stop: bool


def safe_velocity_command(
    predictions: Iterable[CollisionPrediction],
    config: ReactiveControllerConfig | None = None,
) -> VelocityCommand:
    """Generate a safe forward-speed command from collision predictions.

    The most conservative command across all tracks is selected. Within caution
    and danger levels, the speed is reduced continuously as normalized risk rises.
    Emergency predictions always produce the configured emergency speed.
    """

    settings = config or ReactiveControllerConfig()
    predictions_tuple = tuple(predictions)

    if not predictions_tuple:
        return VelocityCommand(
            linear_speed_mps=settings.cruise_speed_mps,
            level=CollisionLevel.CLEAR,
            source_track_id=None,
            risk_score=0.0,
            emergency_stop=False,
        )

    candidates = [
        _command_for_prediction(prediction, settings)
        for prediction in predictions_tuple
    ]
    return min(
        candidates,
        key=lambda command: (
            command.linear_speed_mps,
            -command.risk_score,
            command.source_track_id if command.source_track_id is not None else -1,
        ),
    )


def _command_for_prediction(
    prediction: CollisionPrediction,
    config: ReactiveControllerConfig,
) -> VelocityCommand:
    score = min(1.0, max(0.0, float(prediction.risk_score)))

    if prediction.level is CollisionLevel.EMERGENCY:
        speed = config.emergency_speed_mps
    elif prediction.level is CollisionLevel.DANGER:
        level_cap = config.cruise_speed_mps * config.danger_speed_fraction
        speed = level_cap * (1.0 - score)
    elif prediction.level is CollisionLevel.CAUTION:
        level_cap = config.cruise_speed_mps * config.caution_speed_fraction
        speed = level_cap + (config.cruise_speed_mps - level_cap) * (1.0 - score)
    else:
        speed = config.cruise_speed_mps

    speed = min(config.cruise_speed_mps, max(config.emergency_speed_mps, speed))
    emergency_stop = (
        prediction.level is CollisionLevel.EMERGENCY
        and speed <= config.emergency_speed_mps
    )

    return VelocityCommand(
        linear_speed_mps=speed,
        level=prediction.level,
        source_track_id=prediction.track_id,
        risk_score=score,
        emergency_stop=emergency_stop,
    )
