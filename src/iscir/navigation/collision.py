"""Collision prediction from normalized radar-track risk estimates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable

from .risk import TrackRisk


class CollisionLevel(str, Enum):
    """Discrete collision-warning level used by navigation controllers."""

    CLEAR = "clear"
    CAUTION = "caution"
    DANGER = "danger"
    EMERGENCY = "emergency"


@dataclass(frozen=True, slots=True)
class CollisionConfig:
    """Thresholds used to convert continuous risk into collision warnings."""

    caution_risk_threshold: float = 0.25
    danger_risk_threshold: float = 0.55
    emergency_risk_threshold: float = 0.80
    emergency_ttc_s: float = 1.0

    def __post_init__(self) -> None:
        thresholds = (
            self.caution_risk_threshold,
            self.danger_risk_threshold,
            self.emergency_risk_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("risk thresholds must be between 0 and 1")
        if not (
            self.caution_risk_threshold
            < self.danger_risk_threshold
            < self.emergency_risk_threshold
        ):
            raise ValueError("risk thresholds must be strictly increasing")
        if self.emergency_ttc_s <= 0.0:
            raise ValueError("emergency_ttc_s must be positive")


@dataclass(frozen=True, slots=True)
class CollisionPrediction:
    """Collision warning associated with one persistent radar track."""

    track_id: int
    level: CollisionLevel
    collision_predicted: bool
    risk_score: float
    time_to_collision_s: float
    range_m: float
    closing_speed_mps: float



def predict_collision(
    risk: TrackRisk,
    config: CollisionConfig | None = None,
) -> CollisionPrediction:
    """Classify one track risk into a collision-warning level."""

    settings = config or CollisionConfig()
    score = min(1.0, max(0.0, float(risk.risk_score)))
    finite_ttc = isfinite(risk.time_to_collision_s)

    emergency_from_ttc = (
        risk.is_approaching
        and finite_ttc
        and risk.time_to_collision_s <= settings.emergency_ttc_s
    )

    if emergency_from_ttc or score >= settings.emergency_risk_threshold:
        level = CollisionLevel.EMERGENCY
    elif score >= settings.danger_risk_threshold:
        level = CollisionLevel.DANGER
    elif score >= settings.caution_risk_threshold:
        level = CollisionLevel.CAUTION
    else:
        level = CollisionLevel.CLEAR

    return CollisionPrediction(
        track_id=risk.track_id,
        level=level,
        collision_predicted=level in {CollisionLevel.DANGER, CollisionLevel.EMERGENCY},
        risk_score=score,
        time_to_collision_s=risk.time_to_collision_s,
        range_m=risk.range_m,
        closing_speed_mps=risk.closing_speed_mps,
    )



def predict_collisions(
    risks: Iterable[TrackRisk],
    config: CollisionConfig | None = None,
) -> tuple[CollisionPrediction, ...]:
    """Predict collisions and order warnings from most to least urgent."""

    predictions = [predict_collision(risk, config) for risk in risks]
    priority = {
        CollisionLevel.EMERGENCY: 0,
        CollisionLevel.DANGER: 1,
        CollisionLevel.CAUTION: 2,
        CollisionLevel.CLEAR: 3,
    }
    predictions.sort(
        key=lambda item: (
            priority[item.level],
            item.time_to_collision_s,
            -item.risk_score,
            item.track_id,
        )
    )
    return tuple(predictions)
