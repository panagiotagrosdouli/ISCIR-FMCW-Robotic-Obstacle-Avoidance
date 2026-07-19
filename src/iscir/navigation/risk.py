"""Collision-risk estimation from persistent radar tracks."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable

from iscir.sensing.tracker import Track


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Parameters controlling distance and time-to-collision risk."""

    safety_distance_m: float = 1.0
    distance_horizon_m: float = 10.0
    ttc_horizon_s: float = 5.0
    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.safety_distance_m < 0:
            raise ValueError("safety_distance_m cannot be negative")
        if self.distance_horizon_m <= self.safety_distance_m:
            raise ValueError("distance_horizon_m must exceed safety_distance_m")
        if self.ttc_horizon_s <= 0:
            raise ValueError("ttc_horizon_s must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TrackRisk:
    """Risk estimate associated with one persistent track."""

    track_id: int
    range_m: float
    radial_velocity_mps: float
    closing_speed_mps: float
    time_to_collision_s: float
    risk_score: float
    confidence: float

    @property
    def is_approaching(self) -> bool:
        return self.closing_speed_mps > 0.0



def estimate_track_risk(
    track: Track,
    config: RiskConfig | None = None,
) -> TrackRisk:
    """Estimate normalized collision risk for one track.

    Negative radial velocity denotes an approaching object. The score combines
    proximity and time-to-collision, then scales the result by track confidence.
    """

    settings = config or RiskConfig()
    range_m = max(0.0, float(track.range_m))
    velocity_mps = float(track.radial_velocity_mps)
    confidence = float(track.confidence)
    closing_speed_mps = max(0.0, -velocity_mps)

    if closing_speed_mps > 0.0:
        remaining_distance_m = max(0.0, range_m - settings.safety_distance_m)
        time_to_collision_s = remaining_distance_m / closing_speed_mps
    else:
        time_to_collision_s = inf

    distance_span_m = settings.distance_horizon_m - settings.safety_distance_m
    distance_risk = 1.0 - (range_m - settings.safety_distance_m) / distance_span_m
    distance_risk = min(1.0, max(0.0, distance_risk))

    if time_to_collision_s == inf:
        ttc_risk = 0.0
    else:
        ttc_risk = 1.0 - time_to_collision_s / settings.ttc_horizon_s
        ttc_risk = min(1.0, max(0.0, ttc_risk))

    if confidence < settings.minimum_confidence:
        risk_score = 0.0
    else:
        risk_score = confidence * max(distance_risk, ttc_risk)
        risk_score = min(1.0, max(0.0, risk_score))

    return TrackRisk(
        track_id=track.track_id,
        range_m=range_m,
        radial_velocity_mps=velocity_mps,
        closing_speed_mps=closing_speed_mps,
        time_to_collision_s=time_to_collision_s,
        risk_score=risk_score,
        confidence=confidence,
    )



def rank_track_risks(
    tracks: Iterable[Track],
    config: RiskConfig | None = None,
) -> tuple[TrackRisk, ...]:
    """Return track risks ordered from highest to lowest risk."""

    risks = [estimate_track_risk(track, config) for track in tracks]
    risks.sort(key=lambda risk: (-risk.risk_score, risk.time_to_collision_s, risk.track_id))
    return tuple(risks)
