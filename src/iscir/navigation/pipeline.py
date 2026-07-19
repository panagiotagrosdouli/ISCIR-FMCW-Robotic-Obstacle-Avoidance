"""End-to-end radar tracking and risk-aware navigation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from iscir.sensing.tracker import MultiTargetTracker, RadarDetection, Track

from .collision import CollisionConfig, CollisionPrediction, predict_collisions
from .reactive_controller import (
    ReactiveControllerConfig,
    VelocityCommand,
    safe_velocity_command,
)
from .risk import RiskConfig, TrackRisk, rank_track_risks


@dataclass(frozen=True, slots=True)
class NavigationFrameResult:
    """Complete output produced for one radar frame."""

    tracks: tuple[Track, ...]
    risks: tuple[TrackRisk, ...]
    collisions: tuple[CollisionPrediction, ...]
    command: VelocityCommand


class RadarNavigationPipeline:
    """Convert radar detections into a safe longitudinal velocity command."""

    def __init__(
        self,
        *,
        tracker: MultiTargetTracker | None = None,
        risk_config: RiskConfig | None = None,
        collision_config: CollisionConfig | None = None,
        controller_config: ReactiveControllerConfig | None = None,
        confirmed_tracks_only: bool = True,
    ) -> None:
        self.tracker = tracker or MultiTargetTracker()
        self.risk_config = risk_config or RiskConfig()
        self.collision_config = collision_config or CollisionConfig()
        self.controller_config = controller_config or ReactiveControllerConfig()
        self.confirmed_tracks_only = bool(confirmed_tracks_only)

    def update(
        self,
        detections: Sequence[RadarDetection],
        dt_s: float,
    ) -> NavigationFrameResult:
        """Process one frame from detections through tracking and control."""

        tracks = self.tracker.update(detections, dt_s)
        navigation_tracks = (
            self.tracker.confirmed_tracks if self.confirmed_tracks_only else tracks
        )
        risks = rank_track_risks(navigation_tracks, self.risk_config)
        collisions = predict_collisions(risks, self.collision_config)
        command = safe_velocity_command(collisions, self.controller_config)

        return NavigationFrameResult(
            tracks=tracks,
            risks=risks,
            collisions=collisions,
            command=command,
        )
