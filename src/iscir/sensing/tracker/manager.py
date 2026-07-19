"""Frame-by-frame orchestration for persistent radar target tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .assignment import RadarDetection, associate_nearest_neighbor
from .kalman import ConstantVelocityKalmanFilter, KalmanFilterConfig
from .track import Track, TrackLifecycleConfig


@dataclass(frozen=True, slots=True)
class TrackerManagerConfig:
    """Configuration for multi-target track creation and association."""

    gate_threshold_squared: float = 9.21
    initial_range_variance_m2: float = 1.0
    initial_velocity_variance_m2ps2: float = 1.0

    def __post_init__(self) -> None:
        if self.gate_threshold_squared <= 0:
            raise ValueError("gate_threshold_squared must be positive")
        if self.initial_range_variance_m2 <= 0:
            raise ValueError("initial_range_variance_m2 must be positive")
        if self.initial_velocity_variance_m2ps2 <= 0:
            raise ValueError("initial_velocity_variance_m2ps2 must be positive")


class MultiTargetTracker:
    """Maintain persistent one-dimensional radar tracks across frames."""

    def __init__(
        self,
        *,
        manager_config: TrackerManagerConfig | None = None,
        filter_config: KalmanFilterConfig | None = None,
        lifecycle_config: TrackLifecycleConfig | None = None,
        first_track_id: int = 0,
    ) -> None:
        if first_track_id < 0:
            raise ValueError("first_track_id cannot be negative")

        self.manager_config = manager_config or TrackerManagerConfig()
        self.filter_config = filter_config or KalmanFilterConfig()
        self.lifecycle_config = lifecycle_config or TrackLifecycleConfig()
        self._tracks: list[Track] = []
        self._next_track_id = first_track_id

    @property
    def tracks(self) -> tuple[Track, ...]:
        """Return all active tracks ordered by persistent track ID."""

        return tuple(self._tracks)

    @property
    def confirmed_tracks(self) -> tuple[Track, ...]:
        """Return active tracks that have passed lifecycle confirmation."""

        return tuple(track for track in self._tracks if track.is_confirmed)

    def update(
        self, detections: Sequence[RadarDetection], dt_s: float
    ) -> tuple[Track, ...]:
        """Process one radar frame and return the resulting active tracks."""

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")

        for track in self._tracks:
            track.predict(dt_s)

        association = associate_nearest_neighbor(
            self._tracks,
            detections,
            gate_threshold_squared=self.manager_config.gate_threshold_squared,
        )

        for track_index, detection_index in association.matches:
            detection = detections[detection_index]
            self._tracks[track_index].update(
                detection.range_m,
                detection.radial_velocity_mps,
                detection_confidence=detection.confidence,
            )

        for track_index in association.unmatched_track_indices:
            self._tracks[track_index].mark_missed()

        self._tracks = [track for track in self._tracks if not track.is_deleted]

        for detection_index in association.unmatched_detection_indices:
            self._tracks.append(self._new_track(detections[detection_index]))

        self._tracks.sort(key=lambda track: track.track_id)
        return self.tracks

    def _new_track(self, detection: RadarDetection) -> Track:
        state_filter = ConstantVelocityKalmanFilter(
            detection.range_m,
            detection.radial_velocity_mps,
            config=self.filter_config,
            initial_range_variance_m2=self.manager_config.initial_range_variance_m2,
            initial_velocity_variance_m2ps2=(
                self.manager_config.initial_velocity_variance_m2ps2
            ),
        )
        track = Track(
            self._next_track_id,
            state_filter,
            lifecycle=self.lifecycle_config,
            initial_confidence=detection.confidence,
        )
        self._next_track_id += 1
        return track
