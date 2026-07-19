"""Data association between radar tracks and Range-Doppler detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .track import Track


@dataclass(frozen=True, slots=True)
class RadarDetection:
    """Range-Doppler detection used by the tracking layer."""

    range_m: float
    radial_velocity_mps: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.range_m < 0:
            raise ValueError("range_m cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """Result of assigning detections to tracks for one radar frame."""

    matches: tuple[tuple[int, int], ...]
    unmatched_track_indices: tuple[int, ...]
    unmatched_detection_indices: tuple[int, ...]


def mahalanobis_cost_matrix(
    tracks: Sequence[Track], detections: Sequence[RadarDetection]
) -> np.ndarray:
    """Return squared Mahalanobis distances for every track-detection pair."""

    costs = np.empty((len(tracks), len(detections)), dtype=float)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            costs[track_index, detection_index] = track.mahalanobis_distance_squared(
                detection.range_m,
                detection.radial_velocity_mps,
            )
    return costs


def associate_nearest_neighbor(
    tracks: Sequence[Track],
    detections: Sequence[RadarDetection],
    *,
    gate_threshold_squared: float = 9.21,
) -> AssociationResult:
    """Greedily assign the lowest-cost gated track-detection pairs.

    The default gate is the 99 percent chi-square threshold for a
    two-dimensional measurement. Each track and detection can be used at most
    once. The greedy implementation is deterministic and deliberately small;
    it can later be replaced by Hungarian assignment without changing the
    result interface.
    """

    if gate_threshold_squared <= 0:
        raise ValueError("gate_threshold_squared must be positive")

    track_count = len(tracks)
    detection_count = len(detections)
    if track_count == 0 or detection_count == 0:
        return AssociationResult(
            matches=(),
            unmatched_track_indices=tuple(range(track_count)),
            unmatched_detection_indices=tuple(range(detection_count)),
        )

    costs = mahalanobis_cost_matrix(tracks, detections)
    candidates = [
        (float(costs[track_index, detection_index]), track_index, detection_index)
        for track_index in range(track_count)
        for detection_index in range(detection_count)
        if np.isfinite(costs[track_index, detection_index])
        and costs[track_index, detection_index] <= gate_threshold_squared
    ]
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    assigned_tracks: set[int] = set()
    assigned_detections: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _, track_index, detection_index in candidates:
        if track_index in assigned_tracks or detection_index in assigned_detections:
            continue
        matches.append((track_index, detection_index))
        assigned_tracks.add(track_index)
        assigned_detections.add(detection_index)

    return AssociationResult(
        matches=tuple(matches),
        unmatched_track_indices=tuple(
            index for index in range(track_count) if index not in assigned_tracks
        ),
        unmatched_detection_indices=tuple(
            index for index in range(detection_count) if index not in assigned_detections
        ),
    )
