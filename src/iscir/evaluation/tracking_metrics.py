"""Sequence-level metrics for persistent radar target tracking."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class TruthTarget:
    """Ground-truth target state for one frame."""

    object_id: int
    range_m: float
    radial_velocity_mps: float


@dataclass(frozen=True, slots=True)
class TrackEstimate:
    """Estimated persistent track state for one frame."""

    track_id: int
    range_m: float
    radial_velocity_mps: float


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    """Aggregate accuracy and identity metrics over a sequence."""

    truth_count: int
    matched_count: int
    missed_count: int
    false_track_count: int
    id_switches: int
    range_rmse_m: float
    velocity_rmse_mps: float
    track_continuity: float

    @property
    def detection_probability(self) -> float:
        return self.matched_count / self.truth_count if self.truth_count else 0.0


class TrackingMetricsAccumulator:
    """Accumulate deterministic nearest-neighbour tracking metrics by frame."""

    def __init__(
        self,
        *,
        maximum_range_error_m: float = 2.0,
        maximum_velocity_error_mps: float = 2.0,
    ) -> None:
        if maximum_range_error_m <= 0.0:
            raise ValueError("maximum_range_error_m must be positive")
        if maximum_velocity_error_mps <= 0.0:
            raise ValueError("maximum_velocity_error_mps must be positive")
        self.maximum_range_error_m = maximum_range_error_m
        self.maximum_velocity_error_mps = maximum_velocity_error_mps
        self._truth_count = 0
        self._matched_count = 0
        self._false_track_count = 0
        self._id_switches = 0
        self._range_squared_error = 0.0
        self._velocity_squared_error = 0.0
        self._last_track_by_object: dict[int, int] = {}

    def update(
        self,
        truths: Sequence[TruthTarget],
        estimates: Sequence[TrackEstimate],
    ) -> None:
        """Associate one frame and update aggregate counters."""

        self._truth_count += len(truths)
        candidates: list[tuple[float, int, int]] = []
        for truth_index, truth in enumerate(truths):
            for estimate_index, estimate in enumerate(estimates):
                range_error = estimate.range_m - truth.range_m
                velocity_error = estimate.radial_velocity_mps - truth.radial_velocity_mps
                if abs(range_error) > self.maximum_range_error_m:
                    continue
                if abs(velocity_error) > self.maximum_velocity_error_mps:
                    continue
                normalized_cost = (
                    (range_error / self.maximum_range_error_m) ** 2
                    + (velocity_error / self.maximum_velocity_error_mps) ** 2
                )
                candidates.append((normalized_cost, truth_index, estimate_index))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        used_truths: set[int] = set()
        used_estimates: set[int] = set()
        for _, truth_index, estimate_index in candidates:
            if truth_index in used_truths or estimate_index in used_estimates:
                continue
            used_truths.add(truth_index)
            used_estimates.add(estimate_index)
            truth = truths[truth_index]
            estimate = estimates[estimate_index]
            self._matched_count += 1
            self._range_squared_error += (estimate.range_m - truth.range_m) ** 2
            self._velocity_squared_error += (
                estimate.radial_velocity_mps - truth.radial_velocity_mps
            ) ** 2
            previous_track_id = self._last_track_by_object.get(truth.object_id)
            if previous_track_id is not None and previous_track_id != estimate.track_id:
                self._id_switches += 1
            self._last_track_by_object[truth.object_id] = estimate.track_id

        self._false_track_count += len(estimates) - len(used_estimates)

    def result(self) -> TrackingMetrics:
        """Return immutable aggregate metrics collected so far."""

        missed_count = self._truth_count - self._matched_count
        if self._matched_count:
            range_rmse = sqrt(self._range_squared_error / self._matched_count)
            velocity_rmse = sqrt(self._velocity_squared_error / self._matched_count)
        else:
            range_rmse = 0.0
            velocity_rmse = 0.0
        continuity = self._matched_count / self._truth_count if self._truth_count else 0.0
        return TrackingMetrics(
            truth_count=self._truth_count,
            matched_count=self._matched_count,
            missed_count=missed_count,
            false_track_count=self._false_track_count,
            id_switches=self._id_switches,
            range_rmse_m=range_rmse,
            velocity_rmse_mps=velocity_rmse,
            track_continuity=continuity,
        )


def evaluate_tracking_sequence(
    truth_frames: Iterable[Sequence[TruthTarget]],
    estimate_frames: Iterable[Sequence[TrackEstimate]],
    *,
    maximum_range_error_m: float = 2.0,
    maximum_velocity_error_mps: float = 2.0,
) -> TrackingMetrics:
    """Evaluate aligned truth and estimate frame sequences."""

    truth_frames_tuple = tuple(truth_frames)
    estimate_frames_tuple = tuple(estimate_frames)
    if len(truth_frames_tuple) != len(estimate_frames_tuple):
        raise ValueError("truth and estimate sequences must have equal length")
    accumulator = TrackingMetricsAccumulator(
        maximum_range_error_m=maximum_range_error_m,
        maximum_velocity_error_mps=maximum_velocity_error_mps,
    )
    for truths, estimates in zip(truth_frames_tuple, estimate_frames_tuple):
        accumulator.update(truths, estimates)
    return accumulator.result()
