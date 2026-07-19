"""Demonstrate persistent multi-target tracking on noisy radar detections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iscir.sensing.tracker import (
    MultiTargetTracker,
    RadarDetection,
    TrackLifecycleConfig,
    TrackerManagerConfig,
)


@dataclass(frozen=True, slots=True)
class SyntheticTarget:
    """Simple one-dimensional target used to generate radar detections."""

    initial_range_m: float
    radial_velocity_mps: float
    detection_confidence: float = 0.95

    def range_at(self, time_s: float) -> float:
        return self.initial_range_m + self.radial_velocity_mps * time_s


def generate_detections(
    targets: tuple[SyntheticTarget, ...],
    *,
    time_s: float,
    rng: np.random.Generator,
    range_noise_std_m: float = 0.12,
    velocity_noise_std_mps: float = 0.18,
    detection_probability: float = 0.92,
) -> list[RadarDetection]:
    """Generate noisy detections with occasional missed observations."""

    detections: list[RadarDetection] = []
    for target in targets:
        if rng.random() > detection_probability:
            continue

        measured_range_m = max(
            0.0,
            target.range_at(time_s) + rng.normal(0.0, range_noise_std_m),
        )
        measured_velocity_mps = (
            target.radial_velocity_mps
            + rng.normal(0.0, velocity_noise_std_mps)
        )
        detections.append(
            RadarDetection(
                range_m=float(measured_range_m),
                radial_velocity_mps=float(measured_velocity_mps),
                confidence=target.detection_confidence,
            )
        )

    # Shuffle detections so persistent IDs cannot depend on input ordering.
    rng.shuffle(detections)
    return detections


def main() -> None:
    rng = np.random.default_rng(17)
    dt_s = 0.1
    frame_count = 40

    targets = (
        SyntheticTarget(initial_range_m=6.0, radial_velocity_mps=-0.55),
        SyntheticTarget(initial_range_m=10.5, radial_velocity_mps=0.30),
        SyntheticTarget(initial_range_m=15.0, radial_velocity_mps=-1.05),
    )

    tracker = MultiTargetTracker(
        manager_config=TrackerManagerConfig(gate_threshold_squared=9.21),
        lifecycle_config=TrackLifecycleConfig(
            confirmation_hits=3,
            maximum_missed_updates=3,
        ),
    )

    for frame_index in range(frame_count):
        time_s = frame_index * dt_s
        detections = generate_detections(
            targets,
            time_s=time_s,
            rng=rng,
        )
        tracks = tracker.update(detections, dt_s)

        print(
            f"Frame {frame_index:02d} | t={time_s:4.1f} s | "
            f"detections={len(detections)} | active_tracks={len(tracks)}"
        )
        if not tracks:
            print("  no active tracks")
            continue

        for track in tracks:
            print(
                f"  Track {track.track_id:02d} | "
                f"{track.status.value:9s} | "
                f"range={track.range_m:6.2f} m | "
                f"velocity={track.radial_velocity_mps:6.2f} m/s | "
                f"confidence={track.confidence:.2f} | "
                f"missed={track.missed_updates}"
            )

    print("\nConfirmed tracks")
    for track in tracker.confirmed_tracks:
        print(
            f"  Track {track.track_id:02d}: "
            f"range={track.range_m:.2f} m, "
            f"velocity={track.radial_velocity_mps:.2f} m/s"
        )


if __name__ == "__main__":
    main()
