"""Scene sensor that estimates obstacle range from synthetic FMCW beat signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin

from iscir.sensing.sensor import Detection
from iscir.sensing.signal import FMCWConfig, PointTarget, estimate_range_m, simulate_beat_signal
from iscir.simulation.models import Scene


@dataclass(slots=True)
class FFTFMCWSensor:
    """Convert simulated obstacle states into detections using a Range FFT.

    Each visible obstacle is simulated as an independent point target. Its range is
    estimated from a noisy synthetic beat signal, while bearing is taken from scene
    geometry. This deliberately separates the range-processing milestone from the
    later angular-estimation and multi-target peak-association milestones.
    """

    config: FMCWConfig = field(default_factory=FMCWConfig)
    max_range_m: float = 20.0
    field_of_view_rad: float = pi
    snr_db: float | None = 20.0
    n_fft: int = 8192
    rng_seed: int | None = 7

    def __post_init__(self) -> None:
        if self.max_range_m <= 0:
            raise ValueError("max_range_m must be positive")
        if not 0 < self.field_of_view_rad <= 2 * pi:
            raise ValueError("field_of_view_rad must be in (0, 2*pi]")
        if self.n_fft < self.config.num_samples:
            raise ValueError("n_fft cannot be smaller than the chirp sample count")

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + pi) % (2 * pi) - pi

    def scan(self, scene: Scene) -> list[Detection]:
        robot = scene.robot
        detections: list[Detection] = []

        for index, obstacle in enumerate(scene.obstacles):
            dx = obstacle.x - robot.x
            dy = obstacle.y - robot.y
            true_range = hypot(dx, dy)
            absolute_bearing = atan2(dy, dx)
            relative_bearing = self._wrap_angle(absolute_bearing - robot.heading)

            if true_range > self.max_range_m:
                continue
            if abs(relative_bearing) > self.field_of_view_rad / 2:
                continue
            if true_range > self.config.maximum_unambiguous_range_m:
                continue

            line_x = cos(absolute_bearing)
            line_y = sin(absolute_bearing)
            radial_velocity = obstacle.vx * line_x + obstacle.vy * line_y

            _, beat_signal = simulate_beat_signal(
                self.config,
                [PointTarget(range_m=true_range, radial_velocity_mps=radial_velocity)],
                snr_db=self.snr_db,
                rng_seed=None if self.rng_seed is None else self.rng_seed + index,
            )
            estimated_range = estimate_range_m(beat_signal, self.config, n_fft=self.n_fft)

            detections.append(
                Detection(
                    range_m=estimated_range,
                    bearing_rad=relative_bearing,
                    radial_velocity_mps=radial_velocity,
                )
            )

        return sorted(detections, key=lambda item: item.range_m)
