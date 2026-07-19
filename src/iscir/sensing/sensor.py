"""Synthetic FMCW-like range and radial-velocity sensor."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np

from iscir.simulation.models import Scene


@dataclass(frozen=True, slots=True)
class Detection:
    range_m: float
    bearing_rad: float
    radial_velocity_mps: float


@dataclass(slots=True)
class FMCWSensor:
    """Generate noisy detections from known simulated obstacle states.

    This is an abstract measurement-level model, not yet a raw chirp/beat-signal
    implementation. It lets the robotics layer be developed before the full
    signal-processing chain is added.
    """

    max_range_m: float = 20.0
    field_of_view_rad: float = pi
    range_std_m: float = 0.05
    velocity_std_mps: float = 0.05
    rng_seed: int | None = 7

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + pi) % (2 * pi) - pi

    def scan(self, scene: Scene) -> list[Detection]:
        robot = scene.robot
        detections: list[Detection] = []

        for obstacle in scene.obstacles:
            dx = obstacle.x - robot.x
            dy = obstacle.y - robot.y
            true_range = hypot(dx, dy)
            bearing = self._wrap_angle(atan2(dy, dx) - robot.heading)

            if true_range > self.max_range_m or abs(bearing) > self.field_of_view_rad / 2:
                continue

            line_x = cos(atan2(dy, dx))
            line_y = sin(atan2(dy, dx))
            radial_velocity = obstacle.vx * line_x + obstacle.vy * line_y

            detections.append(
                Detection(
                    range_m=max(0.0, true_range + self._rng.normal(0.0, self.range_std_m)),
                    bearing_rad=bearing,
                    radial_velocity_mps=radial_velocity
                    + self._rng.normal(0.0, self.velocity_std_mps),
                )
            )

        return sorted(detections, key=lambda item: item.range_m)
