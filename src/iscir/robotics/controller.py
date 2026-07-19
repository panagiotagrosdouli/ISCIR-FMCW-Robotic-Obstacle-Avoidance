"""Reactive obstacle-avoidance controller."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2

from iscir.sensing.sensor import Detection
from iscir.simulation.models import RobotState


@dataclass(slots=True)
class ReactiveController:
    """Goal-seeking controller with a simple frontal safety rule."""

    cruise_speed_mps: float = 0.8
    turn_rate_rps: float = 1.1
    safety_distance_m: float = 1.5
    frontal_half_angle_rad: float = 0.55

    def command(
        self,
        robot: RobotState,
        goal: tuple[float, float],
        detections: list[Detection],
    ) -> tuple[float, float]:
        frontal = [
            item
            for item in detections
            if abs(item.bearing_rad) <= self.frontal_half_angle_rad
            and item.range_m <= self.safety_distance_m
        ]

        if frontal:
            nearest = min(frontal, key=lambda item: item.range_m)
            turn_direction = -1.0 if nearest.bearing_rad >= 0.0 else 1.0
            return 0.15 * self.cruise_speed_mps, turn_direction * self.turn_rate_rps

        desired_heading = atan2(goal[1] - robot.y, goal[0] - robot.x)
        heading_error = (desired_heading - robot.heading + 3.141592653589793) % (
            2 * 3.141592653589793
        ) - 3.141592653589793
        angular_speed = max(-self.turn_rate_rps, min(self.turn_rate_rps, 1.5 * heading_error))
        return self.cruise_speed_mps, angular_speed
