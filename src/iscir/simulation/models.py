"""Minimal two-dimensional simulation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, hypot, sin


@dataclass(slots=True)
class Obstacle:
    """Circular obstacle with optional planar velocity."""

    x: float
    y: float
    radius: float = 0.3
    vx: float = 0.0
    vy: float = 0.0

    def step(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt


@dataclass(slots=True)
class RobotState:
    """Planar unicycle robot state."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    radius: float = 0.25

    def step(self, linear_speed: float, angular_speed: float, dt: float) -> None:
        self.heading += angular_speed * dt
        self.x += linear_speed * cos(self.heading) * dt
        self.y += linear_speed * sin(self.heading) * dt


@dataclass(slots=True)
class Scene:
    """Robot, goal and obstacle collection used by the simulator."""

    robot: RobotState
    goal: tuple[float, float]
    obstacles: list[Obstacle] = field(default_factory=list)

    def step_obstacles(self, dt: float) -> None:
        for obstacle in self.obstacles:
            obstacle.step(dt)

    def distance_to_goal(self) -> float:
        return hypot(self.goal[0] - self.robot.x, self.goal[1] - self.robot.y)

    def has_collision(self) -> bool:
        for obstacle in self.obstacles:
            centre_distance = hypot(obstacle.x - self.robot.x, obstacle.y - self.robot.y)
            if centre_distance <= obstacle.radius + self.robot.radius:
                return True
        return False
