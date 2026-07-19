"""Deterministic synthetic radar scenarios for tracking experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from iscir.sensing.tracker import RadarDetection

from .tracking_metrics import TruthTarget


@dataclass(frozen=True, slots=True)
class MovingTarget:
    """One-dimensional constant-velocity target used by a synthetic scenario."""

    object_id: int
    initial_range_m: float
    radial_velocity_mps: float
    start_time_s: float = 0.0
    end_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.object_id < 0:
            raise ValueError("object_id cannot be negative")
        if self.initial_range_m < 0.0:
            raise ValueError("initial_range_m cannot be negative")
        if self.start_time_s < 0.0:
            raise ValueError("start_time_s cannot be negative")
        if self.end_time_s is not None and self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s cannot precede start_time_s")

    def state_at(self, time_s: float) -> TruthTarget | None:
        """Return the target state at ``time_s`` or ``None`` when inactive."""

        if time_s < self.start_time_s:
            return None
        if self.end_time_s is not None and time_s > self.end_time_s:
            return None
        elapsed_s = time_s - self.start_time_s
        range_m = self.initial_range_m + self.radial_velocity_mps * elapsed_s
        if range_m < 0.0:
            return None
        return TruthTarget(self.object_id, range_m, self.radial_velocity_mps)


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Noise, missed-detection, and clutter controls for synthetic detections."""

    dt_s: float = 0.1
    frame_count: int = 100
    detection_probability: float = 0.95
    range_noise_std_m: float = 0.20
    velocity_noise_std_mps: float = 0.10
    clutter_rate_per_frame: float = 0.0
    minimum_clutter_range_m: float = 0.0
    maximum_clutter_range_m: float = 50.0
    minimum_clutter_velocity_mps: float = -5.0
    maximum_clutter_velocity_mps: float = 5.0

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be between 0 and 1")
        if self.range_noise_std_m < 0.0:
            raise ValueError("range_noise_std_m cannot be negative")
        if self.velocity_noise_std_mps < 0.0:
            raise ValueError("velocity_noise_std_mps cannot be negative")
        if self.clutter_rate_per_frame < 0.0:
            raise ValueError("clutter_rate_per_frame cannot be negative")
        if self.maximum_clutter_range_m <= self.minimum_clutter_range_m:
            raise ValueError("maximum_clutter_range_m must exceed minimum_clutter_range_m")
        if self.maximum_clutter_velocity_mps <= self.minimum_clutter_velocity_mps:
            raise ValueError(
                "maximum_clutter_velocity_mps must exceed minimum_clutter_velocity_mps"
            )


@dataclass(frozen=True, slots=True)
class ScenarioFrame:
    """Ground truth and radar detections for one simulated frame."""

    frame_index: int
    time_s: float
    truths: tuple[TruthTarget, ...]
    detections: tuple[RadarDetection, ...]


@dataclass(frozen=True, slots=True)
class SyntheticScenario:
    """Named collection of moving targets and simulation controls."""

    name: str
    targets: tuple[MovingTarget, ...]
    config: ScenarioConfig = ScenarioConfig()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name cannot be empty")
        object_ids = [target.object_id for target in self.targets]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("target object IDs must be unique")

    def generate(self, *, seed: int = 0) -> tuple[ScenarioFrame, ...]:
        """Generate a reproducible sequence of truth states and detections."""

        rng = np.random.default_rng(seed)
        frames: list[ScenarioFrame] = []
        for frame_index in range(self.config.frame_count):
            time_s = frame_index * self.config.dt_s
            truths = tuple(
                state
                for target in self.targets
                if (state := target.state_at(time_s)) is not None
            )
            detections: list[RadarDetection] = []
            for truth in truths:
                if rng.random() > self.config.detection_probability:
                    continue
                detections.append(
                    RadarDetection(
                        range_m=max(
                            0.0,
                            truth.range_m
                            + float(rng.normal(0.0, self.config.range_noise_std_m)),
                        ),
                        radial_velocity_mps=(
                            truth.radial_velocity_mps
                            + float(
                                rng.normal(0.0, self.config.velocity_noise_std_mps)
                            )
                        ),
                        confidence=0.9,
                    )
                )

            clutter_count = int(rng.poisson(self.config.clutter_rate_per_frame))
            for _ in range(clutter_count):
                detections.append(
                    RadarDetection(
                        range_m=float(
                            rng.uniform(
                                self.config.minimum_clutter_range_m,
                                self.config.maximum_clutter_range_m,
                            )
                        ),
                        radial_velocity_mps=float(
                            rng.uniform(
                                self.config.minimum_clutter_velocity_mps,
                                self.config.maximum_clutter_velocity_mps,
                            )
                        ),
                        confidence=0.25,
                    )
                )

            rng.shuffle(detections)
            frames.append(
                ScenarioFrame(
                    frame_index=frame_index,
                    time_s=time_s,
                    truths=truths,
                    detections=tuple(detections),
                )
            )
        return tuple(frames)


def head_on_scenario(config: ScenarioConfig | None = None) -> SyntheticScenario:
    """Single approaching obstacle suitable for collision-response experiments."""

    return SyntheticScenario(
        name="head_on",
        targets=(MovingTarget(0, initial_range_m=20.0, radial_velocity_mps=-2.0),),
        config=config or ScenarioConfig(),
    )


def crossing_scenario(config: ScenarioConfig | None = None) -> SyntheticScenario:
    """Two targets whose range trajectories cross during the sequence."""

    return SyntheticScenario(
        name="crossing",
        targets=(
            MovingTarget(0, initial_range_m=10.0, radial_velocity_mps=1.0),
            MovingTarget(1, initial_range_m=25.0, radial_velocity_mps=-1.0),
        ),
        config=config or ScenarioConfig(),
    )


def dense_multi_target_scenario(
    target_count: int = 5,
    config: ScenarioConfig | None = None,
) -> SyntheticScenario:
    """Create a repeatable mixed-velocity multi-target benchmark."""

    if target_count < 1:
        raise ValueError("target_count must be positive")
    targets = tuple(
        MovingTarget(
            object_id=index,
            initial_range_m=6.0 + 4.0 * index,
            radial_velocity_mps=(-1.0 if index % 2 else 0.75) + 0.05 * index,
        )
        for index in range(target_count)
    )
    return SyntheticScenario(
        name=f"dense_{target_count}",
        targets=targets,
        config=config or ScenarioConfig(),
    )


def truth_frames(frames: Iterable[ScenarioFrame]) -> tuple[Sequence[TruthTarget], ...]:
    """Extract aligned ground-truth frames from generated scenario frames."""

    return tuple(frame.truths for frame in frames)
