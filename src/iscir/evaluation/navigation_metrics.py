"""Sequence-level metrics for risk-aware navigation commands."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter

from iscir.navigation import RadarNavigationPipeline

from .scenarios import SyntheticScenario


@dataclass(frozen=True, slots=True)
class NavigationMetrics:
    """Aggregate controller behavior over one synthetic scenario."""

    frame_count: int
    mean_command_speed_mps: float
    minimum_command_speed_mps: float
    reduced_speed_fraction: float
    emergency_stop_count: int
    first_intervention_frame: int | None
    mean_pipeline_latency_ms: float
    maximum_pipeline_latency_ms: float



def evaluate_navigation_scenario(
    scenario: SyntheticScenario,
    *,
    seed: int = 0,
    pipeline: RadarNavigationPipeline | None = None,
) -> NavigationMetrics:
    """Run detections through the navigation pipeline and summarize commands."""

    navigation = pipeline or RadarNavigationPipeline()
    speeds: list[float] = []
    latencies_ms: list[float] = []
    emergency_stop_count = 0
    first_intervention_frame: int | None = None
    cruise_speed = navigation.controller_config.cruise_speed_mps

    for frame_index, frame in enumerate(scenario.generate(seed=seed)):
        start = perf_counter()
        result = navigation.update(frame.detections, scenario.config.dt_s)
        latencies_ms.append((perf_counter() - start) * 1000.0)

        speed = result.command.linear_speed_mps
        speeds.append(speed)
        if speed < cruise_speed and first_intervention_frame is None:
            first_intervention_frame = frame_index
        if result.command.emergency_stop:
            emergency_stop_count += 1

    reduced_count = sum(speed < cruise_speed for speed in speeds)
    frame_count = len(speeds)
    return NavigationMetrics(
        frame_count=frame_count,
        mean_command_speed_mps=mean(speeds) if speeds else 0.0,
        minimum_command_speed_mps=min(speeds, default=0.0),
        reduced_speed_fraction=reduced_count / frame_count if frame_count else 0.0,
        emergency_stop_count=emergency_stop_count,
        first_intervention_frame=first_intervention_frame,
        mean_pipeline_latency_ms=mean(latencies_ms) if latencies_ms else 0.0,
        maximum_pipeline_latency_ms=max(latencies_ms, default=0.0),
    )
