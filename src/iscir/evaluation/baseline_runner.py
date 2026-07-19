"""Command-line runner for CFAR, Kalman, and ISCIR baseline comparisons."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import isfinite
from statistics import mean
from time import perf_counter

from iscir.navigation import CollisionLevel, RadarNavigationPipeline
from iscir.sensing.tracker import MultiTargetTracker, TrackLifecycleConfig

from .scenarios import (
    ScenarioConfig,
    crossing_scenario,
    dense_multi_target_scenario,
    head_on_scenario,
)
from .tracking_metrics import TrackEstimate, TrackingMetricsAccumulator


@dataclass(frozen=True, slots=True)
class BaselineResult:
    backend: str
    scenario: str
    trials: int
    frames_per_trial: int
    detection_probability: float
    range_rmse_m: float
    velocity_rmse_mps: float
    track_continuity: float
    false_track_count: float
    false_tracks_per_frame: float
    id_switches: float
    mean_frame_latency_ms: float
    warning_frame_rate: float | None
    emergency_stop_rate: float | None
    mean_commanded_velocity_mps: float | None
    minimum_time_to_collision_s: float | None


@dataclass(frozen=True, slots=True)
class NavigationRunMetrics:
    warning_frame_rate: float
    emergency_stop_rate: float
    mean_commanded_velocity_mps: float
    minimum_time_to_collision_s: float | None


def _scenario(name: str, frames: int, target_count: int):
    config = ScenarioConfig(frame_count=frames)
    if name == "head_on":
        return head_on_scenario(config)
    if name == "crossing":
        return crossing_scenario(config)
    return dense_multi_target_scenario(target_count, config)


def _run_once(backend: str, scenario, seed: int):
    accumulator = TrackingMetricsAccumulator()
    latencies: list[float] = []
    warning_frames = 0
    emergency_stop_frames = 0
    commanded_velocities: list[float] = []
    finite_ttc_values: list[float] = []

    tracker = None
    pipeline = None
    if backend == "kalman":
        tracker = MultiTargetTracker(
            lifecycle_config=TrackLifecycleConfig(
                confirmation_hits=2,
                maximum_missed_updates=3,
            )
        )
    elif backend == "iscir":
        pipeline = RadarNavigationPipeline(
            tracker=MultiTargetTracker(
                lifecycle_config=TrackLifecycleConfig(
                    confirmation_hits=2,
                    maximum_missed_updates=3,
                )
            )
        )

    frames = scenario.generate(seed=seed)
    for frame in frames:
        start = perf_counter()
        if backend == "cfar":
            estimates = tuple(
                TrackEstimate(
                    track_id=index,
                    range_m=detection.range_m,
                    radial_velocity_mps=detection.radial_velocity_mps,
                )
                for index, detection in enumerate(frame.detections)
            )
        elif backend == "kalman":
            assert tracker is not None
            tracks = tracker.update(frame.detections, scenario.config.dt_s)
            estimates = tuple(
                TrackEstimate(
                    track_id=track.track_id,
                    range_m=track.range_m,
                    radial_velocity_mps=track.radial_velocity_mps,
                )
                for track in tracks
                if track.is_confirmed
            )
        else:
            assert pipeline is not None
            result = pipeline.update(frame.detections, scenario.config.dt_s)
            estimates = tuple(
                TrackEstimate(
                    track_id=track.track_id,
                    range_m=track.range_m,
                    radial_velocity_mps=track.radial_velocity_mps,
                )
                for track in result.tracks
                if track.is_confirmed
            )
            commanded_velocities.append(result.command.linear_speed_mps)
            if result.command.level is not CollisionLevel.CLEAR:
                warning_frames += 1
            if result.command.emergency_stop:
                emergency_stop_frames += 1
            finite_ttc_values.extend(
                risk.time_to_collision_s
                for risk in result.risks
                if isfinite(risk.time_to_collision_s)
            )

        latencies.append((perf_counter() - start) * 1000.0)
        accumulator.update(frame.truths, estimates)

    navigation_metrics = None
    if backend == "iscir":
        frame_count = len(frames)
        navigation_metrics = NavigationRunMetrics(
            warning_frame_rate=warning_frames / frame_count,
            emergency_stop_rate=emergency_stop_frames / frame_count,
            mean_commanded_velocity_mps=(
                mean(commanded_velocities) if commanded_velocities else 0.0
            ),
            minimum_time_to_collision_s=(
                min(finite_ttc_values) if finite_ttc_values else None
            ),
        )

    return (
        accumulator.result(),
        mean(latencies) if latencies else 0.0,
        navigation_metrics,
    )


def _mean_optional(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def run_baseline(
    backend: str,
    scenario_name: str,
    trials: int,
    frames: int,
    target_count: int,
    first_seed: int,
) -> BaselineResult:
    if trials < 1:
        raise ValueError("trials must be positive")
    if frames < 1:
        raise ValueError("frames must be positive")

    scenario = _scenario(scenario_name, frames, target_count)
    runs = [_run_once(backend, scenario, first_seed + index) for index in range(trials)]
    metrics = [run[0] for run in runs]
    navigation_runs = [run[2] for run in runs if run[2] is not None]

    false_track_count = mean(float(m.false_track_count) for m in metrics)
    return BaselineResult(
        backend=backend,
        scenario=scenario.name,
        trials=trials,
        frames_per_trial=frames,
        detection_probability=mean(m.detection_probability for m in metrics),
        range_rmse_m=mean(m.range_rmse_m for m in metrics),
        velocity_rmse_mps=mean(m.velocity_rmse_mps for m in metrics),
        track_continuity=mean(m.track_continuity for m in metrics),
        false_track_count=false_track_count,
        false_tracks_per_frame=false_track_count / frames,
        id_switches=mean(float(m.id_switches) for m in metrics),
        mean_frame_latency_ms=mean(run[1] for run in runs),
        warning_frame_rate=(
            mean(item.warning_frame_rate for item in navigation_runs)
            if navigation_runs
            else None
        ),
        emergency_stop_rate=(
            mean(item.emergency_stop_rate for item in navigation_runs)
            if navigation_runs
            else None
        ),
        mean_commanded_velocity_mps=(
            mean(item.mean_commanded_velocity_mps for item in navigation_runs)
            if navigation_runs
            else None
        ),
        minimum_time_to_collision_s=_mean_optional(
            [item.minimum_time_to_collision_s for item in navigation_runs]
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cfar", "kalman", "iscir"), required=True)
    parser.add_argument(
        "--scenario", choices=("head_on", "crossing", "dense"), default="crossing"
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--target-count", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--output", type=str)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_baseline(
        backend=args.backend,
        scenario_name=args.scenario,
        trials=args.trials,
        frames=args.frames,
        target_count=args.target_count,
        first_seed=args.first_seed,
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    print(payload)
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
