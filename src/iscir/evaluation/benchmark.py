"""Monte Carlo benchmark runner for synthetic radar tracking scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
from statistics import mean, pstdev
from time import perf_counter
from typing import Iterable, Sequence

from iscir.sensing.tracker import MultiTargetTracker, TrackLifecycleConfig

from .scenarios import SyntheticScenario
from .tracking_metrics import (
    TrackEstimate,
    TrackingMetrics,
    TrackingMetricsAccumulator,
)


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Metrics and latency from one scenario realization."""

    seed: int
    metrics: TrackingMetrics
    mean_frame_latency_ms: float
    maximum_frame_latency_ms: float


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Aggregate Monte Carlo statistics for one scenario."""

    scenario_name: str
    repetitions: int
    detection_probability_mean: float
    detection_probability_std: float
    range_rmse_mean_m: float
    range_rmse_std_m: float
    velocity_rmse_mean_mps: float
    velocity_rmse_std_mps: float
    track_continuity_mean: float
    track_continuity_std: float
    false_track_count_mean: float
    id_switches_mean: float
    mean_frame_latency_ms: float
    maximum_frame_latency_ms: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a serialization-friendly dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Complete per-run and aggregate benchmark output."""

    summary: BenchmarkSummary
    runs: tuple[BenchmarkRun, ...]


def run_tracking_once(
    scenario: SyntheticScenario,
    *,
    seed: int = 0,
    maximum_range_error_m: float = 2.0,
    maximum_velocity_error_mps: float = 2.0,
) -> BenchmarkRun:
    """Generate one scenario realization, run tracking, and compute metrics."""

    tracker = MultiTargetTracker(
        lifecycle_config=TrackLifecycleConfig(
            confirmation_hits=2,
            maximum_missed_updates=3,
        )
    )
    accumulator = TrackingMetricsAccumulator(
        maximum_range_error_m=maximum_range_error_m,
        maximum_velocity_error_mps=maximum_velocity_error_mps,
    )
    latencies_ms: list[float] = []

    for frame in scenario.generate(seed=seed):
        start = perf_counter()
        tracks = tracker.update(frame.detections, scenario.config.dt_s)
        elapsed_ms = (perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        estimates = tuple(
            TrackEstimate(
                track_id=track.track_id,
                range_m=track.range_m,
                radial_velocity_mps=track.radial_velocity_mps,
            )
            for track in tracks
            if track.is_confirmed
        )
        accumulator.update(frame.truths, estimates)

    return BenchmarkRun(
        seed=seed,
        metrics=accumulator.result(),
        mean_frame_latency_ms=mean(latencies_ms) if latencies_ms else 0.0,
        maximum_frame_latency_ms=max(latencies_ms, default=0.0),
    )


def run_monte_carlo(
    scenario: SyntheticScenario,
    *,
    repetitions: int = 100,
    first_seed: int = 0,
) -> BenchmarkResult:
    """Run reproducible Monte Carlo tracking experiments."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if first_seed < 0:
        raise ValueError("first_seed cannot be negative")

    runs = tuple(
        run_tracking_once(scenario, seed=first_seed + index)
        for index in range(repetitions)
    )
    detection_probabilities = [run.metrics.detection_probability for run in runs]
    range_rmses = [run.metrics.range_rmse_m for run in runs]
    velocity_rmses = [run.metrics.velocity_rmse_mps for run in runs]
    continuities = [run.metrics.track_continuity for run in runs]

    summary = BenchmarkSummary(
        scenario_name=scenario.name,
        repetitions=repetitions,
        detection_probability_mean=mean(detection_probabilities),
        detection_probability_std=_population_std(detection_probabilities),
        range_rmse_mean_m=mean(range_rmses),
        range_rmse_std_m=_population_std(range_rmses),
        velocity_rmse_mean_mps=mean(velocity_rmses),
        velocity_rmse_std_mps=_population_std(velocity_rmses),
        track_continuity_mean=mean(continuities),
        track_continuity_std=_population_std(continuities),
        false_track_count_mean=mean(
            [float(run.metrics.false_track_count) for run in runs]
        ),
        id_switches_mean=mean([float(run.metrics.id_switches) for run in runs]),
        mean_frame_latency_ms=mean([run.mean_frame_latency_ms for run in runs]),
        maximum_frame_latency_ms=max(
            (run.maximum_frame_latency_ms for run in runs), default=0.0
        ),
    )
    return BenchmarkResult(summary=summary, runs=runs)


def write_summary_csv(
    summaries: Iterable[BenchmarkSummary], path: str | Path
) -> Path:
    """Write one row per scenario for downstream plots and paper tables."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary.to_dict() for summary in summaries]
    if not rows:
        raise ValueError("at least one benchmark summary is required")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_summary_json(summary: BenchmarkSummary, path: str | Path) -> Path:
    """Write a benchmark summary as formatted JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_latex_table(
    summaries: Sequence[BenchmarkSummary], path: str | Path
) -> Path:
    """Write a compact LaTeX table suitable for the paper results section."""

    if not summaries:
        raise ValueError("at least one benchmark summary is required")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Scenario & $P_d$ & Range RMSE (m) & Velocity RMSE (m/s) & Continuity & ID switches \\",
        r"\hline",
    ]
    for summary in summaries:
        lines.append(
            f"{summary.scenario_name.replace('_', r'\_')} & "
            f"{summary.detection_probability_mean:.3f} & "
            f"{summary.range_rmse_mean_m:.3f} & "
            f"{summary.velocity_rmse_mean_mps:.3f} & "
            f"{summary.track_continuity_mean:.3f} & "
            f"{summary.id_switches_mean:.2f} \\\\" 
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _population_std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0
