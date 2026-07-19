"""Scalable, resumable execution of very large synthetic experiment campaigns."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product
from math import sqrt
from pathlib import Path
import csv
import json
from typing import Iterable, Iterator, Sequence

from .benchmark import run_tracking_once
from .scenarios import (
    ScenarioConfig,
    crossing_scenario,
    dense_multi_target_scenario,
    head_on_scenario,
)


@dataclass(frozen=True, slots=True)
class ExperimentCondition:
    """One point in the synthetic evaluation parameter grid."""

    scenario_name: str
    detection_probability: float
    range_noise_std_m: float
    velocity_noise_std_mps: float
    clutter_rate_per_frame: float
    target_count: int = 1
    frame_count: int = 40

    def __post_init__(self) -> None:
        if self.scenario_name not in {"head_on", "crossing", "dense"}:
            raise ValueError("scenario_name must be head_on, crossing, or dense")
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be between 0 and 1")
        if self.target_count < 1 or self.frame_count < 1:
            raise ValueError("target_count and frame_count must be positive")


@dataclass(frozen=True, slots=True)
class MassiveCampaignConfig:
    """Execution settings for a large Monte Carlo campaign."""

    total_trials: int = 1_000_000
    workers: int | None = None
    chunk_size: int = 1_000
    first_seed: int = 0
    checkpoint_every: int = 10_000

    def __post_init__(self) -> None:
        if self.total_trials < 1:
            raise ValueError("total_trials must be positive")
        if self.chunk_size < 1 or self.checkpoint_every < 1:
            raise ValueError("chunk_size and checkpoint_every must be positive")
        if self.first_seed < 0:
            raise ValueError("first_seed cannot be negative")


@dataclass(slots=True)
class OnlineStatistics:
    """Numerically stable streaming mean and population variance."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        return sqrt(self.m2 / self.count) if self.count else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {"count": self.count, "mean": self.mean, "std": self.std, "m2": self.m2}

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> "OnlineStatistics":
        return cls(count=int(data["count"]), mean=float(data["mean"]), m2=float(data["m2"]))


@dataclass(slots=True)
class CampaignAccumulator:
    detection_probability: OnlineStatistics
    range_rmse_m: OnlineStatistics
    velocity_rmse_mps: OnlineStatistics
    track_continuity: OnlineStatistics
    false_track_count: OnlineStatistics
    id_switches: OnlineStatistics
    mean_frame_latency_ms: OnlineStatistics

    @classmethod
    def empty(cls) -> "CampaignAccumulator":
        return cls(*(OnlineStatistics() for _ in range(7)))

    def update(self, row: dict[str, float | int | str]) -> None:
        self.detection_probability.update(float(row["detection_probability"]))
        self.range_rmse_m.update(float(row["range_rmse_m"]))
        self.velocity_rmse_mps.update(float(row["velocity_rmse_mps"]))
        self.track_continuity.update(float(row["track_continuity"]))
        self.false_track_count.update(float(row["false_track_count"]))
        self.id_switches.update(float(row["id_switches"]))
        self.mean_frame_latency_ms.update(float(row["mean_frame_latency_ms"]))

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        return {name: value.to_dict() for name, value in vars(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float | int]]) -> "CampaignAccumulator":
        return cls(**{name: OnlineStatistics.from_dict(value) for name, value in data.items()})


def default_condition_grid() -> tuple[ExperimentCondition, ...]:
    """Return a broad grid spanning noise, clutter, detection rate, and density."""

    conditions: list[ExperimentCondition] = []
    for scenario_name, pd, range_noise, velocity_noise, clutter in product(
        ("head_on", "crossing"),
        (0.70, 0.80, 0.90, 0.95, 0.99),
        (0.05, 0.15, 0.30, 0.60, 1.00),
        (0.03, 0.08, 0.15, 0.30),
        (0.0, 0.25, 0.5, 1.0, 2.0),
    ):
        conditions.append(
            ExperimentCondition(
                scenario_name=scenario_name,
                detection_probability=pd,
                range_noise_std_m=range_noise,
                velocity_noise_std_mps=velocity_noise,
                clutter_rate_per_frame=clutter,
                target_count=2 if scenario_name == "crossing" else 1,
            )
        )
    for target_count in (3, 5, 8, 10):
        for pd, clutter in product((0.80, 0.90, 0.95), (0.5, 1.0, 2.0, 4.0)):
            conditions.append(
                ExperimentCondition(
                    scenario_name="dense",
                    detection_probability=pd,
                    range_noise_std_m=0.30,
                    velocity_noise_std_mps=0.15,
                    clutter_rate_per_frame=clutter,
                    target_count=target_count,
                )
            )
    return tuple(conditions)


def iter_trial_specs(
    conditions: Sequence[ExperimentCondition], total_trials: int, first_seed: int = 0
) -> Iterator[tuple[ExperimentCondition, int]]:
    if not conditions:
        raise ValueError("at least one condition is required")
    for index in range(total_trials):
        yield conditions[index % len(conditions)], first_seed + index


def run_massive_campaign(
    output_dir: str | Path,
    *,
    conditions: Sequence[ExperimentCondition] | None = None,
    config: MassiveCampaignConfig | None = None,
) -> Path:
    """Run a memory-bounded, resumable campaign and return the summary path."""

    settings = config or MassiveCampaignConfig()
    condition_grid = tuple(conditions or default_condition_grid())
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.json"
    rows_path = directory / "trials.csv"
    summary_path = directory / "summary.json"

    completed, accumulator = _load_checkpoint(checkpoint_path)
    if completed >= settings.total_trials:
        return summary_path

    specs = iter_trial_specs(
        condition_grid,
        settings.total_trials - completed,
        settings.first_seed + completed,
    )
    write_header = not rows_path.exists() or rows_path.stat().st_size == 0
    with rows_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_result_fieldnames())
        if write_header:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=settings.workers) as pool:
            for offset, row in enumerate(
                pool.map(_run_trial, specs, chunksize=settings.chunk_size), start=completed + 1
            ):
                writer.writerow(row)
                accumulator.update(row)
                if offset % settings.checkpoint_every == 0:
                    handle.flush()
                    _write_checkpoint(checkpoint_path, offset, accumulator)

    _write_checkpoint(checkpoint_path, settings.total_trials, accumulator)
    summary_path.write_text(
        json.dumps(
            {
                "total_trials": settings.total_trials,
                "condition_count": len(condition_grid),
                "statistics": accumulator.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def _run_trial(spec: tuple[ExperimentCondition, int]) -> dict[str, float | int | str]:
    condition, seed = spec
    scenario_config = ScenarioConfig(
        frame_count=condition.frame_count,
        detection_probability=condition.detection_probability,
        range_noise_std_m=condition.range_noise_std_m,
        velocity_noise_std_mps=condition.velocity_noise_std_mps,
        clutter_rate_per_frame=condition.clutter_rate_per_frame,
    )
    if condition.scenario_name == "head_on":
        scenario = head_on_scenario(scenario_config)
    elif condition.scenario_name == "crossing":
        scenario = crossing_scenario(scenario_config)
    else:
        scenario = dense_multi_target_scenario(condition.target_count, scenario_config)
    run = run_tracking_once(scenario, seed=seed)
    return {
        **asdict(condition),
        "seed": seed,
        "detection_probability": run.metrics.detection_probability,
        "range_rmse_m": run.metrics.range_rmse_m,
        "velocity_rmse_mps": run.metrics.velocity_rmse_mps,
        "track_continuity": run.metrics.track_continuity,
        "false_track_count": run.metrics.false_track_count,
        "id_switches": run.metrics.id_switches,
        "mean_frame_latency_ms": run.mean_frame_latency_ms,
    }


def _result_fieldnames() -> list[str]:
    return [
        "scenario_name", "detection_probability", "range_noise_std_m",
        "velocity_noise_std_mps", "clutter_rate_per_frame", "target_count",
        "frame_count", "seed", "range_rmse_m", "velocity_rmse_mps",
        "track_continuity", "false_track_count", "id_switches",
        "mean_frame_latency_ms",
    ]


def _load_checkpoint(path: Path) -> tuple[int, CampaignAccumulator]:
    if not path.exists():
        return 0, CampaignAccumulator.empty()
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data["completed_trials"]), CampaignAccumulator.from_dict(data["statistics"])


def _write_checkpoint(path: Path, completed: int, accumulator: CampaignAccumulator) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"completed_trials": completed, "statistics": accumulator.to_dict()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
