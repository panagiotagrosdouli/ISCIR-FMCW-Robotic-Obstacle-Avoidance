"""Publication-ready plotting helpers for benchmark and scenario outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .benchmark import BenchmarkSummary
from .scenarios import SyntheticScenario


def plot_benchmark_summary(
    summaries: Sequence[BenchmarkSummary],
    output_dir: str | Path,
    *,
    dpi: int = 180,
) -> tuple[Path, ...]:
    """Create paper-ready benchmark figures and return their paths."""

    if not summaries:
        raise ValueError("at least one benchmark summary is required")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labels = [summary.scenario_name for summary in summaries]

    figures = (
        _bar_figure(
            labels,
            [summary.detection_probability_mean for summary in summaries],
            ylabel="Detection probability",
            title="Tracking detection probability",
            path=destination / "detection_probability.png",
            ylim=(0.0, 1.0),
            dpi=dpi,
        ),
        _bar_figure(
            labels,
            [summary.range_rmse_mean_m for summary in summaries],
            ylabel="Range RMSE (m)",
            title="Range estimation error",
            path=destination / "range_rmse.png",
            dpi=dpi,
        ),
        _bar_figure(
            labels,
            [summary.velocity_rmse_mean_mps for summary in summaries],
            ylabel="Velocity RMSE (m/s)",
            title="Velocity estimation error",
            path=destination / "velocity_rmse.png",
            dpi=dpi,
        ),
        _bar_figure(
            labels,
            [summary.track_continuity_mean for summary in summaries],
            ylabel="Track continuity",
            title="Persistent-track continuity",
            path=destination / "track_continuity.png",
            ylim=(0.0, 1.0),
            dpi=dpi,
        ),
        _bar_figure(
            labels,
            [summary.mean_frame_latency_ms for summary in summaries],
            ylabel="Mean frame latency (ms)",
            title="Tracker execution latency",
            path=destination / "frame_latency.png",
            dpi=dpi,
        ),
    )
    return figures


def plot_scenario_truth(
    scenario: SyntheticScenario,
    path: str | Path,
    *,
    dpi: int = 180,
) -> Path:
    """Plot ground-truth range trajectories for a synthetic scenario."""

    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    frames = scenario.generate(seed=0)
    time_s = [frame_index * scenario.config.dt_s for frame_index in range(len(frames))]
    object_ids = sorted({truth.object_id for frame in frames for truth in frame.truths})

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for object_id in object_ids:
        ranges = [
            next(
                (truth.range_m for truth in frame.truths if truth.object_id == object_id),
                float("nan"),
            )
            for frame in frames
        ]
        axis.plot(time_s, ranges, label=f"Target {object_id}")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Range (m)")
    axis.set_title(scenario.name.replace("_", " ").title())
    axis.grid(True, alpha=0.3)
    if object_ids:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _bar_figure(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
    ylim: tuple[float, float] | None = None,
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = list(range(len(labels)))
    axis.bar(positions, values)
    axis.set_xticks(positions, [label.replace("_", " ") for label in labels], rotation=20)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)
    if ylim is not None:
        axis.set_ylim(*ylim)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path
