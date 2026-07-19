from pathlib import Path

import pytest

from iscir.evaluation import BenchmarkSummary, head_on_scenario
from iscir.evaluation.plotting import plot_benchmark_summary, plot_scenario_truth


def _summary(name: str) -> BenchmarkSummary:
    return BenchmarkSummary(
        scenario_name=name,
        repetitions=3,
        detection_probability_mean=0.9,
        detection_probability_std=0.02,
        range_rmse_mean_m=0.3,
        range_rmse_std_m=0.04,
        velocity_rmse_mean_mps=0.2,
        velocity_rmse_std_mps=0.03,
        track_continuity_mean=0.88,
        track_continuity_std=0.05,
        false_track_count_mean=1.0,
        id_switches_mean=0.0,
        mean_frame_latency_ms=0.5,
        maximum_frame_latency_ms=0.9,
    )


def test_plot_benchmark_summary_writes_expected_figures(tmp_path: Path) -> None:
    paths = plot_benchmark_summary([_summary("head_on"), _summary("crossing")], tmp_path)

    assert len(paths) == 5
    assert {path.name for path in paths} == {
        "detection_probability.png",
        "range_rmse.png",
        "velocity_rmse.png",
        "track_continuity.png",
        "frame_latency.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_plot_scenario_truth_writes_png(tmp_path: Path) -> None:
    output = plot_scenario_truth(head_on_scenario(), tmp_path / "head_on_truth.png")

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_benchmark_summary_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_benchmark_summary([], tmp_path)
