from pathlib import Path

from iscir.evaluation.benchmark import (
    run_monte_carlo,
    run_tracking_once,
    write_latex_table,
    write_summary_csv,
    write_summary_json,
)
from iscir.evaluation.scenarios import ScenarioConfig, head_on_scenario


def _perfect_scenario():
    return head_on_scenario(
        ScenarioConfig(
            frame_count=12,
            detection_probability=1.0,
            range_noise_std_m=0.0,
            velocity_noise_std_mps=0.0,
            clutter_rate_per_frame=0.0,
        )
    )


def test_single_run_produces_tracking_metrics_and_latency() -> None:
    result = run_tracking_once(_perfect_scenario(), seed=5)

    assert result.seed == 5
    assert result.metrics.truth_count == 12
    assert result.metrics.detection_probability > 0.8
    assert result.metrics.range_rmse_m < 0.5
    assert result.metrics.velocity_rmse_mps < 0.5
    assert result.mean_frame_latency_ms >= 0.0
    assert result.maximum_frame_latency_ms >= result.mean_frame_latency_ms


def test_monte_carlo_summary_has_requested_repetitions() -> None:
    result = run_monte_carlo(_perfect_scenario(), repetitions=3, first_seed=10)

    assert result.summary.scenario_name == "head_on"
    assert result.summary.repetitions == 3
    assert len(result.runs) == 3
    assert [run.seed for run in result.runs] == [10, 11, 12]
    assert 0.0 <= result.summary.detection_probability_mean <= 1.0


def test_benchmark_writers_create_paper_ready_outputs(tmp_path: Path) -> None:
    summary = run_monte_carlo(_perfect_scenario(), repetitions=2).summary

    csv_path = write_summary_csv([summary], tmp_path / "results.csv")
    json_path = write_summary_json(summary, tmp_path / "results.json")
    tex_path = write_latex_table([summary], tmp_path / "results.tex")

    assert "scenario_name" in csv_path.read_text(encoding="utf-8")
    assert '"scenario_name": "head_on"' in json_path.read_text(encoding="utf-8")
    latex = tex_path.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in latex
    assert "head\\_on" in latex
