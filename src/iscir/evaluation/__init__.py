"""Evaluation scenarios, metrics, and Monte Carlo benchmarks."""

from .benchmark import (
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSummary,
    run_monte_carlo,
    run_tracking_once,
    write_latex_table,
    write_summary_csv,
    write_summary_json,
)
from .scenarios import (
    MovingTarget,
    ScenarioConfig,
    ScenarioFrame,
    SyntheticScenario,
    crossing_scenario,
    dense_multi_target_scenario,
    head_on_scenario,
    truth_frames,
)
from .tracking_metrics import (
    TrackEstimate,
    TrackingMetrics,
    TrackingMetricsAccumulator,
    TruthTarget,
    evaluate_tracking_sequence,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkRun",
    "BenchmarkSummary",
    "MovingTarget",
    "ScenarioConfig",
    "ScenarioFrame",
    "SyntheticScenario",
    "TrackEstimate",
    "TrackingMetrics",
    "TrackingMetricsAccumulator",
    "TruthTarget",
    "crossing_scenario",
    "dense_multi_target_scenario",
    "evaluate_tracking_sequence",
    "head_on_scenario",
    "run_monte_carlo",
    "run_tracking_once",
    "truth_frames",
    "write_latex_table",
    "write_summary_csv",
    "write_summary_json",
]
