"""Evaluation scenarios, metrics, Monte Carlo benchmarks, and plots."""

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
from .navigation_metrics import NavigationMetrics, evaluate_navigation_scenario
from .plotting import plot_benchmark_summary, plot_scenario_truth
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
    "NavigationMetrics",
    "ScenarioConfig",
    "ScenarioFrame",
    "SyntheticScenario",
    "TrackEstimate",
    "TrackingMetrics",
    "TrackingMetricsAccumulator",
    "TruthTarget",
    "crossing_scenario",
    "dense_multi_target_scenario",
    "evaluate_navigation_scenario",
    "evaluate_tracking_sequence",
    "head_on_scenario",
    "plot_benchmark_summary",
    "plot_scenario_truth",
    "run_monte_carlo",
    "run_tracking_once",
    "truth_frames",
    "write_latex_table",
    "write_summary_csv",
    "write_summary_json",
]
