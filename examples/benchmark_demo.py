"""Run reproducible synthetic tracking benchmarks and export paper outputs."""

from pathlib import Path

from iscir.evaluation import (
    ScenarioConfig,
    crossing_scenario,
    dense_multi_target_scenario,
    head_on_scenario,
    plot_benchmark_summary,
    plot_scenario_truth,
    run_monte_carlo,
    write_latex_table,
    write_summary_csv,
    write_summary_json,
)


def main() -> None:
    output_dir = Path("paper/experiments/generated")
    figure_dir = Path("paper/figures/generated")
    config = ScenarioConfig(
        frame_count=100,
        detection_probability=0.92,
        range_noise_std_m=0.25,
        velocity_noise_std_mps=0.12,
        clutter_rate_per_frame=0.5,
    )
    scenarios = (
        head_on_scenario(config),
        crossing_scenario(config),
        dense_multi_target_scenario(5, config),
    )

    summaries = []
    for scenario in scenarios:
        result = run_monte_carlo(scenario, repetitions=25)
        summary = result.summary
        summaries.append(summary)
        write_summary_json(summary, output_dir / f"{scenario.name}.json")
        plot_scenario_truth(
            scenario,
            figure_dir / f"{scenario.name}_truth.png",
        )
        print(
            f"{scenario.name:12s} "
            f"Pd={summary.detection_probability_mean:.3f} "
            f"range_RMSE={summary.range_rmse_mean_m:.3f} m "
            f"velocity_RMSE={summary.velocity_rmse_mean_mps:.3f} m/s "
            f"continuity={summary.track_continuity_mean:.3f} "
            f"latency={summary.mean_frame_latency_ms:.3f} ms"
        )

    write_summary_csv(summaries, output_dir / "tracking_benchmark.csv")
    write_latex_table(summaries, output_dir / "tracking_benchmark.tex")
    plot_benchmark_summary(summaries, figure_dir)
    print(f"Tables and data written to {output_dir}")
    print(f"Figures written to {figure_dir}")


if __name__ == "__main__":
    main()
