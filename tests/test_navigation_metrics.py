from iscir.evaluation import ScenarioConfig, evaluate_navigation_scenario, head_on_scenario


def test_navigation_metrics_cover_complete_scenario() -> None:
    scenario = head_on_scenario(
        ScenarioConfig(
            frame_count=30,
            detection_probability=1.0,
            range_noise_std_m=0.0,
            velocity_noise_std_mps=0.0,
            clutter_rate_per_frame=0.0,
        )
    )

    metrics = evaluate_navigation_scenario(scenario, seed=7)

    assert metrics.frame_count == 30
    assert 0.0 <= metrics.reduced_speed_fraction <= 1.0
    assert metrics.minimum_command_speed_mps <= metrics.mean_command_speed_mps
    assert metrics.mean_pipeline_latency_ms >= 0.0
    assert metrics.maximum_pipeline_latency_ms >= metrics.mean_pipeline_latency_ms


def test_navigation_metrics_are_reproducible_for_same_seed() -> None:
    scenario = head_on_scenario(ScenarioConfig(frame_count=20))

    first = evaluate_navigation_scenario(scenario, seed=11)
    second = evaluate_navigation_scenario(scenario, seed=11)

    assert first.frame_count == second.frame_count
    assert first.mean_command_speed_mps == second.mean_command_speed_mps
    assert first.minimum_command_speed_mps == second.minimum_command_speed_mps
    assert first.reduced_speed_fraction == second.reduced_speed_fraction
    assert first.emergency_stop_count == second.emergency_stop_count
    assert first.first_intervention_frame == second.first_intervention_frame
