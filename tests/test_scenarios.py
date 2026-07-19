from iscir.evaluation.scenarios import (
    MovingTarget,
    ScenarioConfig,
    SyntheticScenario,
    crossing_scenario,
    head_on_scenario,
)


def test_moving_target_state_progresses_with_constant_velocity() -> None:
    target = MovingTarget(3, initial_range_m=10.0, radial_velocity_mps=-2.0)

    state = target.state_at(1.5)

    assert state is not None
    assert state.object_id == 3
    assert state.range_m == 7.0
    assert state.radial_velocity_mps == -2.0


def test_scenario_generation_is_reproducible() -> None:
    config = ScenarioConfig(
        frame_count=6,
        detection_probability=0.8,
        clutter_rate_per_frame=1.0,
    )
    scenario = crossing_scenario(config)

    first = scenario.generate(seed=42)
    second = scenario.generate(seed=42)

    assert first == second


def test_perfect_head_on_scenario_has_one_detection_per_frame() -> None:
    config = ScenarioConfig(
        dt_s=0.1,
        frame_count=8,
        detection_probability=1.0,
        range_noise_std_m=0.0,
        velocity_noise_std_mps=0.0,
        clutter_rate_per_frame=0.0,
    )

    frames = head_on_scenario(config).generate(seed=1)

    assert len(frames) == 8
    assert all(len(frame.truths) == 1 for frame in frames)
    assert all(len(frame.detections) == 1 for frame in frames)
    assert frames[1].truths[0].range_m < frames[0].truths[0].range_m


def test_duplicate_object_ids_are_rejected() -> None:
    targets = (
        MovingTarget(0, initial_range_m=5.0, radial_velocity_mps=0.0),
        MovingTarget(0, initial_range_m=8.0, radial_velocity_mps=0.0),
    )

    try:
        SyntheticScenario("duplicate", targets)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate target IDs should fail")
