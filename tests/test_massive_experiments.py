from __future__ import annotations

import json

import pytest

from iscir.evaluation.massive_experiments import (
    CampaignAccumulator,
    ExperimentCondition,
    MassiveCampaignConfig,
    OnlineStatistics,
    default_condition_grid,
    iter_trial_specs,
)


def test_online_statistics_matches_known_population_values() -> None:
    stats = OnlineStatistics()
    for value in (1.0, 2.0, 3.0):
        stats.update(value)
    assert stats.count == 3
    assert stats.mean == pytest.approx(2.0)
    assert stats.std == pytest.approx((2.0 / 3.0) ** 0.5)


def test_online_statistics_round_trip() -> None:
    stats = OnlineStatistics()
    stats.update(4.0)
    restored = OnlineStatistics.from_dict(stats.to_dict())
    assert restored.count == 1
    assert restored.mean == pytest.approx(4.0)


def test_trial_specs_cycle_conditions_and_increment_seeds() -> None:
    conditions = (
        ExperimentCondition("head_on", 0.9, 0.1, 0.1, 0.0),
        ExperimentCondition("crossing", 0.8, 0.2, 0.1, 0.5, target_count=2),
    )
    specs = list(iter_trial_specs(conditions, 5, first_seed=10))
    assert [condition.scenario_name for condition, _ in specs] == [
        "head_on", "crossing", "head_on", "crossing", "head_on"
    ]
    assert [seed for _, seed in specs] == [10, 11, 12, 13, 14]


def test_default_grid_is_broad() -> None:
    grid = default_condition_grid()
    assert len(grid) > 1000
    assert {condition.scenario_name for condition in grid} == {"head_on", "crossing", "dense"}


def test_campaign_config_defaults_to_one_million_trials() -> None:
    assert MassiveCampaignConfig().total_trials == 1_000_000


def test_accumulator_serialization() -> None:
    accumulator = CampaignAccumulator.empty()
    accumulator.update({
        "detection_probability": 0.9,
        "range_rmse_m": 0.2,
        "velocity_rmse_mps": 0.1,
        "track_continuity": 0.8,
        "false_track_count": 2,
        "id_switches": 1,
        "mean_frame_latency_ms": 0.5,
    })
    restored = CampaignAccumulator.from_dict(
        json.loads(json.dumps(accumulator.to_dict()))
    )
    assert restored.detection_probability.mean == pytest.approx(0.9)
    assert restored.false_track_count.mean == pytest.approx(2.0)
