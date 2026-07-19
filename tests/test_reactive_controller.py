from math import inf

import pytest

from iscir.navigation.collision import CollisionLevel, CollisionPrediction
from iscir.navigation.reactive_controller import (
    ReactiveControllerConfig,
    safe_velocity_command,
)


def prediction(
    track_id: int,
    level: CollisionLevel,
    risk_score: float,
) -> CollisionPrediction:
    return CollisionPrediction(
        track_id=track_id,
        level=level,
        collision_predicted=level in {CollisionLevel.DANGER, CollisionLevel.EMERGENCY},
        risk_score=risk_score,
        time_to_collision_s=inf,
        range_m=5.0,
        closing_speed_mps=0.0,
    )


def test_no_predictions_returns_cruise_speed() -> None:
    command = safe_velocity_command([])

    assert command.linear_speed_mps == pytest.approx(1.0)
    assert command.level is CollisionLevel.CLEAR
    assert command.source_track_id is None
    assert not command.emergency_stop


def test_clear_prediction_keeps_cruise_speed() -> None:
    command = safe_velocity_command(
        [prediction(3, CollisionLevel.CLEAR, 0.1)],
        ReactiveControllerConfig(cruise_speed_mps=1.4),
    )

    assert command.linear_speed_mps == pytest.approx(1.4)
    assert command.source_track_id == 3


def test_caution_reduces_speed() -> None:
    config = ReactiveControllerConfig(cruise_speed_mps=1.0)
    command = safe_velocity_command(
        [prediction(4, CollisionLevel.CAUTION, 0.4)],
        config,
    )

    assert 0.65 < command.linear_speed_mps < 1.0
    assert command.level is CollisionLevel.CAUTION


def test_danger_is_slower_than_caution() -> None:
    caution = safe_velocity_command(
        [prediction(1, CollisionLevel.CAUTION, 0.6)]
    )
    danger = safe_velocity_command(
        [prediction(2, CollisionLevel.DANGER, 0.6)]
    )

    assert danger.linear_speed_mps < caution.linear_speed_mps


def test_emergency_stops_robot() -> None:
    command = safe_velocity_command(
        [prediction(9, CollisionLevel.EMERGENCY, 0.9)]
    )

    assert command.linear_speed_mps == pytest.approx(0.0)
    assert command.level is CollisionLevel.EMERGENCY
    assert command.source_track_id == 9
    assert command.emergency_stop


def test_most_conservative_track_controls_command() -> None:
    command = safe_velocity_command(
        [
            prediction(1, CollisionLevel.CLEAR, 0.1),
            prediction(7, CollisionLevel.DANGER, 0.7),
            prediction(3, CollisionLevel.CAUTION, 0.4),
        ]
    )

    assert command.level is CollisionLevel.DANGER
    assert command.source_track_id == 7


def test_invalid_speed_fraction_order_is_rejected() -> None:
    with pytest.raises(ValueError):
        ReactiveControllerConfig(
            caution_speed_fraction=0.2,
            danger_speed_fraction=0.5,
        )
