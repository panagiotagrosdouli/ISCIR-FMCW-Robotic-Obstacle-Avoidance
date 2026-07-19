from math import inf

import pytest

from iscir.navigation.collision import (
    CollisionConfig,
    CollisionLevel,
    predict_collision,
    predict_collisions,
)
from iscir.navigation.risk import TrackRisk



def make_risk(
    *,
    track_id: int = 0,
    score: float = 0.0,
    ttc_s: float = inf,
    closing_speed_mps: float = 0.0,
) -> TrackRisk:
    return TrackRisk(
        track_id=track_id,
        range_m=5.0,
        radial_velocity_mps=-closing_speed_mps,
        closing_speed_mps=closing_speed_mps,
        time_to_collision_s=ttc_s,
        risk_score=score,
        confidence=1.0,
    )



def test_clear_track_does_not_predict_collision() -> None:
    prediction = predict_collision(make_risk(score=0.1))

    assert prediction.level is CollisionLevel.CLEAR
    assert not prediction.collision_predicted



def test_risk_thresholds_map_to_warning_levels() -> None:
    assert predict_collision(make_risk(score=0.30)).level is CollisionLevel.CAUTION
    assert predict_collision(make_risk(score=0.60)).level is CollisionLevel.DANGER
    assert predict_collision(make_risk(score=0.90)).level is CollisionLevel.EMERGENCY



def test_short_ttc_for_approaching_track_forces_emergency() -> None:
    prediction = predict_collision(
        make_risk(score=0.2, ttc_s=0.5, closing_speed_mps=2.0)
    )

    assert prediction.level is CollisionLevel.EMERGENCY
    assert prediction.collision_predicted



def test_short_ttc_without_closing_speed_does_not_force_emergency() -> None:
    prediction = predict_collision(make_risk(score=0.2, ttc_s=0.5))

    assert prediction.level is CollisionLevel.CLEAR



def test_predictions_are_sorted_by_urgency_then_ttc() -> None:
    predictions = predict_collisions(
        [
            make_risk(track_id=1, score=0.6, ttc_s=3.0, closing_speed_mps=1.0),
            make_risk(track_id=2, score=0.9, ttc_s=2.0, closing_speed_mps=1.0),
            make_risk(track_id=3, score=0.9, ttc_s=0.8, closing_speed_mps=1.0),
        ]
    )

    assert [prediction.track_id for prediction in predictions] == [3, 2, 1]



def test_collision_config_rejects_invalid_threshold_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CollisionConfig(
            caution_risk_threshold=0.6,
            danger_risk_threshold=0.5,
            emergency_risk_threshold=0.8,
        )
