"""Tests for collision-risk estimation from radar tracks."""

from math import isinf

import pytest

from iscir.navigation.risk import RiskConfig, estimate_track_risk, rank_track_risks
from iscir.sensing.tracker import ConstantVelocityKalmanFilter, Track



def make_track(track_id: int, range_m: float, velocity_mps: float, confidence: float = 1.0) -> Track:
    return Track(
        track_id,
        ConstantVelocityKalmanFilter(range_m, velocity_mps),
        initial_confidence=confidence,
    )



def test_approaching_track_has_finite_time_to_collision() -> None:
    risk = estimate_track_risk(make_track(4, 6.0, -2.0))

    assert risk.track_id == 4
    assert risk.closing_speed_mps == pytest.approx(2.0)
    assert risk.time_to_collision_s == pytest.approx(2.5)
    assert 0.0 < risk.risk_score <= 1.0
    assert risk.is_approaching



def test_receding_track_has_no_time_to_collision() -> None:
    risk = estimate_track_risk(make_track(1, 6.0, 1.0))

    assert risk.closing_speed_mps == 0.0
    assert isinf(risk.time_to_collision_s)
    assert not risk.is_approaching



def test_near_track_is_riskier_than_far_track() -> None:
    near = make_track(0, 2.0, -0.5)
    far = make_track(1, 9.0, -0.5)

    ranked = rank_track_risks([far, near])

    assert [risk.track_id for risk in ranked] == [0, 1]
    assert ranked[0].risk_score > ranked[1].risk_score



def test_low_confidence_track_can_be_suppressed() -> None:
    config = RiskConfig(minimum_confidence=0.5)
    risk = estimate_track_risk(make_track(2, 1.5, -2.0, confidence=0.2), config)

    assert risk.risk_score == 0.0



def test_risk_config_validates_horizons() -> None:
    with pytest.raises(ValueError):
        RiskConfig(safety_distance_m=2.0, distance_horizon_m=2.0)

    with pytest.raises(ValueError):
        RiskConfig(ttc_horizon_s=0.0)
