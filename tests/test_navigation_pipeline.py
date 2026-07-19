from iscir.navigation import CollisionLevel
from iscir.navigation.pipeline import RadarNavigationPipeline
from iscir.sensing.tracker import RadarDetection


def test_pipeline_reduces_speed_for_approaching_confirmed_track() -> None:
    pipeline = RadarNavigationPipeline()

    result = None
    for range_m in (4.0, 3.4, 2.8, 2.2):
        result = pipeline.update(
            [RadarDetection(range_m, -1.0, confidence=1.0)],
            dt_s=0.5,
        )

    assert result is not None
    assert result.risks
    assert result.collisions
    assert result.command.source_track_id == result.risks[0].track_id
    assert result.command.linear_speed_mps < 1.0
    assert result.command.level in {
        CollisionLevel.CAUTION,
        CollisionLevel.DANGER,
        CollisionLevel.EMERGENCY,
    }


def test_pipeline_cruises_without_confirmed_tracks() -> None:
    pipeline = RadarNavigationPipeline()

    result = pipeline.update([], dt_s=0.1)

    assert result.tracks == ()
    assert result.risks == ()
    assert result.collisions == ()
    assert result.command.level is CollisionLevel.CLEAR
    assert result.command.linear_speed_mps == 1.0
    assert result.command.source_track_id is None
