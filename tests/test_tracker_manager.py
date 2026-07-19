from __future__ import annotations

import pytest

from iscir.sensing.tracker.assignment import RadarDetection
from iscir.sensing.tracker.manager import MultiTargetTracker, TrackerManagerConfig
from iscir.sensing.tracker.track import TrackLifecycleConfig


def test_unmatched_detections_create_persistent_tracks() -> None:
    tracker = MultiTargetTracker(first_track_id=10)

    tracks = tracker.update(
        [
            RadarDetection(5.0, -0.5, confidence=0.9),
            RadarDetection(12.0, 1.0, confidence=0.8),
        ],
        dt_s=0.1,
    )

    assert [track.track_id for track in tracks] == [10, 11]
    assert [track.range_m for track in tracks] == pytest.approx([5.0, 12.0])


def test_associated_detection_updates_existing_track() -> None:
    tracker = MultiTargetTracker()
    original = tracker.update([RadarDetection(10.0, -1.0)], dt_s=0.1)[0]

    tracks = tracker.update([RadarDetection(9.8, -1.0)], dt_s=0.1)

    assert len(tracks) == 1
    assert tracks[0] is original
    assert tracks[0].track_id == 0
    assert tracks[0].hits == 2
    assert tracks[0].missed_updates == 0


def test_tracks_confirm_and_stale_tracks_are_deleted() -> None:
    tracker = MultiTargetTracker(
        lifecycle_config=TrackLifecycleConfig(
            confirmation_hits=2,
            maximum_missed_updates=1,
        )
    )

    tracker.update([RadarDetection(6.0, 0.0)], dt_s=0.1)
    tracker.update([RadarDetection(6.0, 0.0)], dt_s=0.1)
    assert [track.track_id for track in tracker.confirmed_tracks] == [0]

    tracker.update([], dt_s=0.1)
    assert len(tracker.tracks) == 1
    tracker.update([], dt_s=0.1)
    assert tracker.tracks == ()


def test_far_detection_starts_new_track_and_old_track_is_missed() -> None:
    tracker = MultiTargetTracker(
        manager_config=TrackerManagerConfig(gate_threshold_squared=5.99)
    )
    tracker.update([RadarDetection(2.0, 0.0)], dt_s=0.1)

    tracks = tracker.update([RadarDetection(20.0, 0.0)], dt_s=0.1)

    assert [track.track_id for track in tracks] == [0, 1]
    assert tracks[0].missed_updates == 1
    assert tracks[1].hits == 1


def test_invalid_configuration_and_timestep_are_rejected() -> None:
    with pytest.raises(ValueError):
        TrackerManagerConfig(gate_threshold_squared=0.0)
    with pytest.raises(ValueError):
        MultiTargetTracker(first_track_id=-1)

    tracker = MultiTargetTracker()
    with pytest.raises(ValueError):
        tracker.update([], dt_s=0.0)
