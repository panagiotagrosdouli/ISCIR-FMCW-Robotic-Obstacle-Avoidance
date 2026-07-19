from math import pi

import pytest

from iscir.sensing.fft_sensor import FFTFMCWSensor
from iscir.sensing.signal import FMCWConfig
from iscir.simulation.models import Obstacle, RobotState, Scene


def test_fft_sensor_estimates_visible_obstacle_range() -> None:
    scene = Scene(
        robot=RobotState(),
        goal=(10.0, 0.0),
        obstacles=[Obstacle(x=5.0, y=0.0)],
    )
    sensor = FFTFMCWSensor(snr_db=30.0, rng_seed=3)

    detections = sensor.scan(scene)

    assert len(detections) == 1
    assert detections[0].range_m == pytest.approx(5.0, abs=0.08)
    assert detections[0].bearing_rad == pytest.approx(0.0)


def test_fft_sensor_respects_field_of_view() -> None:
    scene = Scene(
        robot=RobotState(),
        goal=(10.0, 0.0),
        obstacles=[Obstacle(x=0.0, y=5.0)],
    )
    sensor = FFTFMCWSensor(field_of_view_rad=pi / 2)

    assert sensor.scan(scene) == []


def test_fft_sensor_rejects_too_small_fft() -> None:
    config = FMCWConfig()

    with pytest.raises(ValueError, match="n_fft"):
        FFTFMCWSensor(config=config, n_fft=config.num_samples - 1)
