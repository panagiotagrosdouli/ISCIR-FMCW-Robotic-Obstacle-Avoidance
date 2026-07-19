from __future__ import annotations

import numpy as np
import pytest

from iscir.sensing.cfar import CACFARConfig, ca_cfar_2d, detect_cfar_peaks
from iscir.sensing.range_doppler import (
    ChirpSequenceConfig,
    range_doppler_map,
    simulate_chirp_sequence,
)
from iscir.sensing.signal import FMCWConfig, PointTarget


def test_ca_cfar_detects_isolated_target() -> None:
    magnitude = np.ones((31, 25), dtype=float)
    magnitude[15, 12] = 20.0
    config = CACFARConfig(
        training_range_cells=4,
        training_doppler_cells=3,
        guard_range_cells=1,
        guard_doppler_cells=1,
        false_alarm_probability=1e-3,
    )

    result = ca_cfar_2d(magnitude, config)

    assert result.detection_mask[15, 12]
    assert result.valid_cell_mask[15, 12]
    assert not np.any(result.valid_cell_mask[:5, :])
    assert np.isnan(result.threshold_map[0, 0])


def test_ca_cfar_rejects_uniform_background() -> None:
    magnitude = np.ones((31, 25), dtype=float)
    result = ca_cfar_2d(
        magnitude,
        CACFARConfig(false_alarm_probability=1e-4),
    )

    assert not np.any(result.detection_mask)


def test_cfar_recovers_multiple_fmcw_targets() -> None:
    radar = FMCWConfig()
    sequence = ChirpSequenceConfig(num_chirps=64, chirp_repetition_interval_s=60e-6)
    targets = [
        PointTarget(range_m=4.0, radial_velocity_mps=1.0, amplitude=1.0),
        PointTarget(range_m=8.0, radial_velocity_mps=-1.2, amplitude=0.8),
    ]
    signal = simulate_chirp_sequence(radar, sequence, targets, snr_db=30.0, rng_seed=4)
    ranges_m, velocities_mps, magnitude = range_doppler_map(
        signal,
        radar,
        sequence,
        range_fft_size=4096,
        doppler_fft_size=128,
    )

    detections, result = detect_cfar_peaks(
        magnitude,
        ranges_m,
        velocities_mps,
        CACFARConfig(
            training_range_cells=12,
            training_doppler_cells=8,
            guard_range_cells=6,
            guard_doppler_cells=3,
            false_alarm_probability=1e-5,
        ),
        maximum_detections=2,
        suppression_range_bins=10,
        suppression_doppler_bins=6,
    )

    assert result.detection_mask.shape == magnitude.shape
    assert len(detections) == 2
    assert detections[0].range_m == pytest.approx(4.0, abs=0.15)
    assert detections[0].radial_velocity_mps == pytest.approx(1.0, abs=0.35)
    assert detections[1].range_m == pytest.approx(8.0, abs=0.15)
    assert detections[1].radial_velocity_mps == pytest.approx(-1.2, abs=0.35)


def test_invalid_cfar_configuration_and_input() -> None:
    with pytest.raises(ValueError):
        CACFARConfig(training_range_cells=0)
    with pytest.raises(ValueError):
        CACFARConfig(false_alarm_probability=1.0)
    with pytest.raises(ValueError):
        ca_cfar_2d(np.ones((3, 3)), CACFARConfig())
    with pytest.raises(ValueError):
        ca_cfar_2d(np.array([[1.0, -1.0], [1.0, 1.0]]))
