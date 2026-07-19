from __future__ import annotations

import numpy as np
import pytest

from iscir.sensing.range_doppler import (
    ChirpSequenceConfig,
    detect_range_doppler_peaks,
    range_doppler_map,
    simulate_chirp_sequence,
)
from iscir.sensing.signal import FMCWConfig, PointTarget


def test_range_doppler_recovers_multiple_targets() -> None:
    config = FMCWConfig()
    sequence = ChirpSequenceConfig(num_chirps=64, chirp_repetition_interval_s=60e-6)
    targets = [
        PointTarget(range_m=4.0, radial_velocity_mps=1.0, amplitude=1.0),
        PointTarget(range_m=8.0, radial_velocity_mps=-1.2, amplitude=0.8),
    ]

    signal = simulate_chirp_sequence(config, sequence, targets, snr_db=35.0, rng_seed=2)
    ranges_m, velocities_mps, magnitude = range_doppler_map(
        signal,
        config,
        sequence,
        range_fft_size=4096,
        doppler_fft_size=128,
    )
    detections = detect_range_doppler_peaks(
        magnitude,
        ranges_m,
        velocities_mps,
        threshold=0.3,
        maximum_detections=2,
        range_guard_bins=8,
        doppler_guard_bins=5,
    )

    assert len(detections) == 2
    assert detections[0].range_m == pytest.approx(4.0, abs=0.12)
    assert detections[0].radial_velocity_mps == pytest.approx(1.0, abs=0.3)
    assert detections[1].range_m == pytest.approx(8.0, abs=0.12)
    assert detections[1].radial_velocity_mps == pytest.approx(-1.2, abs=0.3)


def test_range_doppler_map_is_normalized() -> None:
    config = FMCWConfig()
    sequence = ChirpSequenceConfig(num_chirps=16)
    signal = simulate_chirp_sequence(config, sequence, [PointTarget(range_m=5.0)])
    _, _, magnitude = range_doppler_map(signal, config, sequence)

    assert magnitude.ndim == 2
    assert float(np.max(magnitude)) == pytest.approx(1.0)
    assert np.all(magnitude >= 0.0)


def test_invalid_sequence_configuration() -> None:
    with pytest.raises(ValueError):
        ChirpSequenceConfig(num_chirps=1)
