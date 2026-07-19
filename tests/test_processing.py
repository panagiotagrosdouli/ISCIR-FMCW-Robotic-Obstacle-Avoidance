from __future__ import annotations

import numpy as np
import pytest

from iscir.sensing.processing import detect_range_peaks
from iscir.sensing.signal import FMCWConfig, PointTarget, simulate_beat_signal


def test_detects_multiple_targets() -> None:
    config = FMCWConfig()
    true_ranges = [2.5, 5.0, 8.5]
    targets = [PointTarget(range_m=value, amplitude=1.0 - 0.15 * index) for index, value in enumerate(true_ranges)]
    _, signal = simulate_beat_signal(config, targets, snr_db=25.0, rng_seed=3)

    detections = detect_range_peaks(
        signal,
        config,
        n_fft=8192,
        threshold=0.15,
        minimum_separation_m=0.3,
        maximum_detections=3,
    )

    detected_ranges = [item.range_m for item in detections]
    assert len(detected_ranges) == len(true_ranges)
    for expected, measured in zip(true_ranges, detected_ranges, strict=True):
        assert measured == pytest.approx(expected, abs=0.08)


def test_returns_empty_list_for_zero_signal() -> None:
    config = FMCWConfig()
    signal = np.zeros(config.num_samples, dtype=np.complex128)
    assert detect_range_peaks(signal, config) == []


def test_rejects_invalid_threshold() -> None:
    config = FMCWConfig()
    signal = np.ones(config.num_samples, dtype=np.complex128)
    with pytest.raises(ValueError, match="threshold"):
        detect_range_peaks(signal, config, threshold=1.1)
