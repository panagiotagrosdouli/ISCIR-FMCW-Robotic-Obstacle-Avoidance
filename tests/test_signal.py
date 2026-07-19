import numpy as np
import pytest

from iscir.sensing.signal import (
    FMCWConfig,
    PointTarget,
    estimate_range_m,
    simulate_beat_signal,
)


def test_range_estimation_for_stationary_target() -> None:
    config = FMCWConfig(
        carrier_frequency_hz=77e9,
        bandwidth_hz=1e9,
        chirp_duration_s=50e-6,
        sample_rate_hz=10e6,
    )
    true_range_m = 8.0
    _, signal = simulate_beat_signal(
        config,
        [PointTarget(range_m=true_range_m)],
        snr_db=30.0,
        rng_seed=1,
    )

    estimate = estimate_range_m(signal, config)

    assert estimate == pytest.approx(true_range_m, abs=0.05)


def test_zero_signal_cannot_produce_range_estimate() -> None:
    config = FMCWConfig()
    signal = np.zeros(config.num_samples, dtype=np.complex128)

    with pytest.raises(ValueError, match="cannot estimate range"):
        estimate_range_m(signal, config)


def test_invalid_bandwidth_is_rejected() -> None:
    with pytest.raises(ValueError, match="bandwidth_hz"):
        FMCWConfig(bandwidth_hz=0.0)
