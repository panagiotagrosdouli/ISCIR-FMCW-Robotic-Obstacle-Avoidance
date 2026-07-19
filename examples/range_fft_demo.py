"""Demonstrate synthetic FMCW beat-signal generation and range estimation."""

import matplotlib.pyplot as plt

from iscir.sensing.signal import (
    FMCWConfig,
    PointTarget,
    estimate_range_m,
    range_spectrum,
    simulate_beat_signal,
)


def main() -> None:
    config = FMCWConfig(
        carrier_frequency_hz=77e9,
        bandwidth_hz=1e9,
        chirp_duration_s=50e-6,
        sample_rate_hz=10e6,
    )
    true_range_m = 8.0
    target = PointTarget(range_m=true_range_m, amplitude=1.0)

    _, beat_signal = simulate_beat_signal(config, [target], snr_db=15.0)
    estimated_range_m = estimate_range_m(beat_signal, config)
    ranges_m, magnitude = range_spectrum(beat_signal, config, n_fft=8192)

    print(f"Theoretical range resolution: {config.range_resolution_m:.3f} m")
    print(f"Maximum unambiguous range: {config.maximum_unambiguous_range_m:.2f} m")
    print(f"True target range: {true_range_m:.3f} m")
    print(f"Estimated target range: {estimated_range_m:.3f} m")
    print(f"Absolute error: {abs(estimated_range_m - true_range_m):.3f} m")

    plt.plot(ranges_m, magnitude)
    plt.xlim(0.0, 15.0)
    plt.xlabel("Range (m)")
    plt.ylabel("Normalized magnitude")
    plt.title("Synthetic FMCW range spectrum")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
