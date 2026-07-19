"""Demonstrate multi-target range and radial-velocity estimation."""

from __future__ import annotations

import matplotlib.pyplot as plt

from iscir.sensing.range_doppler import (
    ChirpSequenceConfig,
    detect_range_doppler_peaks,
    range_doppler_map,
    simulate_chirp_sequence,
)
from iscir.sensing.signal import FMCWConfig, PointTarget


def main() -> None:
    config = FMCWConfig()
    sequence = ChirpSequenceConfig(num_chirps=64, chirp_repetition_interval_s=60e-6)
    targets = [
        PointTarget(range_m=4.0, radial_velocity_mps=1.0, amplitude=1.0),
        PointTarget(range_m=7.5, radial_velocity_mps=-1.5, amplitude=0.8),
        PointTarget(range_m=11.0, radial_velocity_mps=0.3, amplitude=0.7),
    ]

    signal = simulate_chirp_sequence(config, sequence, targets, snr_db=25.0, rng_seed=4)
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
        threshold=0.25,
        maximum_detections=len(targets),
        range_guard_bins=8,
        doppler_guard_bins=5,
    )

    print("True targets:")
    for target in targets:
        print(f"  range={target.range_m:5.2f} m, velocity={target.radial_velocity_mps:5.2f} m/s")

    print("Detected targets:")
    for detection in detections:
        print(
            f"  range={detection.range_m:5.2f} m, "
            f"velocity={detection.radial_velocity_mps:5.2f} m/s, "
            f"magnitude={detection.normalized_magnitude:.2f}"
        )

    plt.imshow(
        20.0 * __import__("numpy").log10(magnitude + 1e-6),
        origin="lower",
        aspect="auto",
        extent=[velocities_mps[0], velocities_mps[-1], ranges_m[0], ranges_m[-1]],
    )
    plt.ylim(0.0, 15.0)
    plt.xlabel("radial velocity [m/s]")
    plt.ylabel("range [m]")
    plt.title("Synthetic FMCW Range-Doppler Map")
    plt.colorbar(label="normalized magnitude [dB]")
    plt.show()


if __name__ == "__main__":
    main()
