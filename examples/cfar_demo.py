"""Demonstrate adaptive 2D CA-CFAR detection on a synthetic FMCW scene."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from iscir.sensing.cfar import CACFARConfig, detect_cfar_peaks
from iscir.sensing.range_doppler import (
    ChirpSequenceConfig,
    range_doppler_map,
    simulate_chirp_sequence,
)
from iscir.sensing.signal import FMCWConfig, PointTarget


def main() -> None:
    radar = FMCWConfig()
    sequence = ChirpSequenceConfig(num_chirps=64, chirp_repetition_interval_s=60e-6)
    targets = [
        PointTarget(range_m=4.0, radial_velocity_mps=1.0, amplitude=1.0),
        PointTarget(range_m=8.0, radial_velocity_mps=-1.2, amplitude=0.75),
        PointTarget(range_m=11.0, radial_velocity_mps=0.35, amplitude=0.55),
    ]

    beat_signal = simulate_chirp_sequence(
        radar,
        sequence,
        targets,
        snr_db=22.0,
        rng_seed=8,
    )
    ranges_m, velocities_mps, magnitude = range_doppler_map(
        beat_signal,
        radar,
        sequence,
        range_fft_size=4096,
        doppler_fft_size=128,
    )

    detections, cfar = detect_cfar_peaks(
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
        maximum_detections=3,
        suppression_range_bins=10,
        suppression_doppler_bins=6,
    )

    print("CA-CFAR detections")
    for detection in detections:
        print(
            f"range={detection.range_m:6.2f} m, "
            f"velocity={detection.radial_velocity_mps:6.2f} m/s, "
            f"magnitude={detection.normalized_magnitude:.3f}"
        )

    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-8))
    extent = [
        float(velocities_mps[0]),
        float(velocities_mps[-1]),
        float(ranges_m[-1]),
        float(ranges_m[0]),
    ]

    plt.figure(figsize=(10, 6))
    plt.imshow(magnitude_db, aspect="auto", extent=extent)
    plt.colorbar(label="Normalized magnitude (dB)")
    for detection in detections:
        plt.scatter(
            detection.radial_velocity_mps,
            detection.range_m,
            marker="x",
            s=90,
            linewidths=2,
        )
    plt.xlabel("Radial velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title(
        "Range-Doppler map with 2D CA-CFAR detections "
        f"({int(np.count_nonzero(cfar.detection_mask))} detected cells)"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
