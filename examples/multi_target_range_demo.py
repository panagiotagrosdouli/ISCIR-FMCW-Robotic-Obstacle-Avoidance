"""Demonstrate multi-target FMCW range estimation from one combined beat signal."""

from __future__ import annotations

import matplotlib.pyplot as plt

from iscir.sensing.processing import detect_range_peaks
from iscir.sensing.signal import FMCWConfig, PointTarget, range_spectrum, simulate_beat_signal


def main() -> None:
    config = FMCWConfig()
    targets = [
        PointTarget(range_m=3.0, amplitude=1.0, phase_rad=0.2),
        PointTarget(range_m=6.5, amplitude=0.75, phase_rad=1.1),
        PointTarget(range_m=10.0, amplitude=0.55, phase_rad=-0.8),
    ]

    _, beat_signal = simulate_beat_signal(config, targets, snr_db=18.0, rng_seed=11)
    ranges_m, magnitude = range_spectrum(beat_signal, config, n_fft=8192)
    detections = detect_range_peaks(
        beat_signal,
        config,
        n_fft=8192,
        threshold=0.18,
        minimum_separation_m=0.35,
        maximum_detections=8,
    )

    print("True target ranges:", [target.range_m for target in targets])
    print("Detected target ranges:", [round(item.range_m, 3) for item in detections])

    plt.plot(ranges_m, magnitude, label="normalized range spectrum")
    if detections:
        plt.scatter(
            [item.range_m for item in detections],
            [item.normalized_magnitude for item in detections],
            marker="x",
            s=70,
            label="detected peaks",
        )
    plt.xlim(0.0, 14.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Range [m]")
    plt.ylabel("Normalized magnitude")
    plt.title("Multi-target FMCW range detection")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
