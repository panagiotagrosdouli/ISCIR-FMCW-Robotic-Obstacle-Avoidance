"""Multi-target processing utilities for one-dimensional FMCW range spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signal import FMCWConfig, range_spectrum


@dataclass(frozen=True, slots=True)
class RangeDetection:
    """A detected peak in a one-dimensional FMCW range spectrum."""

    range_m: float
    normalized_magnitude: float
    bin_index: int


def detect_range_peaks(
    signal: np.ndarray,
    config: FMCWConfig,
    *,
    n_fft: int = 8192,
    threshold: float = 0.2,
    minimum_separation_m: float | None = None,
    maximum_detections: int | None = None,
) -> list[RangeDetection]:
    """Detect multiple local maxima in the positive-frequency range spectrum.

    The method intentionally uses a deterministic amplitude threshold and
    non-maximum suppression. It is the project baseline before CFAR is added.
    Peaks are returned in ascending range order.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if n_fft < 2:
        raise ValueError("n_fft must be at least 2")
    if minimum_separation_m is not None and minimum_separation_m < 0:
        raise ValueError("minimum_separation_m cannot be negative")
    if maximum_detections is not None and maximum_detections < 1:
        raise ValueError("maximum_detections must be positive")

    ranges_m, magnitude = range_spectrum(signal, config, n_fft=n_fft)
    if magnitude.size < 3 or not np.any(magnitude):
        return []

    local_maximum = (
        (magnitude[1:-1] > magnitude[:-2])
        & (magnitude[1:-1] >= magnitude[2:])
        & (magnitude[1:-1] >= threshold)
    )
    candidate_indices = np.flatnonzero(local_maximum) + 1

    if candidate_indices.size == 0:
        return []

    separation_m = (
        config.range_resolution_m
        if minimum_separation_m is None
        else minimum_separation_m
    )

    # Keep the strongest candidates first, suppressing neighbours that are too close.
    ranked_indices = sorted(
        candidate_indices.tolist(), key=lambda index: float(magnitude[index]), reverse=True
    )
    selected: list[int] = []
    for index in ranked_indices:
        candidate_range = float(ranges_m[index])
        if any(abs(candidate_range - float(ranges_m[kept])) < separation_m for kept in selected):
            continue
        selected.append(index)
        if maximum_detections is not None and len(selected) >= maximum_detections:
            break

    selected.sort(key=lambda index: float(ranges_m[index]))
    return [
        RangeDetection(
            range_m=float(ranges_m[index]),
            normalized_magnitude=float(magnitude[index]),
            bin_index=int(index),
        )
        for index in selected
    ]
