"""Synthetic FMCW chirp sequences and two-dimensional range-Doppler processing."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .signal import FMCWConfig, PointTarget, SPEED_OF_LIGHT_MPS


@dataclass(frozen=True, slots=True)
class ChirpSequenceConfig:
    """Slow-time configuration for a coherent FMCW processing interval."""

    num_chirps: int = 64
    chirp_repetition_interval_s: float = 60e-6

    def __post_init__(self) -> None:
        if self.num_chirps < 2:
            raise ValueError("num_chirps must be at least 2")
        if self.chirp_repetition_interval_s <= 0:
            raise ValueError("chirp_repetition_interval_s must be positive")


@dataclass(frozen=True, slots=True)
class RangeDopplerDetection:
    """Peak detected in a range-Doppler map."""

    range_m: float
    radial_velocity_mps: float
    normalized_magnitude: float
    range_bin: int
    doppler_bin: int


def simulate_chirp_sequence(
    config: FMCWConfig,
    sequence: ChirpSequenceConfig,
    targets: list[PointTarget],
    *,
    snr_db: float | None = None,
    rng_seed: int | None = 7,
) -> np.ndarray:
    """Generate a coherent complex beat-signal matrix.

    The returned array has shape ``(num_chirps, num_fast_time_samples)``.
    Range appears as a fast-time beat frequency, while radial velocity creates
    coherent slow-time phase progression across chirps.
    """

    fast_time_s = np.arange(config.num_samples, dtype=float) / config.sample_rate_hz
    slow_time_s = (
        np.arange(sequence.num_chirps, dtype=float)
        * sequence.chirp_repetition_interval_s
    )
    signal = np.zeros((sequence.num_chirps, config.num_samples), dtype=np.complex128)
    wavelength_m = SPEED_OF_LIGHT_MPS / config.carrier_frequency_hz

    for target in targets:
        range_frequency_hz = (
            2.0 * config.chirp_slope_hz_per_s * target.range_m / SPEED_OF_LIGHT_MPS
        )
        doppler_frequency_hz = 2.0 * target.radial_velocity_mps / wavelength_m
        fast_phase = 2.0 * pi * (range_frequency_hz + doppler_frequency_hz) * fast_time_s
        slow_phase = 2.0 * pi * doppler_frequency_hz * slow_time_s
        signal += target.amplitude * np.exp(
            1j * (slow_phase[:, None] + fast_phase[None, :] + target.phase_rad)
        )

    if snr_db is not None and np.any(signal):
        signal_power = float(np.mean(np.abs(signal) ** 2))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        rng = np.random.default_rng(rng_seed)
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
        )
        signal = signal + noise

    return signal


def range_doppler_map(
    signal: np.ndarray,
    config: FMCWConfig,
    sequence: ChirpSequenceConfig,
    *,
    range_fft_size: int = 2048,
    doppler_fft_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return range axis, velocity axis and normalized range-Doppler magnitude."""

    samples = np.asarray(signal, dtype=np.complex128)
    expected_shape = (sequence.num_chirps, config.num_samples)
    if samples.shape != expected_shape:
        raise ValueError(f"signal must have shape {expected_shape}")
    if range_fft_size < config.num_samples:
        raise ValueError("range_fft_size cannot be smaller than fast-time sample count")

    doppler_size = doppler_fft_size or sequence.num_chirps
    if doppler_size < sequence.num_chirps:
        raise ValueError("doppler_fft_size cannot be smaller than num_chirps")

    range_window = np.hanning(config.num_samples)[None, :]
    range_fft = np.fft.fft(samples * range_window, n=range_fft_size, axis=1)
    positive_bins = range_fft_size // 2 + 1
    range_fft = range_fft[:, :positive_bins]

    doppler_window = np.hanning(sequence.num_chirps)[:, None]
    rd_complex = np.fft.fftshift(
        np.fft.fft(range_fft * doppler_window, n=doppler_size, axis=0), axis=0
    )
    magnitude = np.abs(rd_complex).T
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    if peak > 0:
        magnitude = magnitude / peak

    frequencies_hz = np.fft.rfftfreq(range_fft_size, d=1.0 / config.sample_rate_hz)
    ranges_m = frequencies_hz * SPEED_OF_LIGHT_MPS / (2.0 * config.chirp_slope_hz_per_s)

    doppler_hz = np.fft.fftshift(
        np.fft.fftfreq(doppler_size, d=sequence.chirp_repetition_interval_s)
    )
    wavelength_m = SPEED_OF_LIGHT_MPS / config.carrier_frequency_hz
    velocities_mps = doppler_hz * wavelength_m / 2.0
    return ranges_m, velocities_mps, magnitude


def detect_range_doppler_peaks(
    magnitude: np.ndarray,
    ranges_m: np.ndarray,
    velocities_mps: np.ndarray,
    *,
    threshold: float = 0.2,
    maximum_detections: int | None = None,
    range_guard_bins: int = 2,
    doppler_guard_bins: int = 2,
) -> list[RangeDopplerDetection]:
    """Detect strong two-dimensional peaks using deterministic suppression.

    This is a transparent baseline detector. CFAR will replace the fixed
    threshold in the next research milestone.
    """

    values = np.asarray(magnitude, dtype=float)
    if values.shape != (len(ranges_m), len(velocities_mps)):
        raise ValueError("magnitude shape must match the supplied axes")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if range_guard_bins < 0 or doppler_guard_bins < 0:
        raise ValueError("guard-bin counts cannot be negative")
    if maximum_detections is not None and maximum_detections < 1:
        raise ValueError("maximum_detections must be positive")

    candidates = np.argwhere(values >= threshold)
    ranked = sorted(candidates.tolist(), key=lambda ij: values[ij[0], ij[1]], reverse=True)
    selected: list[tuple[int, int]] = []
    for range_bin, doppler_bin in ranked:
        if any(
            abs(range_bin - kept_range) <= range_guard_bins
            and abs(doppler_bin - kept_doppler) <= doppler_guard_bins
            for kept_range, kept_doppler in selected
        ):
            continue
        selected.append((range_bin, doppler_bin))
        if maximum_detections is not None and len(selected) >= maximum_detections:
            break

    detections = [
        RangeDopplerDetection(
            range_m=float(ranges_m[range_bin]),
            radial_velocity_mps=float(velocities_mps[doppler_bin]),
            normalized_magnitude=float(values[range_bin, doppler_bin]),
            range_bin=int(range_bin),
            doppler_bin=int(doppler_bin),
        )
        for range_bin, doppler_bin in selected
    ]
    return sorted(detections, key=lambda item: item.range_m)
