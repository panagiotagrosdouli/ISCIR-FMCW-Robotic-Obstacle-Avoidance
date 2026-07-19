"""Baseband FMCW signal simulation and one-dimensional range estimation.

The model generates the dechirped beat signal directly. For a stationary target,
the beat frequency is approximately ``f_b = 2 S R / c``, where ``S`` is the
chirp slope, ``R`` is range and ``c`` is the speed of light.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

SPEED_OF_LIGHT_MPS = 299_792_458.0


@dataclass(frozen=True, slots=True)
class FMCWConfig:
    """Parameters of a single linear FMCW chirp."""

    carrier_frequency_hz: float = 77e9
    bandwidth_hz: float = 1e9
    chirp_duration_s: float = 50e-6
    sample_rate_hz: float = 10e6

    def __post_init__(self) -> None:
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")
        if self.chirp_duration_s <= 0:
            raise ValueError("chirp_duration_s must be positive")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.num_samples < 2:
            raise ValueError("configuration must produce at least two samples")

    @property
    def chirp_slope_hz_per_s(self) -> float:
        return self.bandwidth_hz / self.chirp_duration_s

    @property
    def num_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.chirp_duration_s))

    @property
    def range_resolution_m(self) -> float:
        return SPEED_OF_LIGHT_MPS / (2.0 * self.bandwidth_hz)

    @property
    def maximum_unambiguous_range_m(self) -> float:
        # Positive beat frequencies are limited by the Nyquist frequency.
        return (
            SPEED_OF_LIGHT_MPS
            * self.sample_rate_hz
            / (4.0 * self.chirp_slope_hz_per_s)
        )


@dataclass(frozen=True, slots=True)
class PointTarget:
    """Ideal point target used by the baseband simulator."""

    range_m: float
    radial_velocity_mps: float = 0.0
    amplitude: float = 1.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.range_m < 0:
            raise ValueError("range_m cannot be negative")
        if self.amplitude < 0:
            raise ValueError("amplitude cannot be negative")


def simulate_beat_signal(
    config: FMCWConfig,
    targets: list[PointTarget],
    *,
    snr_db: float | None = None,
    rng_seed: int | None = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one complex dechirped chirp for one or more point targets.

    Doppler is represented by ``f_d = 2 v / lambda`` and added to the range beat
    frequency. This narrowband approximation is suitable for the first project
    milestone; a delayed transmit/receive waveform model can be added later.
    """

    time_s = np.arange(config.num_samples, dtype=float) / config.sample_rate_hz
    signal = np.zeros(config.num_samples, dtype=np.complex128)
    wavelength_m = SPEED_OF_LIGHT_MPS / config.carrier_frequency_hz

    for target in targets:
        range_frequency_hz = (
            2.0 * config.chirp_slope_hz_per_s * target.range_m / SPEED_OF_LIGHT_MPS
        )
        doppler_frequency_hz = 2.0 * target.radial_velocity_mps / wavelength_m
        beat_frequency_hz = range_frequency_hz + doppler_frequency_hz
        signal += target.amplitude * np.exp(
            1j * (2.0 * pi * beat_frequency_hz * time_s + target.phase_rad)
        )

    if snr_db is not None and np.any(signal):
        signal_power = float(np.mean(np.abs(signal) ** 2))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        rng = np.random.default_rng(rng_seed)
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(config.num_samples)
            + 1j * rng.standard_normal(config.num_samples)
        )
        signal = signal + noise

    return time_s, signal


def range_spectrum(
    signal: np.ndarray,
    config: FMCWConfig,
    *,
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the positive-frequency range axis and normalized FFT magnitude."""

    samples = np.asarray(signal, dtype=np.complex128)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("signal must be a one-dimensional array with at least two samples")

    fft_size = n_fft or int(2 ** np.ceil(np.log2(samples.size)))
    if fft_size < samples.size:
        raise ValueError("n_fft cannot be smaller than the number of signal samples")

    window = np.hanning(samples.size)
    spectrum = np.fft.fft(samples * window, n=fft_size)
    frequencies_hz = np.fft.fftfreq(fft_size, d=1.0 / config.sample_rate_hz)
    positive = frequencies_hz >= 0

    magnitude = np.abs(spectrum[positive])
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    if peak > 0:
        magnitude = magnitude / peak

    ranges_m = (
        frequencies_hz[positive]
        * SPEED_OF_LIGHT_MPS
        / (2.0 * config.chirp_slope_hz_per_s)
    )
    return ranges_m, magnitude


def estimate_range_m(signal: np.ndarray, config: FMCWConfig, *, n_fft: int = 8192) -> float:
    """Estimate the strongest target range from the range FFT."""

    ranges_m, magnitude = range_spectrum(signal, config, n_fft=n_fft)
    if not magnitude.size or not np.any(magnitude):
        raise ValueError("cannot estimate range from an empty or zero signal")
    return float(ranges_m[int(np.argmax(magnitude))])
