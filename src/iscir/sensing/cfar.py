"""Two-dimensional cell-averaging CFAR for range-Doppler maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .range_doppler import RangeDopplerDetection


@dataclass(frozen=True, slots=True)
class CACFARConfig:
    """Configuration for rectangular two-dimensional CA-CFAR.

    Cell counts are specified independently for the range and Doppler axes.
    Training cells surround the guard region and the cell under test. The
    threshold factor is derived from ``false_alarm_probability`` assuming
    exponentially distributed noise power.
    """

    training_range_cells: int = 8
    training_doppler_cells: int = 4
    guard_range_cells: int = 2
    guard_doppler_cells: int = 1
    false_alarm_probability: float = 1e-4

    def __post_init__(self) -> None:
        if self.training_range_cells < 1 or self.training_doppler_cells < 1:
            raise ValueError("training-cell counts must be positive")
        if self.guard_range_cells < 0 or self.guard_doppler_cells < 0:
            raise ValueError("guard-cell counts cannot be negative")
        if not 0.0 < self.false_alarm_probability < 1.0:
            raise ValueError("false_alarm_probability must be between 0 and 1")

    @property
    def number_of_training_cells(self) -> int:
        outer_rows = 2 * (
            self.training_range_cells + self.guard_range_cells
        ) + 1
        outer_cols = 2 * (
            self.training_doppler_cells + self.guard_doppler_cells
        ) + 1
        guard_rows = 2 * self.guard_range_cells + 1
        guard_cols = 2 * self.guard_doppler_cells + 1
        return outer_rows * outer_cols - guard_rows * guard_cols

    @property
    def threshold_scale(self) -> float:
        count = self.number_of_training_cells
        return count * (
            self.false_alarm_probability ** (-1.0 / count) - 1.0
        )


@dataclass(frozen=True, slots=True)
class CACFARResult:
    """Intermediate products and binary detections produced by CA-CFAR."""

    detection_mask: np.ndarray
    threshold_map: np.ndarray
    noise_power_map: np.ndarray
    valid_cell_mask: np.ndarray


def _rectangle_sum(
    integral: np.ndarray,
    row_start: np.ndarray,
    row_end: np.ndarray,
    col_start: np.ndarray,
    col_end: np.ndarray,
) -> np.ndarray:
    """Return inclusive rectangular sums from a zero-padded integral image."""

    return (
        integral[row_end + 1, col_end + 1]
        - integral[row_start, col_end + 1]
        - integral[row_end + 1, col_start]
        + integral[row_start, col_start]
    )


def ca_cfar_2d(
    magnitude: np.ndarray,
    config: CACFARConfig = CACFARConfig(),
    *,
    input_is_power: bool = False,
) -> CACFARResult:
    """Apply two-dimensional CA-CFAR to a range-Doppler magnitude map.

    Parameters
    ----------
    magnitude:
        Non-negative 2D magnitude or power map. Axis 0 is range and axis 1 is
        Doppler.
    config:
        Training, guard and false-alarm configuration.
    input_is_power:
        Set to ``True`` when ``magnitude`` already contains linear power.

    Notes
    -----
    Border cells without a complete training window are marked invalid and are
    never reported as detections. Thresholds for those cells are ``nan``.
    """

    values = np.asarray(magnitude, dtype=float)
    if values.ndim != 2:
        raise ValueError("magnitude must be a two-dimensional array")
    if values.size == 0:
        raise ValueError("magnitude cannot be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("magnitude must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("magnitude cannot contain negative values")

    power = values if input_is_power else values**2
    rows, cols = power.shape

    range_margin = config.training_range_cells + config.guard_range_cells
    doppler_margin = config.training_doppler_cells + config.guard_doppler_cells
    minimum_shape = (2 * range_margin + 1, 2 * doppler_margin + 1)
    if rows < minimum_shape[0] or cols < minimum_shape[1]:
        raise ValueError(
            "magnitude is too small for the configured CA-CFAR window; "
            f"minimum shape is {minimum_shape}"
        )

    valid_rows = np.arange(range_margin, rows - range_margin)
    valid_cols = np.arange(doppler_margin, cols - doppler_margin)
    row_grid, col_grid = np.meshgrid(valid_rows, valid_cols, indexing="ij")

    integral = np.pad(power, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)

    outer_sum = _rectangle_sum(
        integral,
        row_grid - range_margin,
        row_grid + range_margin,
        col_grid - doppler_margin,
        col_grid + doppler_margin,
    )
    guard_sum = _rectangle_sum(
        integral,
        row_grid - config.guard_range_cells,
        row_grid + config.guard_range_cells,
        col_grid - config.guard_doppler_cells,
        col_grid + config.guard_doppler_cells,
    )

    noise_power = (outer_sum - guard_sum) / config.number_of_training_cells
    threshold = noise_power * config.threshold_scale
    cells_under_test = power[row_grid, col_grid]
    detected = cells_under_test > threshold

    valid_mask = np.zeros_like(power, dtype=bool)
    detection_mask = np.zeros_like(power, dtype=bool)
    threshold_map = np.full_like(power, np.nan, dtype=float)
    noise_map = np.full_like(power, np.nan, dtype=float)

    valid_mask[row_grid, col_grid] = True
    detection_mask[row_grid, col_grid] = detected
    threshold_map[row_grid, col_grid] = threshold
    noise_map[row_grid, col_grid] = noise_power

    return CACFARResult(
        detection_mask=detection_mask,
        threshold_map=threshold_map,
        noise_power_map=noise_map,
        valid_cell_mask=valid_mask,
    )


def detect_cfar_peaks(
    magnitude: np.ndarray,
    ranges_m: np.ndarray,
    velocities_mps: np.ndarray,
    config: CACFARConfig = CACFARConfig(),
    *,
    maximum_detections: int | None = None,
    suppression_range_bins: int = 2,
    suppression_doppler_bins: int = 2,
) -> tuple[list[RangeDopplerDetection], CACFARResult]:
    """Run CA-CFAR and group detected cells into range-Doppler peaks."""

    values = np.asarray(magnitude, dtype=float)
    if values.shape != (len(ranges_m), len(velocities_mps)):
        raise ValueError("magnitude shape must match the supplied axes")
    if maximum_detections is not None and maximum_detections < 1:
        raise ValueError("maximum_detections must be positive")
    if suppression_range_bins < 0 or suppression_doppler_bins < 0:
        raise ValueError("suppression-bin counts cannot be negative")

    result = ca_cfar_2d(values, config)
    candidates = np.argwhere(result.detection_mask)
    ranked = sorted(
        candidates.tolist(),
        key=lambda index: values[index[0], index[1]],
        reverse=True,
    )

    selected: list[tuple[int, int]] = []
    for range_bin, doppler_bin in ranked:
        if any(
            abs(range_bin - kept_range) <= suppression_range_bins
            and abs(doppler_bin - kept_doppler) <= suppression_doppler_bins
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
    detections.sort(key=lambda detection: detection.range_m)
    return detections, result
