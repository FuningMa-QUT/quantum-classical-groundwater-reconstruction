"""Hydraulic-conductivity field generation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-15:
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def _smoothed_noise(
    nx: int,
    ny: int,
    *,
    rng: np.random.Generator,
    correlation_length: float,
) -> np.ndarray:
    sigma = max(float(correlation_length), 1e-6)
    noise = rng.standard_normal((ny, nx))
    return gaussian_filter(noise, sigma=sigma, mode="reflect")


def _base_lognormal_field(
    nx: int,
    ny: int,
    *,
    rng: np.random.Generator,
    correlation_length: float,
    log_mean: float,
    log_variance: float,
) -> np.ndarray:
    smooth = _smoothed_noise(
        nx,
        ny,
        rng=rng,
        correlation_length=correlation_length,
    )
    eta = _standardize(smooth)
    return np.exp(float(log_mean) + np.sqrt(max(float(log_variance), 0.0)) * eta)


def _normalized_coordinates(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, ny),
        np.linspace(0.0, 1.0, nx),
        indexing="ij",
    )
    return yy, xx


def _repeat_to_length(values: Sequence[float], length: int) -> list[float]:
    source = [float(value) for value in values]
    if not source:
        return [1.0] * max(1, length)
    repeats = (max(1, length) + len(source) - 1) // len(source)
    return (source * repeats)[: max(1, length)]


def _resolve_orientation(axis: str, orientation_deg: float | None) -> float:
    if orientation_deg is not None:
        return float(orientation_deg)
    axis_name = str(axis).lower()
    mapping = {
        "horizontal": 0.0,
        "vertical": 90.0,
        "diagonal": 45.0,
        "antidiagonal": -45.0,
    }
    return mapping.get(axis_name, 0.0)


def _rotated_coordinate(nx: int, ny: int, orientation_deg: float) -> np.ndarray:
    yy, xx = _normalized_coordinates(nx, ny)
    theta = np.deg2rad(float(orientation_deg))
    coord = np.cos(theta) * yy + np.sin(theta) * xx
    coord = coord - float(coord.min())
    span = float(coord.max() - coord.min())
    if span < 1e-15:
        return np.zeros_like(coord)
    return coord / span


def _layered_multiplier(
    nx: int,
    ny: int,
    *,
    rng: np.random.Generator,
    correlation_length: float,
    n_layers: int,
    layer_axis: str,
    orientation_deg: float | None,
    layer_multipliers: Sequence[float] | None,
    layer_contrast: float,
    transition_sigma: float,
    jitter_strength: float,
) -> np.ndarray:
    n_effective = max(1, int(n_layers))
    coord = _rotated_coordinate(nx, ny, _resolve_orientation(layer_axis, orientation_deg))

    if float(jitter_strength) > 0.0:
        jitter = _standardize(
            _smoothed_noise(
                nx,
                ny,
                rng=rng,
                correlation_length=max(1.0, float(correlation_length)),
            )
        )
        coord = np.clip(coord + float(jitter_strength) * jitter / n_effective, 0.0, 1.0)

    if layer_multipliers is None:
        contrast = max(float(layer_contrast), 1.0)
        multipliers = np.geomspace(1.0 / contrast, contrast, num=n_effective)
    else:
        multipliers = np.asarray(_repeat_to_length(layer_multipliers, n_effective), dtype=float)

    indices = np.minimum((coord * n_effective).astype(int), n_effective - 1)
    multiplier = multipliers[indices]
    if float(transition_sigma) > 0.0:
        multiplier = gaussian_filter(multiplier, sigma=float(transition_sigma), mode="reflect")
    mean_value = max(float(np.mean(multiplier)), 1e-15)
    return multiplier / mean_value


def _channelized_multiplier(
    nx: int,
    ny: int,
    *,
    channel_multiplier: float,
    background_multiplier: float,
    channel_width_fraction: float,
    channel_center_fraction: float,
    amplitude_fraction: float,
    meander_wavelength_fraction: float,
    phase_fraction: float,
    secondary_channel_scale: float,
    secondary_channel_offset_fraction: float,
    secondary_channel_shift_fraction: float,
) -> np.ndarray:
    yy, xx = _normalized_coordinates(nx, ny)
    width = max(float(channel_width_fraction), 1e-3)
    wavelength = max(float(meander_wavelength_fraction), 1e-3)
    phase = 2.0 * np.pi * float(phase_fraction)
    center = float(channel_center_fraction)
    amplitude = float(amplitude_fraction)

    centerline = center + amplitude * np.sin((2.0 * np.pi * xx / wavelength) + phase)
    mask = np.exp(-0.5 * ((yy - centerline) / width) ** 2)

    if float(secondary_channel_scale) > 0.0:
        centerline_2 = (
            center
            + float(secondary_channel_shift_fraction)
            + 0.7 * amplitude * np.sin((2.0 * np.pi * (xx + float(secondary_channel_offset_fraction)) / wavelength) + phase + np.pi / 3.0)
        )
        mask_2 = np.exp(-0.5 * ((yy - centerline_2) / (1.15 * width)) ** 2)
        mask = np.clip(mask + float(secondary_channel_scale) * mask_2, 0.0, 1.0)

    multiplier = float(background_multiplier) + (float(channel_multiplier) - float(background_multiplier)) * mask
    mean_value = max(float(np.mean(multiplier)), 1e-15)
    return multiplier / mean_value


def _elliptical_mask(
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    center_x_fraction: float,
    center_y_fraction: float,
    radius_x_fraction: float,
    radius_y_fraction: float,
    orientation_deg: float,
) -> np.ndarray:
    x = xx - float(center_x_fraction)
    y = yy - float(center_y_fraction)
    theta = np.deg2rad(float(orientation_deg))
    xr = np.cos(theta) * x + np.sin(theta) * y
    yr = -np.sin(theta) * x + np.cos(theta) * y
    radius_x = max(float(radius_x_fraction), 1e-4)
    radius_y = max(float(radius_y_fraction), 1e-4)
    radius_sq = (xr / radius_x) ** 2 + (yr / radius_y) ** 2
    return np.exp(-0.5 * radius_sq)


def _barrier_lens_multiplier(
    nx: int,
    ny: int,
    *,
    lenses: Sequence[dict[str, float]] | None,
    background_multiplier: float,
    transition_sigma: float,
) -> np.ndarray:
    yy, xx = _normalized_coordinates(nx, ny)
    multiplier = np.full((ny, nx), float(background_multiplier), dtype=float)
    specs = list(lenses) if lenses is not None else [
        {
            "center_x_fraction": 0.56,
            "center_y_fraction": 0.50,
            "radius_x_fraction": 0.18,
            "radius_y_fraction": 0.10,
            "orientation_deg": 18.0,
            "multiplier": 0.08,
        }
    ]

    for spec in specs:
        mask = _elliptical_mask(
            yy,
            xx,
            center_x_fraction=float(spec.get("center_x_fraction", 0.5)),
            center_y_fraction=float(spec.get("center_y_fraction", 0.5)),
            radius_x_fraction=float(spec.get("radius_x_fraction", 0.15)),
            radius_y_fraction=float(spec.get("radius_y_fraction", 0.08)),
            orientation_deg=float(spec.get("orientation_deg", 0.0)),
        )
        local_multiplier = 1.0 + (float(spec.get("multiplier", 0.1)) - 1.0) * mask
        multiplier *= local_multiplier

    if float(transition_sigma) > 0.0:
        multiplier = gaussian_filter(multiplier, sigma=float(transition_sigma), mode="reflect")
    mean_value = max(float(np.mean(multiplier)), 1e-15)
    return multiplier / mean_value


def _facies_multiplier(
    nx: int,
    ny: int,
    *,
    rng: np.random.Generator,
    facies_quantiles: Sequence[float] | None,
    facies_multipliers: Sequence[float] | None,
    facies_correlation_length: float,
    transition_sigma: float,
) -> np.ndarray:
    latent = _standardize(
        _smoothed_noise(
            nx,
            ny,
            rng=rng,
            correlation_length=max(float(facies_correlation_length), 1e-6),
        )
    )
    quantiles = sorted(
        min(max(float(value), 0.0), 1.0)
        for value in (facies_quantiles or [0.25, 0.55, 0.82])
    )
    thresholds = np.quantile(latent, quantiles) if quantiles else np.asarray([], dtype=float)
    multipliers = np.asarray(
        _repeat_to_length(facies_multipliers or [0.08, 0.35, 1.5, 5.5], len(thresholds) + 1),
        dtype=float,
    )
    indices = np.digitize(latent, thresholds, right=False)
    multiplier = multipliers[indices]
    if float(transition_sigma) > 0.0:
        multiplier = gaussian_filter(multiplier, sigma=float(transition_sigma), mode="reflect")
    mean_value = max(float(np.mean(multiplier)), 1e-15)
    return multiplier / mean_value


def _apply_rectangular_overlays(
    field: np.ndarray,
    overlays: Sequence[dict[str, float]] | None,
) -> np.ndarray:
    if not overlays:
        return field

    updated = np.asarray(field, dtype=float).copy()
    ny, nx = updated.shape

    for spec in overlays:
        i_start = max(0, int(spec.get("i_start", 0)))
        i_stop = min(ny, int(spec.get("i_stop", ny)))
        j_start = max(0, int(spec.get("j_start", 0)))
        j_stop = min(nx, int(spec.get("j_stop", nx)))
        if i_stop <= i_start or j_stop <= j_start:
            continue

        mode = str(spec.get("mode", "multiply")).lower()
        if mode == "set":
            value = float(spec.get("value", spec.get("set_value", 0.0)))
            updated[i_start:i_stop, j_start:j_stop] = value
        elif mode == "multiply":
            multiplier = float(spec.get("multiplier", spec.get("value", 1.0)))
            updated[i_start:i_stop, j_start:j_stop] *= multiplier
        else:
            raise ValueError(f"Unknown rectangular overlay mode: {mode}")

    return updated


def generate_k_field(
    nx: int,
    ny: int,
    *,
    model: str = "lognormal",
    seed: int = 42,
    correlation_length: float = 2.0,
    log_mean: float = 2.0,
    log_variance: float = 1.0,
    k_min: float = 2.0,
    k_max: float = 20.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
    n_layers: int = 6,
    layer_axis: str = "horizontal",
    orientation_deg: float | None = None,
    layer_multipliers: Sequence[float] | None = None,
    layer_contrast: float = 6.0,
    transition_sigma: float = 0.8,
    jitter_strength: float = 0.05,
    channel_multiplier: float = 6.0,
    background_multiplier: float = 1.0,
    channel_width_fraction: float = 0.08,
    channel_center_fraction: float = 0.5,
    amplitude_fraction: float = 0.18,
    meander_wavelength_fraction: float = 0.9,
    phase_fraction: float = 0.0,
    secondary_channel_scale: float = 0.0,
    secondary_channel_offset_fraction: float = 0.18,
    secondary_channel_shift_fraction: float = -0.16,
    lenses: Sequence[dict[str, float]] | None = None,
    facies_quantiles: Sequence[float] | None = None,
    facies_multipliers: Sequence[float] | None = None,
    facies_correlation_length: float = 3.0,
    rectangular_overlays: Sequence[dict[str, float]] | None = None,
) -> np.ndarray:
    """Generate a spatially correlated hydraulic-conductivity field.

    Parameters
    ----------
    model:
        ``"lognormal"`` creates ``exp(log_mean + sqrt(log_variance) * eta)``.
        ``"bounded_smooth"`` preserves the old scripts' min-max normalized field.
        ``"constant"`` returns a homogeneous field with value ``k_min``.
        Structured groundwater benchmarks are available through:
        ``"layered_lognormal"``, ``"channelized_lognormal"``,
        ``"barrier_lens_lognormal"``, ``"facies_lognormal"``, and
        ``"channel_lens_lognormal"``.
        ``rectangular_overlays`` can impose piecewise zones for public benchmark
        adaptations such as MODFLOW low-K blocks.
    """

    rng = np.random.default_rng(seed)
    model_name = str(model).lower()

    if model_name == "constant":
        field = np.full((ny, nx), float(k_min))
    else:
        smooth = _smoothed_noise(
            nx,
            ny,
            rng=rng,
            correlation_length=correlation_length,
        )

        if model_name == "lognormal":
            field = np.exp(float(log_mean) + np.sqrt(max(float(log_variance), 0.0)) * _standardize(smooth))
        elif model_name == "bounded_smooth":
            lo = float(smooth.min())
            hi = float(smooth.max())
            if abs(hi - lo) < 1e-15:
                field = np.full((ny, nx), (float(k_min) + float(k_max)) / 2.0)
            else:
                normalized = (smooth - lo) / (hi - lo)
                field = float(k_min) + normalized * (float(k_max) - float(k_min))
        elif model_name in {"layered", "layered_lognormal", "layered_aquifer"}:
            base = _base_lognormal_field(
                nx,
                ny,
                rng=rng,
                correlation_length=correlation_length,
                log_mean=log_mean,
                log_variance=log_variance,
            )
            field = base * _layered_multiplier(
                nx,
                ny,
                rng=rng,
                correlation_length=correlation_length,
                n_layers=n_layers,
                layer_axis=layer_axis,
                orientation_deg=orientation_deg,
                layer_multipliers=layer_multipliers,
                layer_contrast=layer_contrast,
                transition_sigma=transition_sigma,
                jitter_strength=jitter_strength,
            )
        elif model_name in {"channelized", "channelized_lognormal", "channelized_aquifer"}:
            base = _base_lognormal_field(
                nx,
                ny,
                rng=rng,
                correlation_length=correlation_length,
                log_mean=log_mean,
                log_variance=log_variance,
            )
            field = base * _channelized_multiplier(
                nx,
                ny,
                channel_multiplier=channel_multiplier,
                background_multiplier=background_multiplier,
                channel_width_fraction=channel_width_fraction,
                channel_center_fraction=channel_center_fraction,
                amplitude_fraction=amplitude_fraction,
                meander_wavelength_fraction=meander_wavelength_fraction,
                phase_fraction=phase_fraction,
                secondary_channel_scale=secondary_channel_scale,
                secondary_channel_offset_fraction=secondary_channel_offset_fraction,
                secondary_channel_shift_fraction=secondary_channel_shift_fraction,
            )
        elif model_name in {"channel_lens", "channel_lens_lognormal", "channelized_barrier_lens"}:
            base = _base_lognormal_field(
                nx,
                ny,
                rng=rng,
                correlation_length=correlation_length,
                log_mean=log_mean,
                log_variance=log_variance,
            )
            channel = _channelized_multiplier(
                nx,
                ny,
                channel_multiplier=channel_multiplier,
                background_multiplier=background_multiplier,
                channel_width_fraction=channel_width_fraction,
                channel_center_fraction=channel_center_fraction,
                amplitude_fraction=amplitude_fraction,
                meander_wavelength_fraction=meander_wavelength_fraction,
                phase_fraction=phase_fraction,
                secondary_channel_scale=secondary_channel_scale,
                secondary_channel_offset_fraction=secondary_channel_offset_fraction,
                secondary_channel_shift_fraction=secondary_channel_shift_fraction,
            )
            lens = _barrier_lens_multiplier(
                nx,
                ny,
                lenses=lenses,
                background_multiplier=1.0,
                transition_sigma=transition_sigma,
            )
            field = base * channel * lens
        elif model_name in {"barrier_lens", "barrier_lens_lognormal", "barrier_aquifer"}:
            base = _base_lognormal_field(
                nx,
                ny,
                rng=rng,
                correlation_length=correlation_length,
                log_mean=log_mean,
                log_variance=log_variance,
            )
            field = base * _barrier_lens_multiplier(
                nx,
                ny,
                lenses=lenses,
                background_multiplier=background_multiplier,
                transition_sigma=transition_sigma,
            )
        elif model_name in {"facies", "multi_facies", "facies_lognormal"}:
            base = _base_lognormal_field(
                nx,
                ny,
                rng=rng,
                correlation_length=correlation_length,
                log_mean=log_mean,
                log_variance=log_variance,
            )
            field = base * _facies_multiplier(
                nx,
                ny,
                rng=rng,
                facies_quantiles=facies_quantiles,
                facies_multipliers=facies_multipliers,
                facies_correlation_length=facies_correlation_length,
                transition_sigma=transition_sigma,
            )
        else:
            raise ValueError(f"Unknown K-field model: {model}")

    field = _apply_rectangular_overlays(field, rectangular_overlays)

    if clip_min is not None or clip_max is not None:
        field = np.clip(
            field,
            -np.inf if clip_min is None else clip_min,
            np.inf if clip_max is None else clip_max,
        )

    return field.astype(float, copy=False)
