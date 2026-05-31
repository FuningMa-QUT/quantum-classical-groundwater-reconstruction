"""Experiment metrics for SCI-grade reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .operators import WellTerm


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(reference)) ** 2)))


def mae(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(estimate) - np.asarray(reference))))


def max_abs_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(estimate) - np.asarray(reference))))


def relative_l2(estimate: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(estimate).reshape(-1) - np.asarray(reference).reshape(-1)))
    denominator = float(np.linalg.norm(np.asarray(reference).reshape(-1)))
    if denominator < 1e-30:
        return np.nan
    return numerator / denominator


def uncertainty_metrics(
    prefix: str,
    estimate: np.ndarray,
    reference: np.ndarray,
    prediction_std: np.ndarray,
) -> dict[str, float]:
    estimate_arr = np.asarray(estimate, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    std_arr = np.asarray(prediction_std, dtype=float)

    finite_mask = np.isfinite(std_arr)
    positive_mask = finite_mask & (std_arr > 1e-15)
    abs_error = np.abs(estimate_arr - reference_arr)

    metrics = {
        f"{prefix}_prediction_std_mean": float(np.nanmean(std_arr[finite_mask])) if np.any(finite_mask) else np.nan,
        f"{prefix}_prediction_std_p90": float(np.nanpercentile(std_arr[finite_mask], 90))
        if np.any(finite_mask)
        else np.nan,
        f"{prefix}_coverage_1sigma": float(np.mean(abs_error[positive_mask] <= std_arr[positive_mask]))
        if np.any(positive_mask)
        else np.nan,
        f"{prefix}_coverage_2sigma": float(np.mean(abs_error[positive_mask] <= 2.0 * std_arr[positive_mask]))
        if np.any(positive_mask)
        else np.nan,
        f"{prefix}_mean_abs_zscore": float(np.mean(abs_error[positive_mask] / std_arr[positive_mask]))
        if np.any(positive_mask)
        else np.nan,
    }
    return metrics


def field_metrics(
    prefix: str,
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    prediction_std: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        f"{prefix}_rmse": rmse(estimate, reference),
        f"{prefix}_mae": mae(estimate, reference),
        f"{prefix}_max_abs": max_abs_error(estimate, reference),
        f"{prefix}_relative_l2": relative_l2(estimate, reference),
    }
    if prediction_std is not None:
        metrics.update(uncertainty_metrics(prefix, estimate, reference, prediction_std))
    return metrics


def total_mass(concentration: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sum(concentration) * dx * dy)


def peak_bias_percent(estimate: np.ndarray, reference: np.ndarray) -> float:
    ref_peak = float(np.max(reference))
    if abs(ref_peak) < 1e-30:
        return np.nan
    return float((np.max(estimate) - ref_peak) / ref_peak * 100.0)


def _positive_concentration(concentration: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(concentration, dtype=float), 0.0, None)


def plume_moments(
    concentration: np.ndarray,
    *,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    dx: float,
    dy: float,
) -> dict[str, float]:
    weights = _positive_concentration(concentration) * dx * dy
    total = float(weights.sum())
    if total < 1e-30:
        return {
            "mass": 0.0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "spread_x": np.nan,
            "spread_y": np.nan,
        }

    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    centroid_x = float(np.sum(weights * xx) / total)
    centroid_y = float(np.sum(weights * yy) / total)
    spread_x = float(np.sqrt(np.sum(weights * (xx - centroid_x) ** 2) / total))
    spread_y = float(np.sqrt(np.sum(weights * (yy - centroid_y) ** 2) / total))
    return {
        "mass": total,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "spread_x": spread_x,
        "spread_y": spread_y,
    }


def control_plane_metrics(
    concentration: np.ndarray,
    *,
    vx: np.ndarray,
    dy: float,
    plane_columns: dict[str, int],
) -> dict[str, float]:
    positive = _positive_concentration(concentration)
    metrics: dict[str, float] = {}
    for plane_name, column in plane_columns.items():
        col = int(column)
        inventory = float(np.sum(positive[:, col]) * dy)
        flux = float(np.sum(np.maximum(vx[:, col], 0.0) * positive[:, col]) * dy)
        metrics[f"plane_{plane_name}_inventory"] = inventory
        metrics[f"plane_{plane_name}_flux"] = flux
    return metrics


def extraction_well_metrics(
    concentration: np.ndarray,
    *,
    flow_wells: list[WellTerm],
) -> dict[str, float]:
    positive = _positive_concentration(concentration)
    metrics: dict[str, float] = {}
    total_sink_strength = 0.0
    total_capture_rate = 0.0

    extraction_index = 0
    for well in flow_wells:
        if float(well.value) >= 0.0:
            continue
        extraction_index += 1
        sink_strength = abs(float(well.value))
        total_sink_strength += sink_strength
        name = str(well.name).strip() or f"ew{extraction_index:02d}"
        concentration_value = float(positive[int(well.i), int(well.j)])
        capture_rate = sink_strength * concentration_value
        total_capture_rate += capture_rate
        metrics[f"well_{name}_capture_concentration"] = concentration_value
        metrics[f"well_{name}_capture_rate"] = capture_rate

    if total_sink_strength > 0.0:
        metrics["well_capture_rate_total"] = total_capture_rate
        metrics["well_capture_concentration_weighted"] = total_capture_rate / total_sink_strength

    return metrics


def breakthrough_summary(
    trace_rows: list[dict[str, float]],
    *,
    plane_names: list[str],
) -> dict[str, float]:
    if not trace_rows:
        return {}

    times = np.asarray([float(row["time"]) for row in trace_rows], dtype=float)
    summary: dict[str, float] = {}

    for plane_name in plane_names:
        flux_key = f"plane_{plane_name}_flux"
        if flux_key not in trace_rows[0]:
            continue

        flux = np.asarray([float(row[flux_key]) for row in trace_rows], dtype=float)
        peak = float(np.max(flux))
        peak_idx = int(np.argmax(flux))
        summary[f"{plane_name}_peak_flux"] = peak
        summary[f"{plane_name}_peak_time"] = float(times[peak_idx])

        if peak <= 1e-30:
            summary[f"{plane_name}_arrival_time_05"] = np.nan
            summary[f"{plane_name}_arrival_time_50"] = np.nan
            continue

        threshold_05 = 0.05 * peak
        threshold_50 = 0.50 * peak
        idx_05 = np.where(flux >= threshold_05)[0]
        idx_50 = np.where(flux >= threshold_50)[0]
        summary[f"{plane_name}_arrival_time_05"] = float(times[idx_05[0]]) if idx_05.size else np.nan
        summary[f"{plane_name}_arrival_time_50"] = float(times[idx_50[0]]) if idx_50.size else np.nan

    return summary


def trace_summary_error_metrics(
    prefix: str,
    estimate_trace: list[dict[str, float]],
    reference_summary: dict[str, float],
    *,
    plane_names: list[str],
) -> dict[str, float]:
    estimated_summary = breakthrough_summary(estimate_trace, plane_names=plane_names)
    metrics: dict[str, float] = {}

    for key, est_value in estimated_summary.items():
        metrics[f"{prefix}_{key}"] = est_value
        ref_value = reference_summary.get(key, np.nan)
        metrics[f"{prefix}_{key}_error"] = np.nan if np.isnan(ref_value) else est_value - ref_value
        if key.endswith("peak_flux"):
            metrics[f"{prefix}_{key}_relative_error"] = (
                np.nan if abs(ref_value) < 1e-30 or np.isnan(ref_value) else (est_value - ref_value) / ref_value
            )
    return metrics


def transport_trace_row(
    concentration: np.ndarray,
    *,
    time: float,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    dx: float,
    dy: float,
    vx: np.ndarray,
    plane_columns: dict[str, int],
) -> dict[str, float]:
    row = {
        "time": float(time),
        "peak": float(np.max(concentration)),
        "total_mass": total_mass(concentration, dx, dy),
    }
    row.update(plume_moments(concentration, x_coords=x_coords, y_coords=y_coords, dx=dx, dy=dy))
    row.update(control_plane_metrics(concentration, vx=vx, dy=dy, plane_columns=plane_columns))
    return row


def concentration_metrics(
    prefix: str,
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    dx: float,
    dy: float,
    prediction_std: np.ndarray | None = None,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    vx: np.ndarray | None = None,
    plane_columns: dict[str, int] | None = None,
    flow_wells: list[WellTerm] | None = None,
) -> dict[str, float]:
    metrics = field_metrics(prefix, estimate, reference, prediction_std=prediction_std)
    mass_est = total_mass(estimate, dx, dy)
    mass_ref = total_mass(reference, dx, dy)
    metrics.update(
        {
            f"{prefix}_mass": mass_est,
            f"{prefix}_mass_reference": mass_ref,
            f"{prefix}_mass_error": mass_est - mass_ref,
            f"{prefix}_mass_relative_error": np.nan
            if abs(mass_ref) < 1e-30
            else (mass_est - mass_ref) / mass_ref,
            f"{prefix}_peak": float(np.max(estimate)),
            f"{prefix}_peak_reference": float(np.max(reference)),
            f"{prefix}_peak_bias_percent": peak_bias_percent(estimate, reference),
        }
    )

    if x_coords is not None and y_coords is not None:
        plume_est = plume_moments(estimate, x_coords=x_coords, y_coords=y_coords, dx=dx, dy=dy)
        plume_ref = plume_moments(reference, x_coords=x_coords, y_coords=y_coords, dx=dx, dy=dy)
        centroid_dx = plume_est["centroid_x"] - plume_ref["centroid_x"]
        centroid_dy = plume_est["centroid_y"] - plume_ref["centroid_y"]
        metrics.update(
            {
                f"{prefix}_centroid_x": plume_est["centroid_x"],
                f"{prefix}_centroid_x_reference": plume_ref["centroid_x"],
                f"{prefix}_centroid_x_error": centroid_dx,
                f"{prefix}_centroid_y": plume_est["centroid_y"],
                f"{prefix}_centroid_y_reference": plume_ref["centroid_y"],
                f"{prefix}_centroid_y_error": centroid_dy,
                f"{prefix}_centroid_distance_error": float(np.hypot(centroid_dx, centroid_dy)),
                f"{prefix}_spread_x": plume_est["spread_x"],
                f"{prefix}_spread_x_reference": plume_ref["spread_x"],
                f"{prefix}_spread_x_error": plume_est["spread_x"] - plume_ref["spread_x"],
                f"{prefix}_spread_y": plume_est["spread_y"],
                f"{prefix}_spread_y_reference": plume_ref["spread_y"],
                f"{prefix}_spread_y_error": plume_est["spread_y"] - plume_ref["spread_y"],
            }
        )

    if vx is not None and plane_columns:
        planes_est = control_plane_metrics(estimate, vx=vx, dy=dy, plane_columns=plane_columns)
        planes_ref = control_plane_metrics(reference, vx=vx, dy=dy, plane_columns=plane_columns)
        reference_flux_scale = max(
            [value for key, value in planes_ref.items() if key.endswith("_flux")],
            default=1.0,
        )
        reference_inventory_scale = max(
            [value for key, value in planes_ref.items() if key.endswith("_inventory")],
            default=1.0,
        )
        for key, est_value in planes_est.items():
            ref_value = planes_ref[key]
            short_key = key
            abs_error = est_value - ref_value
            metrics[f"{prefix}_{short_key}"] = est_value
            metrics[f"{prefix}_{short_key}_reference"] = ref_value
            metrics[f"{prefix}_{short_key}_error"] = abs_error
            metrics[f"{prefix}_{short_key}_relative_error"] = (
                np.nan if abs(ref_value) < 1e-30 else abs_error / ref_value
            )
            if short_key.endswith("_flux"):
                metrics[f"{prefix}_{short_key}_scaled_error"] = (
                    np.nan if reference_flux_scale < 1e-30 else abs_error / reference_flux_scale
                )
            if short_key.endswith("_inventory"):
                metrics[f"{prefix}_{short_key}_scaled_error"] = (
                    np.nan if reference_inventory_scale < 1e-30 else abs_error / reference_inventory_scale
                )

    if flow_wells:
        wells_est = extraction_well_metrics(estimate, flow_wells=flow_wells)
        wells_ref = extraction_well_metrics(reference, flow_wells=flow_wells)
        reference_capture_scale = max(
            [value for key, value in wells_ref.items() if "capture_rate" in key],
            default=1.0,
        )
        for key, est_value in wells_est.items():
            ref_value = wells_ref.get(key, np.nan)
            abs_error = est_value - ref_value
            metrics[f"{prefix}_{key}"] = est_value
            metrics[f"{prefix}_{key}_reference"] = ref_value
            metrics[f"{prefix}_{key}_error"] = abs_error
            metrics[f"{prefix}_{key}_relative_error"] = (
                np.nan if abs(ref_value) < 1e-30 else abs_error / ref_value
            )
            if "capture_rate" in key:
                metrics[f"{prefix}_{key}_scaled_error"] = (
                    np.nan if reference_capture_scale < 1e-30 else abs_error / reference_capture_scale
                )
    return metrics


def matrix_metrics(prefix: str, matrix: sparse.spmatrix) -> dict[str, float]:
    mat = matrix.tocsc()
    n_rows, n_cols = mat.shape
    total = n_rows * n_cols
    return {
        f"{prefix}_n_rows": float(n_rows),
        f"{prefix}_n_cols": float(n_cols),
        f"{prefix}_nnz": float(mat.nnz),
        f"{prefix}_density": float(mat.nnz / total),
        f"{prefix}_sparsity": float(1.0 - mat.nnz / total),
    }


def condition_number_estimate(
    matrix: sparse.spmatrix,
    *,
    maxiter: int = 5000,
    tolerance: float = 1e-8,
) -> float:
    """Estimate sparse 2-norm condition number with extreme singular values."""

    mat = matrix.astype(float).tocsc()
    try:
        sigma_max = float(
            spla.svds(mat, k=1, which="LM", return_singular_vectors=False, tol=tolerance, maxiter=maxiter)[0]
        )
        sigma_min = float(
            spla.svds(mat, k=1, which="SM", return_singular_vectors=False, tol=tolerance, maxiter=maxiter)[0]
        )
    except Exception:
        return np.nan

    if sigma_min <= 1e-30:
        return np.inf
    return sigma_max / sigma_min


def stringify_metrics(row: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = f"{value:.16g}"
        else:
            out[key] = str(value)
    return out
