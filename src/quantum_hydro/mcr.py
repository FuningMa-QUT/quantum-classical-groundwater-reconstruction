"""Multi-point constrained rescaling and regional Bayesian extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MCRResult:
    reconstructed: np.ndarray
    alpha: float
    residual_rmse: float
    denominator: float
    n_constraints: int
    method: str = "global"
    scale_field: np.ndarray | None = None
    prediction_std: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    coefficient_std: np.ndarray | None = None
    posterior_trace: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def estimate_alpha(
    state: np.ndarray,
    indices: np.ndarray | list[int],
    target_values: np.ndarray | list[float],
    *,
    weights: np.ndarray | list[float] | None = None,
) -> tuple[float, float]:
    """Estimate the global MCR scaling factor via weighted least squares."""

    idx = np.asarray(indices, dtype=int)
    targets = np.asarray(target_values, dtype=float)
    q_vals = np.asarray(state, dtype=float).reshape(-1)[idx]

    if weights is None:
        w = np.ones_like(targets)
    else:
        w = np.asarray(weights, dtype=float)

    numerator = float(np.sum(w * q_vals * targets))
    denominator = float(np.sum(w * q_vals * q_vals))
    if denominator < 1e-30:
        return 1.0, denominator
    return numerator / denominator, denominator


def _weights_from_observation_std(
    observation_std: np.ndarray | list[float] | None,
) -> np.ndarray | None:
    if observation_std is None:
        return None
    sigma = np.asarray(observation_std, dtype=float)
    return 1.0 / np.maximum(sigma, 1e-12) ** 2


def _node_coordinate(shape: tuple[int, int], axis: str) -> np.ndarray:
    ny, nx = shape
    if axis == "y":
        return np.repeat(np.linspace(0.0, 1.0, ny), nx)
    return np.tile(np.linspace(0.0, 1.0, nx), ny)


def _normalized_mesh(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, ny),
        np.linspace(0.0, 1.0, nx),
        indexing="ij",
    )
    return yy.reshape(-1), xx.reshape(-1)


def _resolve_basis_axis(kind: str, axis: str) -> str:
    lower_kind = str(kind).lower()
    lower_axis = str(axis).lower()
    if lower_kind.startswith("y_"):
        return "y"
    if lower_kind.startswith("x_"):
        return "x"
    if lower_kind.startswith("radial"):
        return "radial"
    return lower_axis


def _resolve_radial_center(
    output_shape: tuple[int, int],
    *,
    center_i: int | None = None,
    center_j: int | None = None,
    center_fraction_x: float | None = None,
    center_fraction_y: float | None = None,
) -> tuple[float, float]:
    ny, nx = output_shape
    if center_i is not None:
        cy = 0.0 if ny <= 1 else float(center_i) / float(ny - 1)
    elif center_fraction_y is not None:
        cy = float(center_fraction_y)
    else:
        cy = 0.5

    if center_j is not None:
        cx = 0.0 if nx <= 1 else float(center_j) / float(nx - 1)
    elif center_fraction_x is not None:
        cx = float(center_fraction_x)
    else:
        cx = 0.5

    return float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0))


def _center_columns(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    centered = np.asarray(matrix, dtype=float) - np.mean(matrix, axis=0, keepdims=True)
    keep = np.linalg.norm(centered, axis=0) > 1e-12
    if not np.any(keep):
        return np.zeros((matrix.shape[0], 0), dtype=float)
    centered = centered[:, keep]
    norms = np.linalg.norm(centered, axis=0, keepdims=True)
    return centered / np.maximum(norms, 1e-12)


def _hierarchical_radial_feature_bank(
    output_shape: tuple[int, int],
    *,
    basis_width_scale: float,
    hierarchical_levels: int,
    center_i: int | None = None,
    center_j: int | None = None,
    center_fraction_x: float | None = None,
    center_fraction_y: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = _normalized_mesh(output_shape)
    cx, cy = _resolve_radial_center(
        output_shape,
        center_i=center_i,
        center_j=center_j,
        center_fraction_x=center_fraction_x,
        center_fraction_y=center_fraction_y,
    )
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    coords = radius / max(float(radius.max()), 1e-15)

    features: list[np.ndarray] = []
    levels: list[int] = []
    max_levels = max(1, int(hierarchical_levels))
    for level in range(max_levels):
        n_centers = max(1, 2**level)
        centers = (np.arange(n_centers, dtype=float) + 0.5) / n_centers
        spacing = 1.0 / n_centers
        sigma = max(float(basis_width_scale) * spacing, 1e-6)
        for center in centers:
            feature = np.exp(-0.5 * ((coords - center) / sigma) ** 2)
            features.append(feature)
            levels.append(level + 1)

    if not features:
        return np.zeros((coords.size, 0), dtype=float), np.zeros(0, dtype=int)

    stacked = np.column_stack(features)
    centered = _center_columns(stacked)
    if centered.shape[1] == 0:
        return centered, np.zeros(0, dtype=int)

    kept_levels = np.asarray(levels, dtype=int)[: centered.shape[1]]
    return centered, kept_levels


def _resolve_retained_rank(
    singular_values: np.ndarray,
    *,
    requested_rank: int,
    energy_threshold: float | None,
) -> int:
    if singular_values.size == 0 or requested_rank <= 0:
        return 0

    retained = min(int(requested_rank), singular_values.size)
    if energy_threshold is None:
        return retained

    threshold = float(np.clip(energy_threshold, 0.0, 1.0))
    if threshold <= 0.0:
        return retained

    energy = singular_values**2
    cumulative = np.cumsum(energy) / max(float(np.sum(energy)), 1e-30)
    min_rank = int(np.searchsorted(cumulative, threshold, side="left")) + 1
    return min(retained, max(1, min_rank))


def _build_basis_spec(
    output_shape: tuple[int, int],
    *,
    basis_type: str,
    axis: str,
    n_regions: int,
    basis_width_scale: float,
    center_i: int | None = None,
    center_j: int | None = None,
    center_fraction_x: float | None = None,
    center_fraction_y: float | None = None,
    hierarchical_levels: int | None = None,
    low_rank_rank: int | None = None,
    low_rank_energy: float | None = None,
    low_rank_singular_power: float = 1.0,
    low_rank_min_prior_scale: float = 0.15,
) -> dict[str, Any]:
    kind = str(basis_type).lower()
    resolved_axis = _resolve_basis_axis(kind, axis)

    if kind not in {
        "radial_hierarchical_lowrank",
        "hierarchical_lowrank_radial",
        "lowrank_radial",
        "radial_lowrank",
    }:
        basis = build_regional_basis(
            output_shape,
            basis_type=basis_type,
            axis=resolved_axis,
            n_regions=n_regions,
            basis_width_scale=basis_width_scale,
            center_i=center_i,
            center_j=center_j,
            center_fraction_x=center_fraction_x,
            center_fraction_y=center_fraction_y,
        )
        return {
            "basis": basis,
            "prior_mode": "repeat_global",
            "prior_std_scales": np.ones(basis.shape[1], dtype=float),
            "resolved_axis": resolved_axis,
            "raw_feature_count": int(basis.shape[1]),
            "retained_rank": int(basis.shape[1]),
            "retained_energy": 1.0,
            "hierarchical_levels": None,
            "singular_values": np.ones(basis.shape[1], dtype=float),
        }

    total_rank = max(1, int(n_regions))
    correction_rank_request = max(total_rank - 1, 0)
    levels = (
        max(2, int(np.ceil(np.log2(max(correction_rank_request, 1))) + 1))
        if hierarchical_levels is None
        else max(1, int(hierarchical_levels))
    )
    raw_bank, level_ids = _hierarchical_radial_feature_bank(
        output_shape,
        basis_width_scale=basis_width_scale,
        hierarchical_levels=levels,
        center_i=center_i,
        center_j=center_j,
        center_fraction_x=center_fraction_x,
        center_fraction_y=center_fraction_y,
    )

    if raw_bank.shape[1] == 0 or correction_rank_request == 0:
        basis = np.ones((output_shape[0] * output_shape[1], 1), dtype=float)
        return {
            "basis": basis,
            "prior_mode": "intercept_plus_zero",
            "prior_std_scales": np.ones(1, dtype=float),
            "resolved_axis": "radial",
            "raw_feature_count": int(raw_bank.shape[1]),
            "retained_rank": 1,
            "retained_energy": 1.0,
            "hierarchical_levels": levels,
            "singular_values": np.ones(1, dtype=float),
        }

    _, singular_values, right_vectors = np.linalg.svd(raw_bank, full_matrices=False)
    requested_rank = correction_rank_request if low_rank_rank is None else max(1, int(low_rank_rank) - 1)
    retained_correction_rank = _resolve_retained_rank(
        singular_values,
        requested_rank=min(requested_rank, correction_rank_request),
        energy_threshold=low_rank_energy,
    )

    correction_basis = raw_bank @ right_vectors[:retained_correction_rank, :].T
    correction_basis = _center_columns(correction_basis)
    retained_correction_rank = correction_basis.shape[1]

    if retained_correction_rank == 0:
        basis = np.ones((output_shape[0] * output_shape[1], 1), dtype=float)
        return {
            "basis": basis,
            "prior_mode": "intercept_plus_zero",
            "prior_std_scales": np.ones(1, dtype=float),
            "resolved_axis": "radial",
            "raw_feature_count": int(raw_bank.shape[1]),
            "retained_rank": 1,
            "retained_energy": 1.0,
            "hierarchical_levels": levels,
            "singular_values": np.ones(1, dtype=float),
        }

    basis = np.column_stack([np.ones(raw_bank.shape[0], dtype=float), correction_basis])
    singular_scale = singular_values[:retained_correction_rank] / max(float(singular_values[0]), 1e-15)
    singular_scale = np.maximum(singular_scale, float(low_rank_min_prior_scale)) ** max(
        float(low_rank_singular_power),
        1e-6,
    )
    prior_std_scales = np.concatenate([[1.0], singular_scale])
    retained_energy = float(
        np.sum(singular_values[:retained_correction_rank] ** 2) / max(np.sum(singular_values**2), 1e-30)
    )
    return {
        "basis": basis,
        "prior_mode": "intercept_plus_zero",
        "prior_std_scales": prior_std_scales,
        "resolved_axis": "radial",
        "raw_feature_count": int(raw_bank.shape[1]),
        "retained_rank": int(basis.shape[1]),
        "retained_energy": retained_energy,
        "hierarchical_levels": levels,
        "singular_values": singular_values[:retained_correction_rank],
        "raw_level_count": int(np.max(level_ids)) if level_ids.size else 0,
    }


def _basis_coordinate(
    output_shape: tuple[int, int],
    *,
    axis: str,
    center_i: int | None = None,
    center_j: int | None = None,
    center_fraction_x: float | None = None,
    center_fraction_y: float | None = None,
) -> np.ndarray:
    resolved_axis = _resolve_basis_axis(axis, axis)
    if resolved_axis == "radial":
        yy, xx = _normalized_mesh(output_shape)
        cx, cy = _resolve_radial_center(
            output_shape,
            center_i=center_i,
            center_j=center_j,
            center_fraction_x=center_fraction_x,
            center_fraction_y=center_fraction_y,
        )
        radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        return radius / max(float(radius.max()), 1e-15)
    return _node_coordinate(output_shape, resolved_axis)


def _weighted_rmse(residual: np.ndarray, weights: np.ndarray | None) -> float:
    values = np.asarray(residual, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    if weights is None:
        return float(np.sqrt(np.mean(values**2)))
    w = np.asarray(weights, dtype=float).reshape(-1)
    return float(np.sqrt(np.sum(w * values**2) / max(np.sum(w), 1e-30)))


def _gate_factor(value: float, *, threshold: float, scale: float) -> float:
    if scale <= 0.0:
        return 1.0
    excess = max(float(value) - float(threshold), 0.0)
    factor = 1.0 / (1.0 + excess / max(float(scale), 1e-9))
    return float(np.clip(factor, 0.0, 1.0))


def _tail_aggregate(
    values: list[float],
    *,
    tail_quantile: float,
    tail_weight: float,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 1.0, 1.0, 1.0
    median = float(np.median(arr))
    quantile = float(np.quantile(arr, np.clip(float(tail_quantile), 0.5, 1.0)))
    weight = float(np.clip(tail_weight, 0.0, 1.0))
    aggregate = (1.0 - weight) * median + weight * quantile
    return median, quantile, float(aggregate)


def _holdout_groups_from_order(
    order: np.ndarray,
    *,
    n_groups: int,
    contiguous: bool,
) -> list[np.ndarray]:
    ordered = np.asarray(order, dtype=int).reshape(-1)
    n_items = int(ordered.size)
    if n_items < 2 or int(n_groups) < 2:
        return []

    groups = max(2, min(int(n_groups), n_items))
    out: list[np.ndarray] = []
    if contiguous:
        edges = np.linspace(0, n_items, num=groups + 1, dtype=int)
        for start, stop in zip(edges[:-1], edges[1:]):
            if stop > start:
                out.append(ordered[start:stop])
        return out

    for offset in range(groups):
        subset = ordered[offset::groups]
        if subset.size:
            out.append(subset)
    return out


def _validation_gate_from_groups(
    *,
    flat_state: np.ndarray,
    idx: np.ndarray,
    targets: np.ndarray,
    local_weights: np.ndarray,
    holdout_groups: list[np.ndarray],
    support_coordinate_values: np.ndarray | None,
    min_train_constraints: int,
    ratio_threshold: float,
    ratio_scale: float,
    output_shape: tuple[int, int],
    basis_type: str,
    axis: str,
    requested_regions: int,
    effective_regions: int,
    basis_width_scale: float,
    prior_std_scale: float,
    min_constraints_per_region: int,
    center_i: int | None = None,
    center_j: int | None = None,
    center_fraction_x: float | None = None,
    center_fraction_y: float | None = None,
    hierarchical_levels: int | None = None,
    low_rank_rank: int | None = None,
    low_rank_energy: float | None = None,
    low_rank_singular_power: float = 1.0,
    low_rank_min_prior_scale: float = 0.15,
    reference_scale_field: np.ndarray | None = None,
    tail_quantile: float = 0.75,
    tail_weight: float = 0.0,
    excursion_threshold: float | None = None,
    excursion_scale: float = 0.25,
) -> dict[str, Any]:
    if not holdout_groups or ratio_scale <= 0.0:
        return {
            "factor": 1.0,
            "ratio": 1.0,
            "ratio_tail": 1.0,
            "regional_rmse": np.nan,
            "global_rmse": np.nan,
            "excursion_ratio": np.nan,
            "excursion_ratio_tail": np.nan,
            "excursion_factor": 1.0,
            "scale_field_variance": np.nan,
            "group_ranges": [],
            "group_factors": [],
            "group_ratios": [],
            "group_excursions": [],
            "n_folds_used": 0,
        }

    target_scale = max(float(np.sqrt(np.mean(np.asarray(targets, dtype=float) ** 2))), 1.0)
    error_floor = 1e-9 * target_scale
    ratios: list[float] = []
    regional_errors: list[float] = []
    global_errors: list[float] = []
    excursion_ratios: list[float] = []
    scale_fields: list[np.ndarray] = []
    group_ranges: list[tuple[float, float]] = []
    group_factors: list[float] = []
    group_ratio_values: list[float] = []
    group_excursion_values: list[float] = []
    ref_scale = None if reference_scale_field is None else np.asarray(reference_scale_field, dtype=float).reshape(-1)
    ref_scale_norm = (
        np.nan
        if ref_scale is None
        else max(float(np.sqrt(np.mean(ref_scale**2))), 1e-9)
    )
    support_coords = None if support_coordinate_values is None else np.asarray(support_coordinate_values, dtype=float).reshape(-1)

    full_regions = max(1, int(effective_regions))
    minimum_train = max(2, int(min_train_constraints))

    for hold_positions in holdout_groups:
        hold_pos = np.asarray(hold_positions, dtype=int).reshape(-1)
        if hold_pos.size == 0 or hold_pos.size >= idx.size:
            continue

        train_mask = np.ones(idx.size, dtype=bool)
        train_mask[hold_pos] = False
        train_size = int(np.count_nonzero(train_mask))
        if train_size < minimum_train:
            continue

        train_idx = idx[train_mask]
        train_targets = targets[train_mask]
        train_weights = local_weights[train_mask]
        hold_idx = idx[hold_pos]
        hold_targets = targets[hold_pos]
        hold_weights = local_weights[hold_pos]

        alpha_train, _ = estimate_alpha(flat_state, train_idx, train_targets, weights=train_weights)
        global_pred = alpha_train * flat_state[hold_idx]
        global_error = _weighted_rmse(global_pred - hold_targets, hold_weights)

        train_min_constraints = min(
            max(1, int(min_constraints_per_region)),
            max(1, train_size // full_regions),
        )
        regional_post = _regional_posterior_components(
            flat_state,
            train_idx,
            train_targets,
            output_shape=output_shape,
            weights=train_weights,
            observation_std=None,
            basis_type=basis_type,
            axis=axis,
            n_regions=full_regions,
            basis_width_scale=basis_width_scale,
            prior_std_scale=prior_std_scale,
            min_constraints_per_region=train_min_constraints,
            center_i=center_i,
            center_j=center_j,
            center_fraction_x=center_fraction_x,
            center_fraction_y=center_fraction_y,
            hierarchical_levels=hierarchical_levels,
            low_rank_rank=low_rank_rank,
            low_rank_energy=low_rank_energy,
            low_rank_singular_power=low_rank_singular_power,
            low_rank_min_prior_scale=low_rank_min_prior_scale,
        )
        hold_scale = regional_post["basis"][hold_idx, :] @ regional_post["coefficients"]
        scale_field = regional_post["basis"] @ regional_post["coefficients"]
        regional_pred = flat_state[hold_idx] * hold_scale
        regional_error = _weighted_rmse(regional_pred - hold_targets, hold_weights)

        if global_error <= error_floor and regional_error <= error_floor:
            ratio = 1.0
        else:
            ratio = regional_error / max(global_error, error_floor)

        ratios.append(float(ratio))
        regional_errors.append(float(regional_error))
        global_errors.append(float(global_error))
        scale_fields.append(np.asarray(scale_field, dtype=float))
        local_excursion = np.nan
        if ref_scale is not None:
            local_excursion = float(
                np.sqrt(np.mean((np.asarray(scale_field, dtype=float) - ref_scale) ** 2))
                / max(float(ref_scale_norm), 1e-9)
            )
            excursion_ratios.append(local_excursion)
        local_factor = _gate_factor(
            ratio,
            threshold=float(ratio_threshold),
            scale=float(ratio_scale),
        )
        if excursion_threshold is not None and excursion_scale > 0.0 and np.isfinite(local_excursion):
            local_factor *= _gate_factor(
                local_excursion,
                threshold=float(excursion_threshold),
                scale=float(excursion_scale),
            )
        group_factors.append(float(np.clip(local_factor, 0.0, 1.0)))
        group_ratio_values.append(float(ratio))
        group_excursion_values.append(float(local_excursion) if np.isfinite(local_excursion) else np.nan)
        if support_coords is not None and hold_pos.size:
            group_coord = support_coords[hold_pos]
            group_ranges.append((float(np.min(group_coord)), float(np.max(group_coord))))
        else:
            group_ranges.append((np.nan, np.nan))

    if not ratios:
        return {
            "factor": 1.0,
            "ratio": 1.0,
            "ratio_tail": 1.0,
            "regional_rmse": np.nan,
            "global_rmse": np.nan,
            "excursion_ratio": np.nan,
            "excursion_ratio_tail": np.nan,
            "excursion_factor": 1.0,
            "scale_field_variance": np.nan,
            "group_ranges": [],
            "group_factors": [],
            "group_ratios": [],
            "group_excursions": [],
            "n_folds_used": 0,
        }

    ratio_median, ratio_tail, aggregate_ratio = _tail_aggregate(
        ratios,
        tail_quantile=tail_quantile,
        tail_weight=tail_weight,
    )
    factor = _gate_factor(
        aggregate_ratio,
        threshold=float(ratio_threshold),
        scale=float(ratio_scale),
    )
    excursion_factor = 1.0
    excursion_ratio = np.nan
    excursion_ratio_tail = np.nan
    if excursion_ratios and excursion_threshold is not None and excursion_scale > 0.0:
        excursion_ratio, excursion_ratio_tail, aggregate_excursion = _tail_aggregate(
            excursion_ratios,
            tail_quantile=tail_quantile,
            tail_weight=tail_weight,
        )
        excursion_factor = _gate_factor(
            aggregate_excursion,
            threshold=float(excursion_threshold),
            scale=float(excursion_scale),
        )
        factor *= excursion_factor

    scale_field_variance = np.nan
    if len(scale_fields) >= 2:
        scale_stack = np.vstack(scale_fields)
        scale_field_variance = float(np.mean(np.var(scale_stack, axis=0, ddof=0)))
    return {
        "factor": float(np.clip(factor, 0.0, 1.0)),
        "ratio": float(aggregate_ratio),
        "ratio_median": float(ratio_median),
        "ratio_tail": float(ratio_tail),
        "regional_rmse": float(np.mean(regional_errors)),
        "global_rmse": float(np.mean(global_errors)),
        "excursion_ratio": excursion_ratio,
        "excursion_ratio_tail": excursion_ratio_tail,
        "excursion_factor": float(np.clip(excursion_factor, 0.0, 1.0)),
        "scale_field_variance": scale_field_variance,
        "group_ranges": group_ranges,
        "group_factors": group_factors,
        "group_ratios": group_ratio_values,
        "group_excursions": group_excursion_values,
        "n_folds_used": int(len(ratios)),
    }


def build_regional_basis(
    output_shape: tuple[int, int],
    *,
    basis_type: str = "x_gaussian",
    axis: str = "x",
    n_regions: int = 4,
    basis_width_scale: float = 1.25,
    center_i: int | None = None,
    center_j: int | None = None,
    center_fraction_x: float | None = None,
    center_fraction_y: float | None = None,
) -> np.ndarray:
    """Build a smooth regional basis for piecewise or Gaussian scale fields."""

    n_basis = max(1, int(n_regions))
    kind = basis_type.lower()
    resolved_axis = _resolve_basis_axis(kind, axis)

    if resolved_axis == "radial":
        yy, xx = _normalized_mesh(output_shape)
        cx, cy = _resolve_radial_center(
            output_shape,
            center_i=center_i,
            center_j=center_j,
            center_fraction_x=center_fraction_x,
            center_fraction_y=center_fraction_y,
        )
        radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_radius = float(radius.max())
        coords = radius / max(max_radius, 1e-15)
    else:
        coords = _node_coordinate(output_shape, resolved_axis)
    basis = np.zeros((coords.size, n_basis), dtype=float)

    if n_basis == 1:
        basis[:, 0] = 1.0
        return basis

    if kind in {
        "piecewise",
        "x_piecewise",
        "y_piecewise",
        "regional_piecewise",
        "radial_piecewise",
        "radial_shells",
    }:
        edges = np.linspace(0.0, 1.0, n_basis + 1)
        for region in range(n_basis):
            left = edges[region]
            right = edges[region + 1]
            if region == n_basis - 1:
                mask = (coords >= left) & (coords <= right)
            else:
                mask = (coords >= left) & (coords < right)
            basis[mask, region] = 1.0
        return basis

    if kind in {
        "gaussian",
        "x_gaussian",
        "y_gaussian",
        "regional_gaussian",
        "radial_gaussian",
        "radial_rbf",
    }:
        centers = np.linspace(0.0, 1.0, n_basis)
        spacing = centers[1] - centers[0] if n_basis > 1 else 1.0
        sigma = max(spacing * float(basis_width_scale), 1e-6)
        for region, center in enumerate(centers):
            basis[:, region] = np.exp(-0.5 * ((coords - center) / sigma) ** 2)
        row_sum = np.maximum(basis.sum(axis=1, keepdims=True), 1e-15)
        return basis / row_sum

    raise ValueError(f"Unknown regional basis type: {basis_type}")


def _regional_posterior_components(
    state: np.ndarray,
    indices: np.ndarray | list[int],
    target_values: np.ndarray | list[float],
    *,
    output_shape: tuple[int, int],
    weights: np.ndarray | list[float] | None = None,
    observation_std: np.ndarray | list[float] | None = None,
    basis_type: str = "x_gaussian",
    axis: str = "x",
    n_regions: int = 4,
    basis_width_scale: float = 1.25,
    prior_std_scale: float = 2.0,
    min_constraints_per_region: int = 4,
    center_i: int | None = None,
    center_j: int | None = None,
    center_fraction_x: float | None = None,
    center_fraction_y: float | None = None,
    hierarchical_levels: int | None = None,
    low_rank_rank: int | None = None,
    low_rank_energy: float | None = None,
    low_rank_singular_power: float = 1.0,
    low_rank_min_prior_scale: float = 0.15,
) -> dict[str, Any]:
    flat_state = np.asarray(state, dtype=float).reshape(-1)
    idx = np.asarray(indices, dtype=int)
    targets = np.asarray(target_values, dtype=float)
    resolved_axis = _resolve_basis_axis(basis_type, axis)
    effective_regions = max(1, min(int(n_regions), int(idx.size) // max(1, int(min_constraints_per_region))))
    if idx.size > 0 and effective_regions == 0:
        effective_regions = 1

    basis_spec = _build_basis_spec(
        output_shape,
        basis_type=basis_type,
        axis=resolved_axis,
        n_regions=effective_regions,
        basis_width_scale=basis_width_scale,
        center_i=center_i,
        center_j=center_j,
        center_fraction_x=center_fraction_x,
        center_fraction_y=center_fraction_y,
        hierarchical_levels=hierarchical_levels,
        low_rank_rank=low_rank_rank,
        low_rank_energy=low_rank_energy,
        low_rank_singular_power=low_rank_singular_power,
        low_rank_min_prior_scale=low_rank_min_prior_scale,
    )
    basis = np.asarray(basis_spec["basis"], dtype=float)

    local_weights = weights if weights is not None else _weights_from_observation_std(observation_std)
    if local_weights is None:
        local_weights = np.ones_like(targets)
    else:
        local_weights = np.asarray(local_weights, dtype=float)

    global_alpha, denominator = estimate_alpha(flat_state, idx, targets, weights=local_weights)
    global_alpha_std = float(1.0 / np.sqrt(max(denominator, 1e-30)))
    prior_mean = np.full(basis.shape[1], float(global_alpha), dtype=float)
    if str(basis_spec.get("prior_mode", "repeat_global")) == "intercept_plus_zero":
        prior_mean = np.zeros(basis.shape[1], dtype=float)
        prior_mean[0] = float(global_alpha)

    base_prior_std = max(abs(global_alpha), 1.0) * max(float(prior_std_scale), 1e-6)
    prior_std_scales = np.asarray(basis_spec.get("prior_std_scales", np.ones(basis.shape[1])), dtype=float)
    if prior_std_scales.size != basis.shape[1]:
        prior_std_scales = np.ones(basis.shape[1], dtype=float)
    prior_std_vector = base_prior_std * np.maximum(prior_std_scales, 1e-6)
    prior_precision = np.diag(1.0 / np.maximum(prior_std_vector, 1e-12) ** 2)

    design = flat_state[idx, None] * basis[idx, :]
    weighted_design = design * local_weights[:, None]
    precision = design.T @ weighted_design + prior_precision
    jitter = max(1e-12, float(np.trace(precision)) / max(1, precision.shape[0]) * 1e-12)
    precision = precision + np.eye(precision.shape[0], dtype=float) * jitter
    rhs = design.T @ (local_weights * targets) + prior_precision @ prior_mean

    try:
        coefficients = np.linalg.solve(precision, rhs)
        covariance = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(precision)
        coefficients = covariance @ rhs

    return {
        "flat_state": flat_state,
        "idx": idx,
        "targets": targets,
        "basis": basis,
        "coefficients": coefficients,
        "covariance": covariance,
        "prior_mean": prior_mean,
        "prior_std_vector": prior_std_vector,
        "global_alpha": float(global_alpha),
        "global_alpha_std": global_alpha_std,
        "denominator": float(denominator),
        "effective_regions": int(basis.shape[1]),
        "requested_regions": int(n_regions),
        "basis_type": basis_type,
        "axis": basis_spec["resolved_axis"],
        "basis_width_scale": float(basis_width_scale),
        "prior_std_scale": float(prior_std_scale),
        "min_constraints_per_region": int(min_constraints_per_region),
        "center_i": center_i,
        "center_j": center_j,
        "center_fraction_x": center_fraction_x,
        "center_fraction_y": center_fraction_y,
        "basis_rank": int(basis.shape[1]),
        "raw_feature_count": int(basis_spec.get("raw_feature_count", basis.shape[1])),
        "retained_energy": float(basis_spec.get("retained_energy", 1.0)),
        "hierarchical_levels": basis_spec.get("hierarchical_levels"),
        "low_rank_rank": low_rank_rank,
        "low_rank_energy": low_rank_energy,
        "low_rank_singular_power": float(low_rank_singular_power),
        "low_rank_min_prior_scale": float(low_rank_min_prior_scale),
    }


def _assemble_regional_result(
    *,
    flat_state: np.ndarray,
    idx: np.ndarray,
    targets: np.ndarray,
    output_shape: tuple[int, int],
    basis: np.ndarray,
    coefficients: np.ndarray,
    covariance: np.ndarray,
    denominator: float,
    method: str,
    metadata: dict[str, Any],
    global_alpha_std: float = 0.0,
    shrinkage_weight: float = 1.0,
) -> MCRResult:
    coefficient_std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    scale_field = basis @ coefficients
    reconstructed = flat_state * scale_field
    residual = reconstructed[idx] - targets
    residual_rmse = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0

    scale_variance_regional = np.einsum("ij,jk,ik->i", basis, covariance, basis)
    scale_variance = (max(0.0, 1.0 - float(shrinkage_weight)) ** 2) * (global_alpha_std**2)
    scale_variance = scale_variance + (float(shrinkage_weight) ** 2) * np.maximum(scale_variance_regional, 0.0)
    prediction_std = np.abs(flat_state) * np.sqrt(np.maximum(scale_variance, 0.0))

    return MCRResult(
        reconstructed=reconstructed.reshape(output_shape),
        alpha=float(np.mean(scale_field)),
        residual_rmse=residual_rmse,
        denominator=float(denominator),
        n_constraints=int(idx.size),
        method=method,
        scale_field=scale_field.reshape(output_shape),
        prediction_std=prediction_std.reshape(output_shape),
        coefficients=np.asarray(coefficients, dtype=float),
        coefficient_std=coefficient_std,
        posterior_trace=float(np.trace(covariance)),
        metadata=metadata,
    )


def apply_mcr(
    state: np.ndarray,
    indices: np.ndarray | list[int],
    target_values: np.ndarray | list[float],
    *,
    output_shape: tuple[int, int] | None = None,
    weights: np.ndarray | list[float] | None = None,
    observation_std: np.ndarray | list[float] | None = None,
) -> MCRResult:
    flat_state = np.asarray(state, dtype=float).reshape(-1)
    idx = np.asarray(indices, dtype=int)
    targets = np.asarray(target_values, dtype=float)
    local_weights = weights if weights is not None else _weights_from_observation_std(observation_std)
    alpha, denominator = estimate_alpha(flat_state, idx, targets, weights=local_weights)

    reconstructed = alpha * flat_state
    residual = reconstructed[idx] - targets
    residual_rmse = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
    scale_field = np.full_like(flat_state, fill_value=float(alpha), dtype=float)

    prediction_std = None
    if local_weights is not None:
        sigma_alpha = float(1.0 / np.sqrt(max(denominator, 1e-30)))
        prediction_std = np.abs(flat_state) * sigma_alpha

    if output_shape is not None:
        reconstructed = reconstructed.reshape(output_shape)
        scale_field = scale_field.reshape(output_shape)
        if prediction_std is not None:
            prediction_std = prediction_std.reshape(output_shape)

    return MCRResult(
        reconstructed=reconstructed,
        alpha=float(alpha),
        residual_rmse=residual_rmse,
        denominator=denominator,
        n_constraints=int(idx.size),
        method="global",
        scale_field=scale_field,
        prediction_std=prediction_std,
        coefficients=np.asarray([float(alpha)]),
        coefficient_std=None,
        posterior_trace=None,
        metadata={"basis_type": "global", "n_regions": 1},
    )


def apply_regional_mcr(
    state: np.ndarray,
    indices: np.ndarray | list[int],
    target_values: np.ndarray | list[float],
    *,
    output_shape: tuple[int, int],
    weights: np.ndarray | list[float] | None = None,
    observation_std: np.ndarray | list[float] | None = None,
    basis_type: str = "x_gaussian",
    axis: str = "x",
    n_regions: int = 4,
    basis_width_scale: float = 1.25,
    prior_std_scale: float = 2.0,
    min_constraints_per_region: int = 4,
    center_i: int | None = None,
    center_j: int | None = None,
    center_fraction_x: float | None = None,
    center_fraction_y: float | None = None,
    hierarchical_levels: int | None = None,
    low_rank_rank: int | None = None,
    low_rank_energy: float | None = None,
    low_rank_singular_power: float = 1.0,
    low_rank_min_prior_scale: float = 0.15,
) -> MCRResult:
    posterior = _regional_posterior_components(
        state,
        indices,
        target_values,
        output_shape=output_shape,
        weights=weights,
        observation_std=observation_std,
        basis_type=basis_type,
        axis=axis,
        n_regions=n_regions,
        basis_width_scale=basis_width_scale,
        prior_std_scale=prior_std_scale,
        min_constraints_per_region=min_constraints_per_region,
        center_i=center_i,
        center_j=center_j,
        center_fraction_x=center_fraction_x,
        center_fraction_y=center_fraction_y,
        hierarchical_levels=hierarchical_levels,
        low_rank_rank=low_rank_rank,
        low_rank_energy=low_rank_energy,
        low_rank_singular_power=low_rank_singular_power,
        low_rank_min_prior_scale=low_rank_min_prior_scale,
    )

    return _assemble_regional_result(
        flat_state=posterior["flat_state"],
        idx=posterior["idx"],
        targets=posterior["targets"],
        output_shape=output_shape,
        basis=posterior["basis"],
        coefficients=posterior["coefficients"],
        covariance=posterior["covariance"],
        denominator=posterior["denominator"],
        method="regional_bayesian",
        global_alpha_std=posterior["global_alpha_std"],
        shrinkage_weight=1.0,
        metadata={
            "basis_type": posterior["basis_type"],
            "axis": posterior["axis"],
            "n_regions": posterior["effective_regions"],
            "requested_n_regions": posterior["requested_regions"],
            "basis_width_scale": posterior["basis_width_scale"],
            "prior_std_scale": posterior["prior_std_scale"],
            "min_constraints_per_region": posterior["min_constraints_per_region"],
            "global_alpha": posterior["global_alpha"],
            "center_i": posterior["center_i"],
            "center_j": posterior["center_j"],
            "center_fraction_x": posterior["center_fraction_x"],
            "center_fraction_y": posterior["center_fraction_y"],
            "basis_rank": posterior["basis_rank"],
            "raw_feature_count": posterior["raw_feature_count"],
            "retained_energy": posterior["retained_energy"],
            "hierarchical_levels": posterior["hierarchical_levels"],
            "low_rank_rank": posterior["low_rank_rank"],
            "low_rank_energy": posterior["low_rank_energy"],
            "low_rank_singular_power": posterior["low_rank_singular_power"],
            "low_rank_min_prior_scale": posterior["low_rank_min_prior_scale"],
        },
    )


def apply_adaptive_regional_mcr(
    state: np.ndarray,
    indices: np.ndarray | list[int],
    target_values: np.ndarray | list[float],
    *,
    output_shape: tuple[int, int],
    weights: np.ndarray | list[float] | None = None,
    observation_std: np.ndarray | list[float] | None = None,
    basis_type: str = "x_gaussian",
    axis: str = "x",
    n_regions: int = 4,
    basis_width_scale: float = 1.25,
    prior_std_scale: float = 2.0,
    min_constraints_per_region: int = 4,
    center_i: int | None = None,
    center_j: int | None = None,
    center_fraction_x: float | None = None,
    center_fraction_y: float | None = None,
    hierarchical_levels: int | None = None,
    low_rank_rank: int | None = None,
    low_rank_energy: float | None = None,
    low_rank_singular_power: float = 1.0,
    low_rank_min_prior_scale: float = 0.15,
    min_shrinkage: float = 0.0,
    max_shrinkage: float = 0.98,
    support_power: float = 1.0,
    stability_cv_threshold: float = 0.25,
    stability_cv_scale: float = 0.75,
    stability_level_threshold: float | None = None,
    stability_level_scale: float = 0.75,
    anchor_jackknife_folds: int | None = None,
    anchor_jackknife_min_train_constraints: int = 0,
    anchor_jackknife_ratio_threshold: float = 1.1,
    anchor_jackknife_ratio_scale: float = 0.25,
    anchor_jackknife_tail_quantile: float = 0.75,
    anchor_jackknife_tail_weight: float = 0.0,
    anchor_jackknife_excursion_threshold: float | None = None,
    anchor_jackknife_excursion_scale: float = 0.25,
    stress_period_holdout_folds: int | None = None,
    stress_period_holdout_min_train_constraints: int = 0,
    stress_period_holdout_ratio_threshold: float = 1.05,
    stress_period_holdout_ratio_scale: float = 0.2,
    stress_period_holdout_tail_quantile: float = 0.75,
    stress_period_holdout_tail_weight: float = 0.0,
    stress_period_holdout_excursion_threshold: float | None = None,
    stress_period_holdout_excursion_scale: float = 0.25,
    validation_variance_scale: float = 1.0,
    validation_hard_gate_threshold: float | None = None,
    external_validation_factor: float = 1.0,
) -> MCRResult:
    posterior = _regional_posterior_components(
        state,
        indices,
        target_values,
        output_shape=output_shape,
        weights=weights,
        observation_std=observation_std,
        basis_type=basis_type,
        axis=axis,
        n_regions=n_regions,
        basis_width_scale=basis_width_scale,
        prior_std_scale=prior_std_scale,
        min_constraints_per_region=min_constraints_per_region,
        center_i=center_i,
        center_j=center_j,
        center_fraction_x=center_fraction_x,
        center_fraction_y=center_fraction_y,
        hierarchical_levels=hierarchical_levels,
        low_rank_rank=low_rank_rank,
        low_rank_energy=low_rank_energy,
        low_rank_singular_power=low_rank_singular_power,
        low_rank_min_prior_scale=low_rank_min_prior_scale,
    )
    local_weights = weights if weights is not None else _weights_from_observation_std(observation_std)
    if local_weights is None:
        local_weights = np.ones_like(np.asarray(posterior["targets"], dtype=float))
    else:
        local_weights = np.asarray(local_weights, dtype=float)

    coefficients = np.asarray(posterior["coefficients"], dtype=float)
    covariance = np.asarray(posterior["covariance"], dtype=float)
    basis = np.asarray(posterior["basis"], dtype=float)
    prior_mean = np.asarray(posterior["prior_mean"], dtype=float)
    global_alpha = float(posterior["global_alpha"])
    deviations = coefficients - prior_mean
    scale_field = basis @ coefficients
    prior_scale_field = basis @ prior_mean
    signal_variance = float(np.mean((scale_field - prior_scale_field) ** 2))
    scale_variance_nodes = np.einsum("ij,jk,ik->i", basis, covariance, basis)
    noise_variance = float(np.mean(np.maximum(scale_variance_nodes, 0.0)))
    scale_mean = float(np.mean(scale_field))
    scale_cv_raw = (
        np.inf
        if abs(scale_mean) < 1e-30
        else float(np.std(scale_field) / abs(scale_mean))
    )
    prior_scale_mean = float(np.mean(prior_scale_field))
    scale_level_ratio_raw = float(
        np.sqrt(max(signal_variance, 0.0)) / max(abs(prior_scale_mean), 1e-9)
    )
    excess_cv = max(scale_cv_raw - float(stability_cv_threshold), 0.0)
    stability_factor = 1.0 / (1.0 + excess_cv / max(float(stability_cv_scale), 1e-9))
    level_factor = 1.0
    if stability_level_threshold is not None:
        excess_level = max(scale_level_ratio_raw - float(stability_level_threshold), 0.0)
        level_factor = 1.0 / (1.0 + excess_level / max(float(stability_level_scale), 1e-9))
    support_ratio = min(
        1.0,
        posterior["idx"].size
        / max(1.0, posterior["effective_regions"] * posterior["min_constraints_per_region"]),
    )
    coordinate = _basis_coordinate(
        output_shape,
        axis=posterior["axis"],
        center_i=center_i,
        center_j=center_j,
        center_fraction_x=center_fraction_x,
        center_fraction_y=center_fraction_y,
    )
    anchor_order = np.argsort(coordinate[posterior["idx"]], kind="stable")
    jackknife_gate = _validation_gate_from_groups(
        flat_state=posterior["flat_state"],
        idx=posterior["idx"],
        targets=posterior["targets"],
        local_weights=local_weights,
        holdout_groups=_holdout_groups_from_order(
            anchor_order,
            n_groups=0 if anchor_jackknife_folds is None else int(anchor_jackknife_folds),
            contiguous=False,
        ),
        support_coordinate_values=coordinate[posterior["idx"]],
        min_train_constraints=max(
            int(anchor_jackknife_min_train_constraints),
            posterior["effective_regions"] * max(1, posterior["min_constraints_per_region"] // 2),
        ),
        ratio_threshold=float(anchor_jackknife_ratio_threshold),
        ratio_scale=float(anchor_jackknife_ratio_scale),
        output_shape=output_shape,
        basis_type=basis_type,
        axis=posterior["axis"],
        requested_regions=posterior["requested_regions"],
        effective_regions=posterior["effective_regions"],
        basis_width_scale=posterior["basis_width_scale"],
        prior_std_scale=posterior["prior_std_scale"],
        min_constraints_per_region=posterior["min_constraints_per_region"],
        center_i=center_i,
        center_j=center_j,
        center_fraction_x=center_fraction_x,
        center_fraction_y=center_fraction_y,
        hierarchical_levels=hierarchical_levels,
        low_rank_rank=low_rank_rank,
        low_rank_energy=low_rank_energy,
        low_rank_singular_power=low_rank_singular_power,
        low_rank_min_prior_scale=low_rank_min_prior_scale,
        reference_scale_field=prior_scale_field,
        tail_quantile=float(anchor_jackknife_tail_quantile),
        tail_weight=float(anchor_jackknife_tail_weight),
        excursion_threshold=anchor_jackknife_excursion_threshold,
        excursion_scale=float(anchor_jackknife_excursion_scale),
    )
    stress_gate = _validation_gate_from_groups(
        flat_state=posterior["flat_state"],
        idx=posterior["idx"],
        targets=posterior["targets"],
        local_weights=local_weights,
        # In radial/transient settings, contiguous coordinate blocks act as a proxy
        # for phase-specific support across the stress-period trajectory.
        holdout_groups=_holdout_groups_from_order(
            anchor_order,
            n_groups=0 if stress_period_holdout_folds is None else int(stress_period_holdout_folds),
            contiguous=True,
        ),
        support_coordinate_values=coordinate[posterior["idx"]],
        min_train_constraints=max(
            int(stress_period_holdout_min_train_constraints),
            posterior["effective_regions"] * max(1, posterior["min_constraints_per_region"] // 2),
        ),
        ratio_threshold=float(stress_period_holdout_ratio_threshold),
        ratio_scale=float(stress_period_holdout_ratio_scale),
        output_shape=output_shape,
        basis_type=basis_type,
        axis=posterior["axis"],
        requested_regions=posterior["requested_regions"],
        effective_regions=posterior["effective_regions"],
        basis_width_scale=posterior["basis_width_scale"],
        prior_std_scale=posterior["prior_std_scale"],
        min_constraints_per_region=posterior["min_constraints_per_region"],
        center_i=center_i,
        center_j=center_j,
        center_fraction_x=center_fraction_x,
        center_fraction_y=center_fraction_y,
        hierarchical_levels=hierarchical_levels,
        low_rank_rank=low_rank_rank,
        low_rank_energy=low_rank_energy,
        low_rank_singular_power=low_rank_singular_power,
        low_rank_min_prior_scale=low_rank_min_prior_scale,
        reference_scale_field=prior_scale_field,
        tail_quantile=float(stress_period_holdout_tail_quantile),
        tail_weight=float(stress_period_holdout_tail_weight),
        excursion_threshold=stress_period_holdout_excursion_threshold,
        excursion_scale=float(stress_period_holdout_excursion_scale),
    )
    validation_noise_variance = max(
        0.0,
        float(np.nan_to_num(jackknife_gate.get("scale_field_variance", np.nan), nan=0.0)),
        float(np.nan_to_num(stress_gate.get("scale_field_variance", np.nan), nan=0.0)),
    ) * max(float(validation_variance_scale), 0.0)
    validation_factor = (
        float(jackknife_gate["factor"])
        * float(stress_gate["factor"])
        * float(np.clip(external_validation_factor, 0.0, 1.0))
    )
    effective_noise_variance = float(noise_variance) + float(validation_noise_variance)

    if signal_variance <= 1e-30 and noise_variance <= 1e-30:
        shrinkage_weight = 0.0
    else:
        empirical_weight = signal_variance / (signal_variance + effective_noise_variance + 1e-30)
        support_ratio = support_ratio ** max(float(support_power), 1e-6)
        shrinkage_weight = empirical_weight * support_ratio * stability_factor * level_factor * validation_factor
        shrinkage_weight = float(np.clip(shrinkage_weight, min_shrinkage, max_shrinkage))
    hard_fallback_applied = False
    if (
        validation_hard_gate_threshold is not None
        and float(validation_factor) <= float(validation_hard_gate_threshold)
    ):
        shrinkage_weight = 0.0
        hard_fallback_applied = True

    shrunken_coefficients = prior_mean + shrinkage_weight * deviations
    shrunken_covariance = covariance * (shrinkage_weight**2)

    return _assemble_regional_result(
        flat_state=posterior["flat_state"],
        idx=posterior["idx"],
        targets=posterior["targets"],
        output_shape=output_shape,
        basis=posterior["basis"],
        coefficients=shrunken_coefficients,
        covariance=shrunken_covariance,
        denominator=posterior["denominator"],
        method="regional_adaptive_eb",
        global_alpha_std=posterior["global_alpha_std"],
        shrinkage_weight=shrinkage_weight,
        metadata={
            "basis_type": posterior["basis_type"],
            "axis": posterior["axis"],
            "n_regions": posterior["effective_regions"],
            "requested_n_regions": posterior["requested_regions"],
            "basis_width_scale": posterior["basis_width_scale"],
            "prior_std_scale": posterior["prior_std_scale"],
            "min_constraints_per_region": posterior["min_constraints_per_region"],
            "global_alpha": posterior["global_alpha"],
            "center_i": posterior["center_i"],
            "center_j": posterior["center_j"],
            "center_fraction_x": posterior["center_fraction_x"],
            "center_fraction_y": posterior["center_fraction_y"],
            "basis_rank": posterior["basis_rank"],
            "raw_feature_count": posterior["raw_feature_count"],
            "retained_energy": posterior["retained_energy"],
            "hierarchical_levels": posterior["hierarchical_levels"],
            "low_rank_rank": posterior["low_rank_rank"],
            "low_rank_energy": posterior["low_rank_energy"],
            "low_rank_singular_power": posterior["low_rank_singular_power"],
            "low_rank_min_prior_scale": posterior["low_rank_min_prior_scale"],
            "adaptive_shrinkage_weight": shrinkage_weight,
            "adaptive_signal_variance": signal_variance,
            "adaptive_noise_variance": noise_variance,
            "adaptive_effective_noise_variance": effective_noise_variance,
            "adaptive_validation_noise_variance": validation_noise_variance,
            "adaptive_support_ratio": support_ratio,
            "adaptive_support_power": float(support_power),
            "adaptive_min_shrinkage": float(min_shrinkage),
            "adaptive_max_shrinkage": float(max_shrinkage),
            "adaptive_stability_factor": stability_factor,
            "adaptive_level_stability_factor": level_factor,
            "adaptive_validation_factor": validation_factor,
            "adaptive_external_validation_factor": float(np.clip(external_validation_factor, 0.0, 1.0)),
            "adaptive_hard_fallback_applied": hard_fallback_applied,
            "adaptive_validation_hard_gate_threshold": validation_hard_gate_threshold,
            "adaptive_anchor_jackknife_factor": jackknife_gate["factor"],
            "adaptive_anchor_jackknife_ratio": jackknife_gate["ratio"],
            "adaptive_anchor_jackknife_ratio_tail": jackknife_gate.get("ratio_tail"),
            "adaptive_anchor_jackknife_regional_rmse": jackknife_gate["regional_rmse"],
            "adaptive_anchor_jackknife_global_rmse": jackknife_gate["global_rmse"],
            "adaptive_anchor_jackknife_excursion_ratio": jackknife_gate.get("excursion_ratio"),
            "adaptive_anchor_jackknife_excursion_ratio_tail": jackknife_gate.get("excursion_ratio_tail"),
            "adaptive_anchor_jackknife_excursion_factor": jackknife_gate.get("excursion_factor"),
            "adaptive_anchor_jackknife_scale_field_variance": jackknife_gate.get("scale_field_variance"),
            "adaptive_anchor_jackknife_group_ranges": jackknife_gate.get("group_ranges"),
            "adaptive_anchor_jackknife_group_factors": jackknife_gate.get("group_factors"),
            "adaptive_anchor_jackknife_group_ratios": jackknife_gate.get("group_ratios"),
            "adaptive_anchor_jackknife_group_excursions": jackknife_gate.get("group_excursions"),
            "adaptive_anchor_jackknife_folds_used": jackknife_gate["n_folds_used"],
            "adaptive_stress_holdout_factor": stress_gate["factor"],
            "adaptive_stress_holdout_ratio": stress_gate["ratio"],
            "adaptive_stress_holdout_ratio_tail": stress_gate.get("ratio_tail"),
            "adaptive_stress_holdout_regional_rmse": stress_gate["regional_rmse"],
            "adaptive_stress_holdout_global_rmse": stress_gate["global_rmse"],
            "adaptive_stress_holdout_excursion_ratio": stress_gate.get("excursion_ratio"),
            "adaptive_stress_holdout_excursion_ratio_tail": stress_gate.get("excursion_ratio_tail"),
            "adaptive_stress_holdout_excursion_factor": stress_gate.get("excursion_factor"),
            "adaptive_stress_holdout_scale_field_variance": stress_gate.get("scale_field_variance"),
            "adaptive_stress_holdout_group_ranges": stress_gate.get("group_ranges"),
            "adaptive_stress_holdout_group_factors": stress_gate.get("group_factors"),
            "adaptive_stress_holdout_group_ratios": stress_gate.get("group_ratios"),
            "adaptive_stress_holdout_group_excursions": stress_gate.get("group_excursions"),
            "adaptive_stress_holdout_folds_used": stress_gate["n_folds_used"],
            "adaptive_scale_cv_raw": scale_cv_raw,
            "adaptive_scale_level_ratio_raw": scale_level_ratio_raw,
            "adaptive_stability_cv_threshold": float(stability_cv_threshold),
            "adaptive_stability_cv_scale": float(stability_cv_scale),
            "adaptive_stability_level_threshold": stability_level_threshold,
            "adaptive_stability_level_scale": float(stability_level_scale),
            "adaptive_anchor_jackknife_folds": anchor_jackknife_folds,
            "adaptive_anchor_jackknife_min_train_constraints": int(anchor_jackknife_min_train_constraints),
            "adaptive_anchor_jackknife_ratio_threshold": float(anchor_jackknife_ratio_threshold),
            "adaptive_anchor_jackknife_ratio_scale": float(anchor_jackknife_ratio_scale),
            "adaptive_anchor_jackknife_tail_quantile": float(anchor_jackknife_tail_quantile),
            "adaptive_anchor_jackknife_tail_weight": float(anchor_jackknife_tail_weight),
            "adaptive_anchor_jackknife_excursion_threshold": anchor_jackknife_excursion_threshold,
            "adaptive_anchor_jackknife_excursion_scale": float(anchor_jackknife_excursion_scale),
            "adaptive_stress_holdout_folds": stress_period_holdout_folds,
            "adaptive_stress_holdout_min_train_constraints": int(stress_period_holdout_min_train_constraints),
            "adaptive_stress_holdout_ratio_threshold": float(stress_period_holdout_ratio_threshold),
            "adaptive_stress_holdout_ratio_scale": float(stress_period_holdout_ratio_scale),
            "adaptive_stress_holdout_tail_quantile": float(stress_period_holdout_tail_quantile),
            "adaptive_stress_holdout_tail_weight": float(stress_period_holdout_tail_weight),
            "adaptive_stress_holdout_excursion_threshold": stress_period_holdout_excursion_threshold,
            "adaptive_stress_holdout_excursion_scale": float(stress_period_holdout_excursion_scale),
            "adaptive_validation_variance_scale": float(validation_variance_scale),
        },
    )


def apply_mcr_method(
    state: np.ndarray,
    indices: np.ndarray | list[int],
    target_values: np.ndarray | list[float],
    *,
    output_shape: tuple[int, int],
    method_config: dict[str, Any] | None = None,
    weights: np.ndarray | list[float] | None = None,
    observation_std: np.ndarray | list[float] | None = None,
) -> MCRResult:
    config = method_config or {"type": "global"}
    method_type = str(config.get("type", "global")).lower()

    if method_type in {"global", "mcr", "scalar"}:
        result = apply_mcr(
            state,
            indices,
            target_values,
            output_shape=output_shape,
            weights=weights,
            observation_std=observation_std,
        )
        result.metadata["method_name"] = str(config.get("name", "global"))
        return result

    if method_type in {"regional", "regional_bayesian", "regional_extension"}:
        result = apply_regional_mcr(
            state,
            indices,
            target_values,
            output_shape=output_shape,
            weights=weights,
            observation_std=observation_std,
            basis_type=str(config.get("basis", "x_gaussian")),
            axis=str(config.get("axis", "x")),
            n_regions=int(config.get("n_regions", 4)),
            basis_width_scale=float(config.get("basis_width_scale", 1.25)),
            prior_std_scale=float(config.get("prior_std_scale", 2.0)),
            min_constraints_per_region=int(config.get("min_constraints_per_region", 4)),
            center_i=None if config.get("center_i") is None else int(config.get("center_i")),
            center_j=None if config.get("center_j") is None else int(config.get("center_j")),
            center_fraction_x=None
            if config.get("center_fraction_x") is None
            else float(config.get("center_fraction_x")),
            center_fraction_y=None
            if config.get("center_fraction_y") is None
            else float(config.get("center_fraction_y")),
            hierarchical_levels=None
            if config.get("hierarchical_levels") is None
            else int(config.get("hierarchical_levels")),
            low_rank_rank=None
            if config.get("low_rank_rank") is None
            else int(config.get("low_rank_rank")),
            low_rank_energy=None
            if config.get("low_rank_energy") is None
            else float(config.get("low_rank_energy")),
            low_rank_singular_power=float(config.get("low_rank_singular_power", 1.0)),
            low_rank_min_prior_scale=float(config.get("low_rank_min_prior_scale", 0.15)),
        )
        result.metadata["method_name"] = str(config.get("name", "regional"))
        return result

    if method_type in {"regional_adaptive_eb", "adaptive_regional", "regional_shrinkage_eb"}:
        result = apply_adaptive_regional_mcr(
            state,
            indices,
            target_values,
            output_shape=output_shape,
            weights=weights,
            observation_std=observation_std,
            basis_type=str(config.get("basis", "x_gaussian")),
            axis=str(config.get("axis", "x")),
            n_regions=int(config.get("n_regions", 4)),
            basis_width_scale=float(config.get("basis_width_scale", 1.25)),
            prior_std_scale=float(config.get("prior_std_scale", 2.0)),
            min_constraints_per_region=int(config.get("min_constraints_per_region", 4)),
            center_i=None if config.get("center_i") is None else int(config.get("center_i")),
            center_j=None if config.get("center_j") is None else int(config.get("center_j")),
            center_fraction_x=None
            if config.get("center_fraction_x") is None
            else float(config.get("center_fraction_x")),
            center_fraction_y=None
            if config.get("center_fraction_y") is None
            else float(config.get("center_fraction_y")),
            hierarchical_levels=None
            if config.get("hierarchical_levels") is None
            else int(config.get("hierarchical_levels")),
            low_rank_rank=None
            if config.get("low_rank_rank") is None
            else int(config.get("low_rank_rank")),
            low_rank_energy=None
            if config.get("low_rank_energy") is None
            else float(config.get("low_rank_energy")),
            low_rank_singular_power=float(config.get("low_rank_singular_power", 1.0)),
            low_rank_min_prior_scale=float(config.get("low_rank_min_prior_scale", 0.15)),
            min_shrinkage=float(config.get("min_shrinkage", 0.0)),
            max_shrinkage=float(config.get("max_shrinkage", 0.98)),
            support_power=float(config.get("support_power", 1.0)),
            stability_cv_threshold=float(config.get("stability_cv_threshold", 0.25)),
            stability_cv_scale=float(config.get("stability_cv_scale", 0.75)),
            stability_level_threshold=None
            if config.get("stability_level_threshold") is None
            else float(config.get("stability_level_threshold")),
            stability_level_scale=float(config.get("stability_level_scale", 0.75)),
            anchor_jackknife_folds=None
            if config.get("anchor_jackknife_folds") is None
            else int(config.get("anchor_jackknife_folds")),
            anchor_jackknife_min_train_constraints=int(config.get("anchor_jackknife_min_train_constraints", 0)),
            anchor_jackknife_ratio_threshold=float(config.get("anchor_jackknife_ratio_threshold", 1.1)),
            anchor_jackknife_ratio_scale=float(config.get("anchor_jackknife_ratio_scale", 0.25)),
            anchor_jackknife_tail_quantile=float(config.get("anchor_jackknife_tail_quantile", 0.75)),
            anchor_jackknife_tail_weight=float(config.get("anchor_jackknife_tail_weight", 0.0)),
            anchor_jackknife_excursion_threshold=None
            if config.get("anchor_jackknife_excursion_threshold") is None
            else float(config.get("anchor_jackknife_excursion_threshold")),
            anchor_jackknife_excursion_scale=float(config.get("anchor_jackknife_excursion_scale", 0.25)),
            stress_period_holdout_folds=None
            if config.get("stress_period_holdout_folds") is None
            else int(config.get("stress_period_holdout_folds")),
            stress_period_holdout_min_train_constraints=int(config.get("stress_period_holdout_min_train_constraints", 0)),
            stress_period_holdout_ratio_threshold=float(config.get("stress_period_holdout_ratio_threshold", 1.05)),
            stress_period_holdout_ratio_scale=float(config.get("stress_period_holdout_ratio_scale", 0.2)),
            stress_period_holdout_tail_quantile=float(config.get("stress_period_holdout_tail_quantile", 0.75)),
            stress_period_holdout_tail_weight=float(config.get("stress_period_holdout_tail_weight", 0.0)),
            stress_period_holdout_excursion_threshold=None
            if config.get("stress_period_holdout_excursion_threshold") is None
            else float(config.get("stress_period_holdout_excursion_threshold")),
            stress_period_holdout_excursion_scale=float(config.get("stress_period_holdout_excursion_scale", 0.25)),
            validation_variance_scale=float(config.get("validation_variance_scale", 1.0)),
            validation_hard_gate_threshold=None
            if config.get("validation_hard_gate_threshold") is None
            else float(config.get("validation_hard_gate_threshold")),
            external_validation_factor=float(config.get("external_validation_factor", 1.0)),
        )
        result.metadata["method_name"] = str(config.get("name", "adaptive_regional"))
        return result

    raise ValueError(f"Unknown MCR method type: {method_type}")
