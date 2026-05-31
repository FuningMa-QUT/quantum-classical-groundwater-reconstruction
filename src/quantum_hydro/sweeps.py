"""Batch sweeps over heterogeneity, noise, and MCR constraint density."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .benchmarks import merge_nested_dicts
from .config import load_config
from .experiments import run_experiment
from .public_benchmarks import resolve_any_benchmark_case


def _slug(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _existing_columns(df: pd.DataFrame, names: list[str]) -> list[str]:
    return [name for name in names if name in df.columns]


def _benchmark_columns(df: pd.DataFrame) -> list[str]:
    return _existing_columns(df, ["benchmark_family", "benchmark_name", "benchmark_label"])


def _concentration_key_columns(df: pd.DataFrame) -> list[str]:
    return _benchmark_columns(df) + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "transport_quantum_mode",
        "constraint_type",
        "constraint_name",
        "mcr_method_name",
        "mcr_method_type",
        "n_constraints",
    ]


def _merge_concentration_and_diagnostics(
    table: pd.DataFrame,
    diagnostics_table: pd.DataFrame,
) -> pd.DataFrame:
    concentration = table[table["field"] == "concentration"].copy()
    concentration = _preferred_transport_subset(concentration)
    diagnostics = _preferred_transport_subset(diagnostics_table.copy())
    if concentration.empty:
        return concentration
    if diagnostics.empty:
        return concentration

    merge_cols = [column for column in _concentration_key_columns(concentration) if column in diagnostics.columns]
    diag_value_cols = [
        column
        for column in diagnostics.columns
        if column not in merge_cols and column != "n_runs"
    ]
    merged = concentration.merge(
        diagnostics.loc[:, merge_cols + diag_value_cols],
        on=merge_cols,
        how="left",
        suffixes=("", "_diag"),
    )
    return merged


def _relative_regret(values: pd.Series, best_values: pd.Series) -> pd.Series:
    value_arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    best_arr = pd.to_numeric(best_values, errors="coerce").to_numpy(dtype=float)
    out = np.full(value_arr.shape, np.nan, dtype=float)
    finite = np.isfinite(value_arr) & np.isfinite(best_arr)
    scale_mask = finite & (np.abs(best_arr) > 1e-30)
    out[scale_mask] = (value_arr[scale_mask] - best_arr[scale_mask]) / np.abs(best_arr[scale_mask])
    zero_mask = finite & ~scale_mask
    out[zero_mask] = value_arr[zero_mask] - best_arr[zero_mask]
    return pd.Series(out, index=values.index)


def _resolved_benchmark_cases(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    raw_cases = sweep.get("benchmark_cases")
    if raw_cases is None:
        return [{}]
    return [resolve_any_benchmark_case(case) for case in raw_cases]


def _apply_benchmark_case(
    base_config: dict[str, Any],
    benchmark_case: dict[str, Any] | None,
) -> dict[str, Any]:
    if not benchmark_case:
        return deepcopy(base_config)

    case_override = {key: value for key, value in benchmark_case.items() if key != "name"}
    merged = merge_nested_dicts(base_config, case_override)
    benchmark_meta = deepcopy(merged.get("benchmark", {}))
    benchmark_meta.setdefault("name", str(benchmark_case.get("name", benchmark_meta.get("name", "benchmark"))))
    benchmark_meta.setdefault("label", str(benchmark_meta["name"]).replace("_", " ").title())
    benchmark_meta.setdefault("family", str(merged.get("k_field", {}).get("model", "lognormal")))
    merged["benchmark"] = benchmark_meta
    return merged


def _constraint_sets(
    counts: list[int],
    *,
    transport_strategy: str = "source_control_plane_hybrid",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flow_sets: list[dict[str, Any]] = [
        {"name": "boundary", "type": "dirichlet_boundary"},
        {"name": "corners", "type": "corners"},
        {"name": "sources", "type": "sources"},
    ]
    transport_sets: list[dict[str, Any]] = [{"name": "sources", "type": "sources"}]
    strategy = str(transport_strategy).lower()

    if strategy in {"hybrid", "source_control_plane_hybrid", "hybrid_monitoring"}:
        transport_constraint_type = "source_control_plane_hybrid"
    elif strategy in {"pathline", "pathline_monitoring"}:
        transport_constraint_type = "pathline_monitoring"
    elif strategy in {"control_plane", "control_plane_monitoring"}:
        transport_constraint_type = "control_plane_monitoring"
    else:
        raise ValueError(f"Unknown transport constraint strategy: {transport_strategy}")

    for count in counts:
        if count > 0:
            flow_item = {"name": f"random{count}", "type": "random_internal", "m": int(count)}
            transport_item = {
                "name": f"{strategy}{count}",
                "type": transport_constraint_type,
                "m": int(count),
            }
            flow_sets.append(flow_item)
            transport_sets.append(transport_item)

    return flow_sets, transport_sets


def _make_run_config(
    base_config: dict[str, Any],
    *,
    sweep_name: str,
    log_variance: float,
    k_seed: int,
    quantum_noise: dict[str, Any],
    observation_noise_relative: float,
    observation_noise_absolute: float,
    constraint_seed: int,
    constraint_counts: list[int],
    benchmark_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _apply_benchmark_case(base_config, benchmark_case)
    noise_name = str(quantum_noise["name"])
    benchmark_name = ""
    if config.get("benchmark"):
        benchmark_name = f"{_slug(config['benchmark'].get('name', 'benchmark'))}_"
    run_name = (
        f"{sweep_name}_{benchmark_name}var{_slug(log_variance)}_k{k_seed}_"
        f"q{_slug(noise_name)}_obs{_slug(observation_noise_relative)}_c{constraint_seed}"
    )

    config["name"] = run_name
    config.setdefault("k_field", {})
    config["k_field"].setdefault("model", "lognormal")
    config["k_field"]["seed"] = int(k_seed)
    config["k_field"]["log_variance"] = float(log_variance)

    solver_config = deepcopy(quantum_noise.get("solver", {"type": "exact"}))
    if solver_config.get("type", "exact") != "exact":
        solver_config.setdefault("seed", int(k_seed + constraint_seed))
    config["quantum_solver"] = solver_config
    config.setdefault("transport_quantum", {})
    config["transport_quantum"].setdefault("mode", "operator_stepwise_hybrid")
    config["transport_quantum"].setdefault("observation_stride", 1)
    config["transport_quantum"].setdefault("store_trace_metrics", True)

    flow_sets, transport_sets = _constraint_sets(
        constraint_counts,
        transport_strategy=str(base_config.get("sweep_options", {}).get("transport_constraint_strategy", "source_control_plane_hybrid")),
    )
    config.setdefault("mcr", {})
    config["mcr"]["flow_constraint_sets"] = flow_sets
    config["mcr"]["transport_constraint_sets"] = transport_sets
    config["mcr"]["observation_noise"] = {
        "relative_std": float(observation_noise_relative),
        "absolute_std": float(observation_noise_absolute),
        "seed": int(constraint_seed),
    }

    return config


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_main_table(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    numeric_columns = [
        "n_constraints",
        "log_variance",
        "observation_noise_relative",
        "head_rmse",
        "concentration_rmse",
        "concentration_mass_relative_error",
        "concentration_peak_bias_percent",
        "concentration_centroid_distance_error",
        "concentration_prediction_std_mean",
        "concentration_coverage_2sigma",
        "mcr_shrinkage_weight",
    ]
    df = _to_numeric(df.copy(), numeric_columns)
    df["primary_rmse"] = np.where(
        df["field"] == "head",
        df["head_rmse"],
        df["concentration_rmse"],
    )
    df["abs_mass_relative_error"] = df["concentration_mass_relative_error"].abs()
    df["abs_peak_bias_percent"] = df["concentration_peak_bias_percent"].abs()

    group_cols = _benchmark_columns(df) + [
        "field",
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "transport_quantum_mode",
        "constraint_type",
        "constraint_name",
        "mcr_method_name",
        "mcr_method_type",
        "n_constraints",
    ]

    def p95(values: pd.Series) -> float:
        clean = values.dropna().to_numpy(dtype=float)
        if clean.size == 0:
            return np.nan
        return float(np.percentile(clean, 95))

    table = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("primary_rmse", "count"),
            rmse_mean=("primary_rmse", "mean"),
            rmse_std=("primary_rmse", "std"),
            rmse_p95=("primary_rmse", p95),
            mass_rel_error_abs_mean=("abs_mass_relative_error", "mean"),
            mass_rel_error_abs_p95=("abs_mass_relative_error", p95),
            peak_bias_abs_mean=("abs_peak_bias_percent", "mean"),
            peak_bias_abs_p95=("abs_peak_bias_percent", p95),
            prediction_std_mean=("concentration_prediction_std_mean", "mean"),
            coverage_2sigma_mean=("concentration_coverage_2sigma", "mean"),
            shrinkage_weight_mean=("mcr_shrinkage_weight", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )

    table.to_csv(output_path, index=False)
    return table


def build_diagnostics_table(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    numeric_columns = [
        "n_constraints",
        "log_variance",
        "observation_noise_relative",
        "concentration_centroid_distance_error",
        "concentration_spread_x_error",
        "concentration_spread_y_error",
        "concentration_prediction_std_mean",
        "concentration_coverage_2sigma",
        "mcr_shrinkage_weight",
    ]
    for column in df.columns:
        if "concentration_plane_" in column and (
            column.endswith("_relative_error") or column.endswith("_scaled_error")
        ):
            numeric_columns.append(column)
        if "concentration_well_" in column and (
            column.endswith("_relative_error") or column.endswith("_scaled_error")
        ):
            numeric_columns.append(column)

    diag = _to_numeric(df.copy(), numeric_columns)
    diag = diag[diag["field"] == "concentration"].copy()
    diag["abs_centroid_distance_error"] = diag["concentration_centroid_distance_error"].abs()
    diag["abs_spread_x_error"] = diag["concentration_spread_x_error"].abs()
    diag["abs_spread_y_error"] = diag["concentration_spread_y_error"].abs()

    plane_scaled_cols = [
        column
        for column in diag.columns
        if "concentration_plane_" in column and column.endswith("_flux_scaled_error")
    ]
    downstream_col = sorted(plane_scaled_cols)[-1] if plane_scaled_cols else None
    if downstream_col is not None:
        diag["abs_downstream_flux_scaled_error"] = diag[downstream_col].abs()
    else:
        diag["abs_downstream_flux_scaled_error"] = np.nan

    peak_flux_cols = [
        column
        for column in diag.columns
        if column.startswith("estimated_trace_cp") and column.endswith("_peak_flux_relative_error")
    ]
    arrival_cols = [
        column
        for column in diag.columns
        if column.startswith("estimated_trace_cp") and column.endswith("_arrival_time_50_error")
    ]
    peak_flux_col = sorted(peak_flux_cols)[-1] if peak_flux_cols else None
    arrival_col = sorted(arrival_cols)[-1] if arrival_cols else None
    diag["abs_downstream_peak_flux_relative_error"] = (
        diag[peak_flux_col].abs() if peak_flux_col is not None else np.nan
    )
    diag["abs_downstream_arrival_time_error"] = (
        diag[arrival_col].abs() if arrival_col is not None else np.nan
    )
    capture_scaled_cols = [
        column
        for column in diag.columns
        if "concentration_well_" in column and column.endswith("_capture_rate_scaled_error")
    ]
    capture_relative_cols = [
        column
        for column in diag.columns
        if "concentration_well_" in column and column.endswith("_capture_rate_relative_error")
    ]
    total_capture_scaled = "concentration_well_capture_rate_total_scaled_error"
    total_capture_relative = "concentration_well_capture_rate_total_relative_error"
    if total_capture_scaled in diag.columns:
        diag["abs_capture_rate_scaled_error"] = diag[total_capture_scaled].abs()
    elif capture_scaled_cols:
        diag["abs_capture_rate_scaled_error"] = diag[capture_scaled_cols].abs().max(axis=1)
    else:
        diag["abs_capture_rate_scaled_error"] = np.nan
    if total_capture_relative in diag.columns:
        diag["abs_capture_rate_relative_error"] = diag[total_capture_relative].abs()
    elif capture_relative_cols:
        diag["abs_capture_rate_relative_error"] = diag[capture_relative_cols].abs().max(axis=1)
    else:
        diag["abs_capture_rate_relative_error"] = np.nan

    group_cols = _benchmark_columns(diag) + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "transport_quantum_mode",
        "constraint_type",
        "constraint_name",
        "mcr_method_name",
        "mcr_method_type",
        "n_constraints",
    ]

    def p95(values: pd.Series) -> float:
        clean = values.dropna().to_numpy(dtype=float)
        if clean.size == 0:
            return np.nan
        return float(np.percentile(clean, 95))

    table = (
        diag.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("abs_centroid_distance_error", "count"),
            centroid_distance_error_mean=("abs_centroid_distance_error", "mean"),
            centroid_distance_error_p95=("abs_centroid_distance_error", p95),
            spread_x_error_mean=("abs_spread_x_error", "mean"),
            spread_y_error_mean=("abs_spread_y_error", "mean"),
            downstream_flux_scaled_error_mean=("abs_downstream_flux_scaled_error", "mean"),
            downstream_flux_scaled_error_p95=("abs_downstream_flux_scaled_error", p95),
            downstream_peak_flux_relative_error_mean=("abs_downstream_peak_flux_relative_error", "mean"),
            downstream_arrival_time_error_mean=("abs_downstream_arrival_time_error", "mean"),
            capture_rate_scaled_error_mean=("abs_capture_rate_scaled_error", "mean"),
            capture_rate_scaled_error_p95=("abs_capture_rate_scaled_error", p95),
            capture_rate_relative_error_mean=("abs_capture_rate_relative_error", "mean"),
            prediction_std_mean=("concentration_prediction_std_mean", "mean"),
            coverage_2sigma_mean=("concentration_coverage_2sigma", "mean"),
            shrinkage_weight_mean=("mcr_shrinkage_weight", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    table.to_csv(output_path, index=False)
    return table


def _transport_anchor_types() -> list[str]:
    return [
        "source_control_plane_hybrid",
        "hybrid_monitoring",
        "pathline_monitoring",
        "control_plane_monitoring",
        "random_active",
        "random_internal",
    ]


def _preferred_transport_subset(df: pd.DataFrame) -> pd.DataFrame:
    preferred = _transport_anchor_types()
    for constraint_type in preferred:
        subset = df[df["constraint_type"] == constraint_type].copy()
        if not subset.empty:
            return subset
    return df.copy()


def build_manuscript_summary(table: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Create a compact concentration table suitable for a manuscript main table."""

    concentration = table[(table["field"] == "concentration") & (table["n_constraints"].isin([2, 16, 64]))].copy()
    summary = _preferred_transport_subset(concentration)
    keep_cols = _benchmark_columns(summary) + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "transport_quantum_mode",
        "constraint_type",
        "constraint_name",
        "mcr_method_name",
        "mcr_method_type",
        "n_constraints",
        "n_runs",
        "rmse_mean",
        "rmse_p95",
        "mass_rel_error_abs_mean",
        "mass_rel_error_abs_p95",
        "peak_bias_abs_mean",
        "peak_bias_abs_p95",
        "prediction_std_mean",
        "coverage_2sigma_mean",
    ]
    summary = summary[keep_cols].sort_values(
        _benchmark_columns(summary)
        + [
            "log_variance",
            "quantum_noise_name",
            "observation_noise_relative",
            "constraint_type",
            "mcr_method_name",
            "n_constraints",
        ]
    )
    summary.to_csv(output_path, index=False)
    return summary


def _preferred_method_scope(table: pd.DataFrame) -> pd.DataFrame:
    concentration = table[table["field"] == "concentration"].copy()
    return _preferred_transport_subset(concentration)


def build_method_comparison_table(table: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    scoped = _preferred_method_scope(table)
    if scoped.empty:
        output_path.write_text("", encoding="utf-8")
        return scoped

    keep = _benchmark_columns(scoped) + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "transport_quantum_mode",
        "constraint_type",
        "constraint_name",
        "n_constraints",
        "mcr_method_name",
        "mcr_method_type",
        "rmse_mean",
        "mass_rel_error_abs_mean",
        "coverage_2sigma_mean",
        "prediction_std_mean",
        "shrinkage_weight_mean",
    ]
    scoped = scoped[keep].copy()
    index_cols = _benchmark_columns(scoped) + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "transport_quantum_mode",
        "constraint_type",
        "constraint_name",
        "n_constraints",
    ]

    wide = scoped.pivot_table(
        index=index_cols,
        columns="mcr_method_name",
        values=[
            "rmse_mean",
            "mass_rel_error_abs_mean",
            "coverage_2sigma_mean",
            "prediction_std_mean",
            "shrinkage_weight_mean",
        ],
        aggfunc="mean",
    )
    wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
    comparison = wide.reset_index()

    if "rmse_mean__global" in comparison.columns:
        for method in scoped["mcr_method_name"].astype(str).unique():
            if method == "global":
                continue
            rmse_col = f"rmse_mean__{method}"
            if rmse_col in comparison.columns:
                comparison[f"rmse_gain_vs_global__{method}"] = (
                    comparison["rmse_mean__global"] - comparison[rmse_col]
                )
                comparison[f"rmse_relative_gain_vs_global__{method}"] = np.where(
                    np.abs(comparison["rmse_mean__global"]) < 1e-30,
                    np.nan,
                    comparison[f"rmse_gain_vs_global__{method}"] / comparison["rmse_mean__global"],
                )

    comparison = comparison.sort_values(index_cols)
    comparison.to_csv(output_path, index=False)
    return comparison


def build_method_phase_table(table: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    scoped = _preferred_method_scope(table)
    if scoped.empty:
        output_path.write_text("", encoding="utf-8")
        return scoped

    phase_index = _benchmark_columns(scoped) + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "transport_quantum_mode",
        "constraint_type",
        "constraint_name",
        "n_constraints",
    ]
    ordered = scoped.sort_values(phase_index + ["rmse_mean", "mass_rel_error_abs_mean"]).copy()
    winners = ordered.drop_duplicates(phase_index, keep="first").copy()
    ordered["rank"] = ordered.groupby(phase_index, dropna=False)["rmse_mean"].rank(method="first")
    runners = (
        ordered[ordered["rank"] == 2]
        .loc[:, phase_index + ["mcr_method_name", "rmse_mean"]]
        .rename(
            columns={
                "mcr_method_name": "runner_up_method",
                "rmse_mean": "runner_up_rmse_mean",
            }
        )
    )
    phase = winners.merge(runners, on=phase_index, how="left")
    phase = phase.rename(
        columns={
            "mcr_method_name": "winner_method_name",
            "mcr_method_type": "winner_method_type",
            "rmse_mean": "winner_rmse_mean",
            "mass_rel_error_abs_mean": "winner_mass_rel_error_abs_mean",
            "coverage_2sigma_mean": "winner_coverage_2sigma_mean",
            "prediction_std_mean": "winner_prediction_std_mean",
            "shrinkage_weight_mean": "winner_shrinkage_weight_mean",
        }
    )
    phase["winner_margin_rmse"] = phase["runner_up_rmse_mean"] - phase["winner_rmse_mean"]
    phase["winner_margin_relative"] = np.where(
        phase["runner_up_rmse_mean"].abs() < 1e-30,
        np.nan,
        phase["winner_margin_rmse"] / phase["runner_up_rmse_mean"],
    )
    phase = phase.sort_values(phase_index)
    phase.to_csv(output_path, index=False)
    return phase


def build_method_winner_summary(phase_table: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if phase_table.empty:
        output_path.write_text("", encoding="utf-8")
        return phase_table

    summary = (
        phase_table.groupby(
            _benchmark_columns(phase_table)
            + ["quantum_noise_name", "observation_noise_relative", "winner_method_name"],
            dropna=False,
        )
        .agg(
            n_cells=("winner_method_name", "count"),
            mean_margin_rmse=("winner_margin_rmse", "mean"),
            mean_margin_relative=("winner_margin_relative", "mean"),
        )
        .reset_index()
        .sort_values(
            _benchmark_columns(phase_table)
            + ["quantum_noise_name", "observation_noise_relative", "winner_method_name"]
        )
    )
    summary.to_csv(output_path, index=False)
    return summary


def build_benchmark_winner_summary(phase_table: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    benchmark_cols = _benchmark_columns(phase_table)
    if phase_table.empty or "benchmark_name" not in benchmark_cols:
        output_path.write_text("", encoding="utf-8")
        return phase_table.iloc[0:0].copy()

    group_cols = benchmark_cols + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "winner_method_name",
    ]
    summary = (
        phase_table.groupby(group_cols, dropna=False)
        .agg(
            n_cells=("winner_method_name", "count"),
            mean_margin_rmse=("winner_margin_rmse", "mean"),
            mean_margin_relative=("winner_margin_relative", "mean"),
            mean_winner_rmse=("winner_rmse_mean", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    summary.to_csv(output_path, index=False)
    return summary


def build_benchmark_normalized_table(
    table: pd.DataFrame,
    diagnostics_table: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    merged = _merge_concentration_and_diagnostics(table, diagnostics_table)
    if merged.empty:
        output_path.write_text("", encoding="utf-8")
        return merged

    numeric_columns = [
        "rmse_mean",
        "mass_rel_error_abs_mean",
        "peak_bias_abs_mean",
        "prediction_std_mean",
        "coverage_2sigma_mean",
        "downstream_flux_scaled_error_mean",
        "downstream_peak_flux_relative_error_mean",
        "downstream_arrival_time_error_mean",
        "capture_rate_scaled_error_mean",
        "capture_rate_relative_error_mean",
    ]
    merged = _to_numeric(merged.copy(), numeric_columns + ["n_constraints", "log_variance", "observation_noise_relative"])
    group_cols = _benchmark_columns(merged) + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "transport_quantum_mode",
        "constraint_type",
        "constraint_name",
        "n_constraints",
    ]
    metric_cols = [
        "rmse_mean",
        "mass_rel_error_abs_mean",
        "peak_bias_abs_mean",
        "downstream_flux_scaled_error_mean",
        "capture_rate_scaled_error_mean",
    ]
    regret_log_cols: list[str] = []
    available_regret_cols: list[str] = []

    for metric_col in metric_cols:
        if metric_col not in merged.columns:
            continue
        best_col = f"best__{metric_col}"
        regret_col = f"{metric_col}_relative_regret"
        log_col = f"{metric_col}_relative_regret_log10p1"
        merged[best_col] = merged.groupby(group_cols, dropna=False)[metric_col].transform("min")
        merged[regret_col] = _relative_regret(merged[metric_col], merged[best_col])
        merged[log_col] = np.log10(1.0 + np.maximum(merged[regret_col], 0.0))
        available_regret_cols.append(regret_col)
        regret_log_cols.append(log_col)

    if "rmse_mean_relative_regret" in merged.columns:
        merged["near_best_rmse_05"] = merged["rmse_mean_relative_regret"] <= 0.05
        merged["near_best_rmse_10"] = merged["rmse_mean_relative_regret"] <= 0.10
    if "capture_rate_scaled_error_mean_relative_regret" in merged.columns:
        merged["near_best_capture_10"] = merged["capture_rate_scaled_error_mean_relative_regret"] <= 0.10
        merged["near_best_capture_25"] = merged["capture_rate_scaled_error_mean_relative_regret"] <= 0.25
    if regret_log_cols:
        merged["overall_regret_score"] = merged[regret_log_cols].mean(axis=1, skipna=True)
    else:
        merged["overall_regret_score"] = np.nan

    sort_cols = group_cols + ["mcr_method_name"]
    merged = merged.sort_values(sort_cols)
    merged.to_csv(output_path, index=False)
    return merged


def build_benchmark_normalized_summary(
    normalized_table: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if normalized_table.empty:
        output_path.write_text("", encoding="utf-8")
        return normalized_table

    summary = normalized_table.copy()
    benchmark_cols = _benchmark_columns(summary)
    group_cols = benchmark_cols + ["mcr_method_name", "mcr_method_type"]
    aggregations: dict[str, tuple[str, str]] = {}

    for metric_col in [
        "rmse_mean_relative_regret",
        "mass_rel_error_abs_mean_relative_regret",
        "peak_bias_abs_mean_relative_regret",
        "downstream_flux_scaled_error_mean_relative_regret",
        "capture_rate_scaled_error_mean_relative_regret",
        "overall_regret_score",
    ]:
        if metric_col in summary.columns:
            aggregations[f"{metric_col}_median"] = (metric_col, "median")
            aggregations[f"{metric_col}_mean"] = (metric_col, "mean")

    for bool_col in [
        "near_best_rmse_05",
        "near_best_rmse_10",
        "near_best_capture_10",
        "near_best_capture_25",
    ]:
        if bool_col in summary.columns:
            aggregations[f"{bool_col}_share"] = (bool_col, "mean")

    aggregations["n_regimes"] = ("mcr_method_name", "count")
    out = (
        summary.groupby(group_cols, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    sort_metric = (
        "overall_regret_score_median"
        if "overall_regret_score_median" in out.columns
        else ("rmse_mean_relative_regret_median" if "rmse_mean_relative_regret_median" in out.columns else "n_regimes")
    )
    out = out.sort_values(benchmark_cols + [sort_metric, "mcr_method_name"])
    out.to_csv(output_path, index=False)
    return out


def build_capture_manuscript_table(
    normalized_table: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if normalized_table.empty or "capture_rate_scaled_error_mean" not in normalized_table.columns:
        output_path.write_text("", encoding="utf-8")
        return normalized_table.iloc[0:0].copy()

    capture = normalized_table[normalized_table["capture_rate_scaled_error_mean"].notna()].copy()
    capture = capture[capture["n_constraints"].isin([4, 16, 64])].copy()
    if capture.empty:
        output_path.write_text("", encoding="utf-8")
        return capture

    keep_cols = _benchmark_columns(capture) + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "constraint_name",
        "mcr_method_name",
        "mcr_method_type",
        "n_constraints",
        "rmse_mean",
        "rmse_mean_relative_regret",
        "downstream_flux_scaled_error_mean",
        "downstream_flux_scaled_error_mean_relative_regret",
        "capture_rate_scaled_error_mean",
        "capture_rate_scaled_error_mean_relative_regret",
        "overall_regret_score",
    ]
    keep_cols = [column for column in keep_cols if column in capture.columns]
    out = capture.loc[:, keep_cols].sort_values(
        _benchmark_columns(capture)
        + [
            "log_variance",
            "quantum_noise_name",
            "observation_noise_relative",
            "n_constraints",
            "mcr_method_name",
        ]
    )
    out.to_csv(output_path, index=False)
    return out


def _fmt_sci(value: float) -> str:
    if pd.isna(value):
        return "nan"
    if value == 0:
        return "0"
    return f"{float(value):.2e}"


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.2f}%"


def build_results_brief(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_path: Path,
    normalized_summary: pd.DataFrame | None = None,
) -> str:
    """Create a concise manuscript-ready markdown summary from the sweep table."""

    df = summary.copy()
    numeric_cols = [
        "log_variance",
        "observation_noise_relative",
        "n_constraints",
        "rmse_mean",
        "rmse_p95",
        "mass_rel_error_abs_mean",
        "peak_bias_abs_mean",
    ]
    df = _to_numeric(df, numeric_cols)
    benchmark_cols = _benchmark_columns(df)
    single_log_variance = df["log_variance"].dropna().nunique() <= 1
    best_setting_cols = benchmark_cols + [
        "log_variance",
        "quantum_noise_name",
        "observation_noise_relative",
        "constraint_type",
        "n_constraints",
    ]
    df_best = df.sort_values("rmse_mean").drop_duplicates(best_setting_cols)

    lines: list[str] = ["# Sweep Results Brief", ""]

    if "benchmark_name" in benchmark_cols:
        benchmark_group_cols = benchmark_cols + ["log_variance"]
        benchmark_mean = (
            df_best.groupby(benchmark_group_cols, dropna=False)["rmse_mean"]
            .mean()
            .reset_index()
            .sort_values("rmse_mean")
        )
        if not benchmark_mean.empty:
            hardest = benchmark_mean.iloc[-1]
            easiest = benchmark_mean.iloc[0]
            hardest_label = hardest.get("benchmark_label", hardest.get("benchmark_name", "benchmark"))
            easiest_label = easiest.get("benchmark_label", easiest.get("benchmark_name", "benchmark"))
            lines.extend(
                [
                    "## Benchmark spread",
                    "",
                    f"Across the benchmark suite, the easiest tested benchmark regime is "
                    f"`{easiest_label}` at sigma^2_lnK={easiest['log_variance']:g} "
                    f"(mean RMSE={_fmt_sci(easiest['rmse_mean'])}), while the hardest is "
                    f"`{hardest_label}` at sigma^2_lnK={hardest['log_variance']:g} "
                    f"(mean RMSE={_fmt_sci(hardest['rmse_mean'])}).",
                    "",
                ]
            )

    if normalized_summary is not None and not normalized_summary.empty:
        norm = normalized_summary.copy()
        benchmark_group_cols = _benchmark_columns(norm)
        if "overall_regret_score_median" in norm.columns and benchmark_group_cols:
            leaders = (
                norm.sort_values("overall_regret_score_median")
                .drop_duplicates(benchmark_group_cols)
                .loc[:, benchmark_group_cols + ["mcr_method_name", "overall_regret_score_median"]]
            )
            lines.extend(["## Normalized robustness", ""])
            for _, row in leaders.iterrows():
                bench = row.get("benchmark_label", row.get("benchmark_name", "benchmark"))
                score = float(row["overall_regret_score_median"])
                lines.append(
                    f"In benchmark-normalized space, `{row['mcr_method_name']}` is the most robust method for "
                    f"`{bench}` with median regret score {_fmt_sci(score)}."
                )
            lines.append("")

    exact_base = df_best[
        (df_best["quantum_noise_name"] == "exact")
        & (df_best["observation_noise_relative"] == 0.0)
        & (df_best["n_constraints"] == 2)
    ]
    if not exact_base.empty:
        row = exact_base.sort_values("log_variance").iloc[-1]
        heterogeneity_phrase = (
            f"At the evaluated sigma^2_lnK={row['log_variance']:g}, "
            if single_log_variance
            else f"At the strongest tested heterogeneity (sigma^2_lnK={row['log_variance']:g}), "
        )
        lines.extend(
            [
                "## Core finding",
                "",
                "The exact operator-level transport surrogate combined with the preferred anchor design preserves "
                "concentration fields to machine precision in the noise-free limit. "
                f"{heterogeneity_phrase}"
                f"the mean concentration RMSE at 2 constraints is {_fmt_sci(row['rmse_mean'])} "
                f"for `{row['constraint_type']}` with `{row['mcr_method_name']}`.",
                "",
            ]
        )

    exact_obs = df_best[
        (df_best["quantum_noise_name"] == "exact")
        & (df_best["observation_noise_relative"] > 0.0)
    ]
    if not exact_obs.empty:
        best = exact_obs.sort_values("rmse_mean").iloc[0]
        worst = exact_obs.sort_values("rmse_mean").iloc[-1]
        lines.extend(
            [
                "## Observation-noise sensitivity",
                "",
                "Under observation noise alone, increasing the number of truth-independent monitoring constraints generally improves "
                "reconstruction accuracy. "
                f"The best exact-case noisy reconstruction in this sweep occurs at "
                f"sigma^2_lnK={best['log_variance']:g}, obs noise={best['observation_noise_relative']:.2f}, "
                f"{int(best['n_constraints'])} constraints, `{best['constraint_type']}`, and "
                f"`{best['mcr_method_name']}`, with RMSE={_fmt_sci(best['rmse_mean'])}. "
                f"The worst exact noisy case occurs at sigma^2_lnK={worst['log_variance']:g}, "
                f"obs noise={worst['observation_noise_relative']:.2f}, {int(worst['n_constraints'])} constraints, "
                f"`{worst['constraint_type']}`, and `{worst['mcr_method_name']}`, with RMSE={_fmt_sci(worst['rmse_mean'])}.",
                "",
            ]
        )

    noisy = df_best[df_best["quantum_noise_name"].isin(["low_noise", "high_noise"])]
    if not noisy.empty:
        grouped = (
            noisy.groupby(["quantum_noise_name", "n_constraints"], dropna=False)["rmse_mean"]
            .mean()
            .reset_index()
            .sort_values(["quantum_noise_name", "n_constraints"])
        )
        lines.extend(["## Quantum-noise sensitivity", ""])
        for noise_name in ["low_noise", "high_noise"]:
            subset = grouped[grouped["quantum_noise_name"] == noise_name]
            if subset.empty:
                continue
            first = subset.iloc[0]
            last = subset.iloc[-1]
            lines.append(
                f"For `{noise_name}`, the mean RMSE changes from {_fmt_sci(first['rmse_mean'])} "
                f"at {int(first['n_constraints'])} constraints to {_fmt_sci(last['rmse_mean'])} "
                f"at {int(last['n_constraints'])} constraints."
            )
        lines.append("")

    adaptive_rows = df[df["mcr_method_name"].astype(str).str.contains("adaptive", case=False, na=False)]
    if not adaptive_rows.empty:
        pivot_cols = benchmark_cols + [
            "log_variance",
            "quantum_noise_name",
            "observation_noise_relative",
            "constraint_type",
            "n_constraints",
        ]
        adaptive_name = str(adaptive_rows["mcr_method_name"].iloc[0])
        adaptive_cmp = df.pivot_table(
            index=pivot_cols,
            columns="mcr_method_name",
            values="rmse_mean",
            aggfunc="mean",
        ).reset_index()
        adaptive_global_col = "global"
        if adaptive_global_col in adaptive_cmp.columns and adaptive_name in adaptive_cmp.columns:
            adaptive_cmp["gain"] = adaptive_cmp[adaptive_global_col] - adaptive_cmp[adaptive_name]
            moderate = adaptive_cmp[adaptive_cmp["n_constraints"] >= 16].copy()
            positive = moderate[moderate["gain"] > 0.0].copy()
            if not positive.empty:
                best = positive.sort_values("gain", ascending=False).iloc[0]
                median_gain = float(positive["gain"].median())
                lines.extend(
                    [
                        "## Adaptive shrinkage",
                        "",
                        "The adaptive empirical-Bayes regional correction behaves as a guarded extension of global MCR: "
                        "it keeps the global solution as the default backbone and only releases regional freedom when the anchor budget supports it. "
                        f"Across the positive-gain regime with at least 16 constraints, the median adaptive gain over global MCR is {_fmt_sci(median_gain)}. "
                        f"The strongest gain appears at sigma^2_lnK={best['log_variance']:g}, "
                        f"noise=`{best['quantum_noise_name']}`, obs noise={best['observation_noise_relative']:.2f}, "
                        f"`{best['constraint_type']}`, and {int(best['n_constraints'])} constraints, "
                        f"where `{adaptive_name}` reduces RMSE by {_fmt_sci(best['gain'])}.",
                        "",
                    ]
                )

    if {"global", "regional_bayesian"}.issubset(set(df["mcr_method_type"].astype(str))):
        pivot_cols = benchmark_cols + [
            "log_variance",
            "quantum_noise_name",
            "observation_noise_relative",
            "constraint_type",
            "n_constraints",
        ]
        paired = df.pivot_table(
            index=pivot_cols,
            columns="mcr_method_type",
            values="rmse_mean",
            aggfunc="mean",
        ).reset_index()
        if "global" in paired.columns and "regional_bayesian" in paired.columns:
            paired["regional_gain"] = paired["global"] - paired["regional_bayesian"]
            moderate = paired[paired["n_constraints"] >= 16].copy()
            positive = moderate[moderate["regional_gain"] > 0.0].copy()
            sparse = paired[paired["n_constraints"] == paired["n_constraints"].min()].copy()
            if not positive.empty:
                best_gain = positive.sort_values("regional_gain", ascending=False).iloc[0]
                sparse_penalty = float(sparse["regional_gain"].mean()) if not sparse.empty else np.nan
                sparse_sentence = (
                    f"At the sparsest tested budget, the mean regional-over-global gain is {_fmt_sci(sparse_penalty)}."
                    if not np.isnan(sparse_penalty) and sparse_penalty >= 0.0
                    else f"At the sparsest tested budget, the mean regional-over-global penalty is {_fmt_sci(-sparse_penalty)}."
                )
                lines.extend(
                    [
                        "## Regional extension",
                        "",
                        "The regional Bayesian MCR extension becomes useful once the anchor budget is high enough "
                        "to support spatially varying scale factors; at sparse budgets the global scalar remains more robust. "
                        f"The largest gain appears at sigma^2_lnK={best_gain['log_variance']:g}, "
                        f"noise=`{best_gain['quantum_noise_name']}`, obs noise={best_gain['observation_noise_relative']:.2f}, "
                        f"`{best_gain['constraint_type']}`, and {int(best_gain['n_constraints'])} constraints, "
                        f"where regional MCR reduces RMSE by {_fmt_sci(best_gain['regional_gain'])} relative to global MCR. "
                        ,
                        sparse_sentence,
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "## Regional extension",
                        "",
                        "In this sweep, the regional Bayesian MCR behaves primarily as an uncertainty-aware extension "
                        "rather than a universal RMSE win; the global scalar remains the most robust choice under very sparse monitoring.",
                        "",
                    ]
                )

    extreme = df_best[df_best["log_variance"] == df_best["log_variance"].max()]
    if not extreme.empty:
        stable = extreme.sort_values("rmse_mean").iloc[0]
        conservative = extreme.sort_values("mass_rel_error_abs_mean").iloc[0]
        regime_title = "Representative regime" if single_log_variance else "High-heterogeneity regime"
        regime_lead = (
            f"At the evaluated sigma^2_lnK={stable['log_variance']:g}, "
            if single_log_variance
            else f"At sigma^2_lnK={stable['log_variance']:g}, "
        )
        lines.extend(
            [
                f"## {regime_title}",
                "",
                f"{regime_lead}the most accurate tested setting in the compact "
                f"summary table is `{stable['quantum_noise_name']}` with obs noise="
                f"{stable['observation_noise_relative']:.2f}, `{stable['constraint_type']}`, "
                f"`{stable['mcr_method_name']}`, and {int(stable['n_constraints'])} constraints "
                f"(RMSE={_fmt_sci(stable['rmse_mean'])}). "
                f"The smallest mean mass-balance distortion in the same regime is "
                f"{_fmt_pct(100.0 * conservative['mass_rel_error_abs_mean'])}, obtained with "
                f"`{conservative['quantum_noise_name']}` / `{conservative['mcr_method_name']}` at "
                f"{int(conservative['n_constraints'])} constraints.",
                "",
            ]
        )

    high_noise_worst = df_best[df_best["quantum_noise_name"] == "high_noise"]
    if not high_noise_worst.empty:
        row = high_noise_worst.sort_values("rmse_mean", ascending=False).iloc[0]
        lines.extend(
            [
                "## Practical takeaway",
                "",
                "The fragile corner of the design space is the combination of strong quantum-state noise and very "
                "sparse constraints. "
                f"In this sweep, the worst `high_noise` case appears at sigma^2_lnK={row['log_variance']:g}, "
                f"obs noise={row['observation_noise_relative']:.2f}, `{row['constraint_type']}`, "
                f"`{row['mcr_method_name']}`, and {int(row['n_constraints'])} constraints, "
                f"with RMSE={_fmt_sci(row['rmse_mean'])} and mean absolute peak bias="
                f"{_fmt_pct(row['peak_bias_abs_mean'])}.",
                "",
            ]
        )

    diag = diagnostics.copy()
    diag_numeric_cols = [
        "log_variance",
        "observation_noise_relative",
        "n_constraints",
        "centroid_distance_error_mean",
        "downstream_flux_scaled_error_mean",
        "capture_rate_scaled_error_mean",
    ]
    diag = _to_numeric(diag, diag_numeric_cols)
    diag = _preferred_transport_subset(diag)
    if not diag.empty:
        best_centroid = diag.sort_values("centroid_distance_error_mean").iloc[0]
        worst_flux = diag.sort_values("downstream_flux_scaled_error_mean", ascending=False).iloc[0]
        lines.extend(
            [
                "## Transport diagnostics",
                "",
                f"The smallest plume-centroid displacement error in this sweep is "
                f"{_fmt_sci(best_centroid['centroid_distance_error_mean'])}, achieved for "
                f"`{best_centroid['quantum_noise_name']}` / `{best_centroid['mcr_method_name']}` at sigma^2_lnK={best_centroid['log_variance']:g}, "
                f"obs noise={best_centroid['observation_noise_relative']:.2f}, and "
                f"{int(best_centroid['n_constraints'])} constraints. "
                f"The largest downstream control-plane flux distortion, scaled by the reference peak control-plane flux, is "
                f"{_fmt_pct(100.0 * worst_flux['downstream_flux_scaled_error_mean'])}, observed for "
                f"`{worst_flux['quantum_noise_name']}` / `{worst_flux['mcr_method_name']}` at sigma^2_lnK={worst_flux['log_variance']:g}, "
                f"obs noise={worst_flux['observation_noise_relative']:.2f}, and "
                f"{int(worst_flux['n_constraints'])} constraints.",
                "",
            ]
        )
        if "capture_rate_scaled_error_mean" in diag.columns and diag["capture_rate_scaled_error_mean"].notna().any():
            best_capture = diag.sort_values("capture_rate_scaled_error_mean").iloc[0]
            worst_capture = diag.sort_values("capture_rate_scaled_error_mean", ascending=False).iloc[0]
            lines.extend(
                [
                    "## Capture diagnostics",
                    "",
                    f"The smallest extraction-capture error in this sweep is "
                    f"{_fmt_pct(100.0 * best_capture['capture_rate_scaled_error_mean'])}, achieved for "
                    f"`{best_capture['quantum_noise_name']}` / `{best_capture['mcr_method_name']}` at "
                    f"sigma^2_lnK={best_capture['log_variance']:g}, obs noise={best_capture['observation_noise_relative']:.2f}, "
                    f"and {int(best_capture['n_constraints'])} constraints. "
                    f"The largest extraction-capture distortion is "
                    f"{_fmt_pct(100.0 * worst_capture['capture_rate_scaled_error_mean'])}, observed for "
                    f"`{worst_capture['quantum_noise_name']}` / `{worst_capture['mcr_method_name']}` at "
                    f"sigma^2_lnK={worst_capture['log_variance']:g}, obs noise={worst_capture['observation_noise_relative']:.2f}, "
                    f"and {int(worst_capture['n_constraints'])} constraints.",
                    "",
                ]
            )

    text = "\n".join(lines).strip() + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


def _plot_rmse_grid(
    table: pd.DataFrame,
    *,
    field: str,
    metric_column: str,
    ylabel: str,
    output_base: Path,
) -> None:
    data = table[table["field"] == field].copy()
    data = _preferred_transport_subset(data)
    if data.empty:
        return

    data["n_constraints"] = pd.to_numeric(data["n_constraints"], errors="coerce")
    data["log_variance"] = pd.to_numeric(data["log_variance"], errors="coerce")
    data["observation_noise_relative"] = pd.to_numeric(
        data["observation_noise_relative"],
        errors="coerce",
    )

    variances = sorted(data["log_variance"].dropna().unique())
    obs_levels = sorted(data["observation_noise_relative"].dropna().unique())
    data["curve_label"] = data["quantum_noise_name"].astype(str) + "/" + data["mcr_method_name"].astype(str)
    noise_names = list(dict.fromkeys(data["curve_label"].astype(str)))

    fig, axes = plt.subplots(
        len(obs_levels),
        len(variances),
        figsize=(3.2 * len(variances), 2.7 * len(obs_levels)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, obs in enumerate(obs_levels):
        for col, variance in enumerate(variances):
            ax = axes[row][col]
            subset = data[
                (data["log_variance"] == variance)
                & (data["observation_noise_relative"] == obs)
            ]
            for noise_name in noise_names:
                curve = subset[subset["curve_label"].astype(str) == noise_name]
                curve = curve.sort_values("n_constraints")
                if curve.empty:
                    continue
                y_values = np.maximum(curve[metric_column].to_numpy(dtype=float), 1e-18)
                ax.plot(
                    curve["n_constraints"],
                    y_values,
                    marker="o",
                    linewidth=1.7,
                    markersize=4,
                    label=noise_name,
                )
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
            if row == 0:
                ax.set_title(rf"$\sigma^2_{{\ln K}}={variance:g}$", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"obs noise={obs:g}\n{ylabel}", fontsize=9)
            if row == len(obs_levels) - 1:
                ax.set_xlabel("MCR constraints", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_mass_peak(table: pd.DataFrame, output_base: Path) -> None:
    data = table[table["field"] == "concentration"].copy()
    data = _preferred_transport_subset(data)
    if data.empty:
        return

    numeric_cols = [
        "n_constraints",
        "log_variance",
        "observation_noise_relative",
        "mass_rel_error_abs_mean",
        "peak_bias_abs_mean",
    ]
    data = _to_numeric(data, numeric_cols)

    max_variance = float(data["log_variance"].max())
    data = data[data["log_variance"] == max_variance]
    obs_levels = sorted(data["observation_noise_relative"].dropna().unique())
    data["curve_label"] = data["quantum_noise_name"].astype(str) + "/" + data["mcr_method_name"].astype(str)
    noise_names = list(dict.fromkeys(data["curve_label"].astype(str)))

    fig, axes = plt.subplots(2, len(obs_levels), figsize=(3.3 * len(obs_levels), 5.2), sharex=True)
    if len(obs_levels) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    metric_specs = [
        ("mass_rel_error_abs_mean", "abs. mass rel. error"),
        ("peak_bias_abs_mean", "abs. peak bias (%)"),
    ]

    for col, obs in enumerate(obs_levels):
        subset_obs = data[data["observation_noise_relative"] == obs]
        for row, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row][col]
            for noise_name in noise_names:
                curve = subset_obs[subset_obs["curve_label"].astype(str) == noise_name]
                curve = curve.sort_values("n_constraints")
                if curve.empty:
                    continue
                y_values = np.maximum(curve[metric].to_numpy(dtype=float), 1e-18)
                ax.plot(curve["n_constraints"], y_values, marker="o", linewidth=1.7, label=noise_name)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
            if row == 0:
                ax.set_title(f"obs noise={obs:g}", fontsize=10)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 1:
                ax.set_xlabel("MCR constraints", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle(rf"Mass and peak robustness at $\sigma^2_{{\ln K}}={max_variance:g}$", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_diagnostics_grid(table: pd.DataFrame, output_base: Path) -> None:
    data = table.copy()
    data = _preferred_transport_subset(data)
    if data.empty:
        return

    data["n_constraints"] = pd.to_numeric(data["n_constraints"], errors="coerce")
    data["log_variance"] = pd.to_numeric(data["log_variance"], errors="coerce")
    data["observation_noise_relative"] = pd.to_numeric(data["observation_noise_relative"], errors="coerce")

    variances = sorted(data["log_variance"].dropna().unique())
    obs_levels = sorted(data["observation_noise_relative"].dropna().unique())
    data["curve_label"] = data["quantum_noise_name"].astype(str) + "/" + data["mcr_method_name"].astype(str)
    noise_names = list(dict.fromkeys(data["curve_label"].astype(str)))
    metric_specs = [
        ("centroid_distance_error_mean", "centroid distance error"),
        ("downstream_flux_scaled_error_mean", "downstream flux scaled error"),
    ]

    fig, axes = plt.subplots(
        len(metric_specs) * len(obs_levels),
        len(variances),
        figsize=(3.2 * len(variances), 2.4 * len(metric_specs) * len(obs_levels)),
        sharex=True,
        squeeze=False,
    )

    for obs_idx, obs in enumerate(obs_levels):
        for metric_idx, (metric_col, ylabel) in enumerate(metric_specs):
            row = obs_idx * len(metric_specs) + metric_idx
            for col, variance in enumerate(variances):
                ax = axes[row][col]
                subset = data[
                    (data["log_variance"] == variance)
                    & (data["observation_noise_relative"] == obs)
                ]
                for noise_name in noise_names:
                    curve = subset[subset["curve_label"].astype(str) == noise_name]
                    curve = curve.sort_values("n_constraints")
                    if curve.empty:
                        continue
                    y_values = np.maximum(curve[metric_col].to_numpy(dtype=float), 1e-18)
                    ax.plot(curve["n_constraints"], y_values, marker="o", linewidth=1.7, label=noise_name)
                ax.set_yscale("log")
                ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
                if row == 0:
                    ax.set_title(rf"$\sigma^2_{{\ln K}}={variance:g}$", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"obs={obs:g}\n{ylabel}", fontsize=9)
                if row == len(metric_specs) * len(obs_levels) - 1:
                    ax.set_xlabel("MCR constraints", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _method_abbreviation(method_name: str) -> str:
    text = str(method_name).lower()
    if "adaptive" in text:
        return "A"
    if "regional" in text:
        return "R"
    if "global" in text:
        return "G"
    return "?"


def _plot_method_phase_diagram(phase_table: pd.DataFrame, output_base: Path) -> None:
    if phase_table.empty:
        return

    data = phase_table.copy()
    data["log_variance"] = pd.to_numeric(data["log_variance"], errors="coerce")
    data["observation_noise_relative"] = pd.to_numeric(data["observation_noise_relative"], errors="coerce")
    data["n_constraints"] = pd.to_numeric(data["n_constraints"], errors="coerce")

    obs_levels = sorted(data["observation_noise_relative"].dropna().unique())
    quantum_levels = list(dict.fromkeys(data["quantum_noise_name"].astype(str)))
    variances = sorted(data["log_variance"].dropna().unique())
    constraints = sorted(data["n_constraints"].dropna().unique())
    method_names = list(dict.fromkeys(data["winner_method_name"].astype(str)))
    method_codes = {name: idx for idx, name in enumerate(method_names)}
    cmap = plt.cm.get_cmap("Set2", max(1, len(method_names)))

    fig, axes = plt.subplots(
        len(obs_levels),
        len(quantum_levels),
        figsize=(3.2 * len(quantum_levels), 2.8 * len(obs_levels)),
        squeeze=False,
    )

    for row, obs in enumerate(obs_levels):
        for col, quantum_name in enumerate(quantum_levels):
            ax = axes[row][col]
            subset = data[
                (data["observation_noise_relative"] == obs)
                & (data["quantum_noise_name"].astype(str) == quantum_name)
            ]
            grid = np.full((len(variances), len(constraints)), fill_value=np.nan)
            labels = np.full((len(variances), len(constraints)), fill_value="", dtype=object)
            for i, variance in enumerate(variances):
                for j, n_constraints in enumerate(constraints):
                    cell = subset[
                        (subset["log_variance"] == variance)
                        & (subset["n_constraints"] == n_constraints)
                    ]
                    if cell.empty:
                        continue
                    winner = str(cell.iloc[0]["winner_method_name"])
                    grid[i, j] = method_codes[winner]
                    labels[i, j] = _method_abbreviation(winner)

            masked = np.ma.masked_invalid(grid)
            ax.imshow(
                masked,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=-0.5,
                vmax=max(-0.5, len(method_names) - 0.5),
                origin="lower",
            )
            for i in range(len(variances)):
                for j in range(len(constraints)):
                    if labels[i, j]:
                        ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=8, color="black")
            ax.set_xticks(range(len(constraints)))
            ax.set_xticklabels([str(int(v)) for v in constraints], fontsize=8)
            ax.set_yticks(range(len(variances)))
            ax.set_yticklabels([f"{v:g}" for v in variances], fontsize=8)
            if row == len(obs_levels) - 1:
                ax.set_xlabel("constraints", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"obs={obs:g}\n" + r"$\sigma^2_{\ln K}$", fontsize=9)
            ax.set_title(str(quantum_name), fontsize=10)

    legend_handles = [
        Patch(facecolor=cmap(code), edgecolor="none", label=f"{_method_abbreviation(name)} = {name}")
        for name, code in method_codes.items()
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", ncol=max(1, len(legend_handles)), frameon=False)
    fig.suptitle("Method-regime phase diagram", y=0.99)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.98, hspace=0.35, wspace=0.22)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_adaptive_gain_heatmap(comparison_table: pd.DataFrame, output_base: Path) -> None:
    if comparison_table.empty:
        return

    adaptive_cols = [
        column
        for column in comparison_table.columns
        if column.startswith("rmse_relative_gain_vs_global__") and "adaptive" in column.lower()
    ]
    if not adaptive_cols:
        return
    gain_col = adaptive_cols[0]
    method_name = gain_col.split("__", 1)[1]

    data = comparison_table.copy()
    data["log_variance"] = pd.to_numeric(data["log_variance"], errors="coerce")
    data["observation_noise_relative"] = pd.to_numeric(data["observation_noise_relative"], errors="coerce")
    data["n_constraints"] = pd.to_numeric(data["n_constraints"], errors="coerce")
    data[gain_col] = pd.to_numeric(data[gain_col], errors="coerce")

    obs_levels = sorted(data["observation_noise_relative"].dropna().unique())
    quantum_levels = list(dict.fromkeys(data["quantum_noise_name"].astype(str)))
    variances = sorted(data["log_variance"].dropna().unique())
    constraints = sorted(data["n_constraints"].dropna().unique())
    finite = data[gain_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(
        len(obs_levels),
        len(quantum_levels),
        figsize=(3.2 * len(quantum_levels), 2.8 * len(obs_levels)),
        squeeze=False,
    )
    image = None

    for row, obs in enumerate(obs_levels):
        for col, quantum_name in enumerate(quantum_levels):
            ax = axes[row][col]
            subset = data[
                (data["observation_noise_relative"] == obs)
                & (data["quantum_noise_name"].astype(str) == quantum_name)
            ]
            grid = np.full((len(variances), len(constraints)), fill_value=np.nan)
            for i, variance in enumerate(variances):
                for j, n_constraints in enumerate(constraints):
                    cell = subset[
                        (subset["log_variance"] == variance)
                        & (subset["n_constraints"] == n_constraints)
                    ]
                    if cell.empty:
                        continue
                    grid[i, j] = float(cell.iloc[0][gain_col])

            image = ax.imshow(
                np.ma.masked_invalid(grid),
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                origin="lower",
            )
            ax.set_xticks(range(len(constraints)))
            ax.set_xticklabels([str(int(v)) for v in constraints], fontsize=8)
            ax.set_yticks(range(len(variances)))
            ax.set_yticklabels([f"{v:g}" for v in variances], fontsize=8)
            if row == len(obs_levels) - 1:
                ax.set_xlabel("constraints", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"obs={obs:g}\n" + r"$\sigma^2_{\ln K}$", fontsize=9)
            ax.set_title(str(quantum_name), fontsize=10)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, pad=0.015)
        colorbar.set_label(f"relative RMSE gain of {method_name} vs global", fontsize=9)
    fig.suptitle("Adaptive-shrinkage gain phase diagram", y=0.99)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.92, hspace=0.35, wspace=0.22)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _benchmark_regime_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "benchmark_name" not in frame.columns:
        return frame.iloc[0:0].copy()

    data = frame.copy()
    data["log_variance"] = pd.to_numeric(data["log_variance"], errors="coerce")
    data["observation_noise_relative"] = pd.to_numeric(data["observation_noise_relative"], errors="coerce")
    data["n_constraints"] = pd.to_numeric(data["n_constraints"], errors="coerce")
    if "benchmark_label" not in data.columns:
        data["benchmark_label"] = data["benchmark_name"]
    data["benchmark_row_label"] = data.apply(
        lambda row: f"{row['benchmark_label']}\n" + rf"$\sigma^2_{{\ln K}}={row['log_variance']:g}$",
        axis=1,
    )
    return data


def _plot_benchmark_phase_diagram(phase_table: pd.DataFrame, output_base: Path) -> None:
    data = _benchmark_regime_rows(phase_table)
    if data.empty:
        return

    obs_levels = sorted(data["observation_noise_relative"].dropna().unique())
    quantum_levels = list(dict.fromkeys(data["quantum_noise_name"].astype(str)))
    row_labels = (
        data.sort_values(["benchmark_label", "log_variance"])[["benchmark_label", "log_variance", "benchmark_row_label"]]
        .drop_duplicates()["benchmark_row_label"]
        .tolist()
    )
    constraints = sorted(data["n_constraints"].dropna().unique())
    method_names = list(dict.fromkeys(data["winner_method_name"].astype(str)))
    if not obs_levels or not quantum_levels or not row_labels or not constraints or not method_names:
        return

    label_to_idx = {label: idx for idx, label in enumerate(row_labels)}
    constraint_to_idx = {value: idx for idx, value in enumerate(constraints)}
    method_to_idx = {name: idx for idx, name in enumerate(method_names)}
    cmap = plt.cm.get_cmap("Set2", max(1, len(method_names)))

    fig_height = max(3.2 * len(obs_levels), 1.1 + 0.45 * len(row_labels))
    fig, axes = plt.subplots(
        len(obs_levels),
        len(quantum_levels),
        figsize=(3.2 * len(quantum_levels), fig_height),
        squeeze=False,
    )

    for row, obs in enumerate(obs_levels):
        for col, quantum_name in enumerate(quantum_levels):
            ax = axes[row, col]
            subset = data[
                (data["observation_noise_relative"] == obs)
                & (data["quantum_noise_name"].astype(str) == str(quantum_name))
            ]
            grid = np.full((len(row_labels), len(constraints)), np.nan)
            for _, cell in subset.iterrows():
                label = str(cell["benchmark_row_label"])
                n_constraints = float(cell["n_constraints"])
                winner = str(cell["winner_method_name"])
                if label not in label_to_idx or n_constraints not in constraint_to_idx or winner not in method_to_idx:
                    continue
                grid[label_to_idx[label], constraint_to_idx[n_constraints]] = method_to_idx[winner]

            ax.imshow(
                np.ma.masked_invalid(grid),
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=-0.5,
                vmax=len(method_names) - 0.5,
                origin="lower",
            )
            ax.set_xticks(range(len(constraints)))
            ax.set_xticklabels([str(int(v)) for v in constraints], fontsize=8)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=7)
            if row == len(obs_levels) - 1:
                ax.set_xlabel("constraints", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"obs={obs:g}", fontsize=9)
            ax.set_title(str(quantum_name), fontsize=10)

    legend_handles = [Patch(facecolor=cmap(idx), edgecolor="none", label=name) for idx, name in enumerate(method_names)]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=max(1, len(legend_handles)),
        frameon=False,
    )
    fig.suptitle("Benchmark-specific method-regime diagram", y=0.995)
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.22, right=0.98, hspace=0.3, wspace=0.22)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_benchmark_adaptive_gain_heatmap(comparison_table: pd.DataFrame, output_base: Path) -> None:
    data = _benchmark_regime_rows(comparison_table)
    if data.empty:
        return

    adaptive_cols = [
        column
        for column in data.columns
        if column.startswith("rmse_relative_gain_vs_global__") and "adaptive" in column.lower()
    ]
    if not adaptive_cols:
        return
    gain_col = adaptive_cols[0]
    method_name = gain_col.split("__", 1)[-1]

    obs_levels = sorted(data["observation_noise_relative"].dropna().unique())
    quantum_levels = list(dict.fromkeys(data["quantum_noise_name"].astype(str)))
    row_labels = (
        data.sort_values(["benchmark_label", "log_variance"])[["benchmark_label", "log_variance", "benchmark_row_label"]]
        .drop_duplicates()["benchmark_row_label"]
        .tolist()
    )
    constraints = sorted(data["n_constraints"].dropna().unique())
    if not obs_levels or not quantum_levels or not row_labels or not constraints:
        return

    label_to_idx = {label: idx for idx, label in enumerate(row_labels)}
    constraint_to_idx = {value: idx for idx, value in enumerate(constraints)}
    vmax = float(np.nanmax(np.abs(pd.to_numeric(data[gain_col], errors="coerce"))))
    vmax = max(vmax, 1e-6)

    fig_height = max(3.2 * len(obs_levels), 1.1 + 0.45 * len(row_labels))
    fig, axes = plt.subplots(
        len(obs_levels),
        len(quantum_levels),
        figsize=(3.2 * len(quantum_levels), fig_height),
        squeeze=False,
    )
    image = None

    for row, obs in enumerate(obs_levels):
        for col, quantum_name in enumerate(quantum_levels):
            ax = axes[row, col]
            subset = data[
                (data["observation_noise_relative"] == obs)
                & (data["quantum_noise_name"].astype(str) == str(quantum_name))
            ]
            grid = np.full((len(row_labels), len(constraints)), np.nan)
            for _, cell in subset.iterrows():
                label = str(cell["benchmark_row_label"])
                n_constraints = float(cell["n_constraints"])
                if label not in label_to_idx or n_constraints not in constraint_to_idx:
                    continue
                grid[label_to_idx[label], constraint_to_idx[n_constraints]] = float(cell[gain_col])

            image = ax.imshow(
                np.ma.masked_invalid(grid),
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                origin="lower",
            )
            ax.set_xticks(range(len(constraints)))
            ax.set_xticklabels([str(int(v)) for v in constraints], fontsize=8)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=7)
            if row == len(obs_levels) - 1:
                ax.set_xlabel("constraints", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"obs={obs:g}", fontsize=9)
            ax.set_title(str(quantum_name), fontsize=10)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92, pad=0.015)
        colorbar.set_label(f"relative RMSE gain of {method_name} vs global", fontsize=9)
    fig.suptitle("Benchmark-specific adaptive gain diagram", y=0.995)
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.22, right=0.92, hspace=0.3, wspace=0.22)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_benchmark_rmse_sensitivity(table: pd.DataFrame, output_base: Path) -> None:
    data = table[table["field"] == "concentration"].copy()
    data = _preferred_transport_subset(data)
    data = _benchmark_regime_rows(data)
    if data.empty:
        return

    data["rmse_mean"] = pd.to_numeric(data["rmse_mean"], errors="coerce")
    focus_variance = float(data["log_variance"].max())
    data = data[data["log_variance"] == focus_variance].copy()
    if data.empty:
        return

    obs_levels = sorted(data["observation_noise_relative"].dropna().unique())
    benchmark_labels = list(dict.fromkeys(data["benchmark_label"].astype(str)))
    method_names = list(dict.fromkeys(data["mcr_method_name"].astype(str)))
    quantum_levels = list(dict.fromkeys(data["quantum_noise_name"].astype(str)))
    constraint_levels = sorted(data["n_constraints"].dropna().unique())
    if not obs_levels or not benchmark_labels or not method_names or not quantum_levels or not constraint_levels:
        return

    cmap = plt.cm.get_cmap("tab10", max(1, len(method_names)))
    colors = {name: cmap(idx) for idx, name in enumerate(method_names)}
    linestyles = ["-", "--", ":"]
    line_map = {name: linestyles[idx % len(linestyles)] for idx, name in enumerate(quantum_levels)}

    fig, axes = plt.subplots(
        len(obs_levels),
        len(benchmark_labels),
        figsize=(3.2 * len(benchmark_labels), 2.8 * len(obs_levels)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, obs in enumerate(obs_levels):
        for col, benchmark_label in enumerate(benchmark_labels):
            ax = axes[row, col]
            subset = data[
                (data["observation_noise_relative"] == obs)
                & (data["benchmark_label"].astype(str) == str(benchmark_label))
            ]
            for method_name in method_names:
                for quantum_name in quantum_levels:
                    curve = subset[
                        (subset["mcr_method_name"].astype(str) == str(method_name))
                        & (subset["quantum_noise_name"].astype(str) == str(quantum_name))
                    ].sort_values("n_constraints")
                    if curve.empty:
                        continue
                    ax.plot(
                        curve["n_constraints"],
                        np.maximum(curve["rmse_mean"].to_numpy(dtype=float), 1e-18),
                        marker="o",
                        linewidth=1.6,
                        markersize=3.8,
                        color=colors[method_name],
                        linestyle=line_map[quantum_name],
                        label=f"{method_name} / {quantum_name}",
                    )

            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
            if row == 0:
                ax.set_title(str(benchmark_label), fontsize=10)
            if col == 0:
                ax.set_ylabel(f"obs={obs:g}\nRMSE", fontsize=9)
            if row == len(obs_levels) - 1:
                ax.set_xlabel("constraints", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=max(1, min(4, len(unique))),
        frameon=False,
    )
    fig.suptitle(rf"Benchmark RMSE sensitivity at $\sigma^2_{{\ln K}}={focus_variance:g}$", y=0.995)
    fig.subplots_adjust(top=0.83, bottom=0.12, left=0.08, right=0.98, hspace=0.3, wspace=0.22)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_capture_error_sensitivity(diagnostics_table: pd.DataFrame, output_base: Path) -> None:
    data = _preferred_transport_subset(diagnostics_table.copy())
    data = _benchmark_regime_rows(data)
    if data.empty or "capture_rate_scaled_error_mean" not in data.columns:
        return

    data["capture_rate_scaled_error_mean"] = pd.to_numeric(data["capture_rate_scaled_error_mean"], errors="coerce")
    data = data[data["capture_rate_scaled_error_mean"].notna()].copy()
    if data.empty:
        return

    focus_variance = float(data["log_variance"].max())
    data = data[data["log_variance"] == focus_variance].copy()
    if data.empty:
        return

    obs_levels = sorted(data["observation_noise_relative"].dropna().unique())
    benchmark_labels = list(dict.fromkeys(data["benchmark_label"].astype(str)))
    method_names = list(dict.fromkeys(data["mcr_method_name"].astype(str)))
    quantum_levels = list(dict.fromkeys(data["quantum_noise_name"].astype(str)))
    constraint_levels = sorted(data["n_constraints"].dropna().unique())
    if not obs_levels or not benchmark_labels or not method_names or not quantum_levels or not constraint_levels:
        return

    cmap = plt.cm.get_cmap("tab10", max(1, len(method_names)))
    colors = {name: cmap(idx) for idx, name in enumerate(method_names)}
    linestyles = ["-", "--", ":"]
    line_map = {name: linestyles[idx % len(linestyles)] for idx, name in enumerate(quantum_levels)}

    fig, axes = plt.subplots(
        len(obs_levels),
        len(benchmark_labels),
        figsize=(3.4 * len(benchmark_labels), 2.9 * len(obs_levels)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, obs in enumerate(obs_levels):
        for col, benchmark_label in enumerate(benchmark_labels):
            ax = axes[row, col]
            subset = data[
                (data["observation_noise_relative"] == obs)
                & (data["benchmark_label"].astype(str) == str(benchmark_label))
            ]
            for method_name in method_names:
                for quantum_name in quantum_levels:
                    curve = subset[
                        (subset["mcr_method_name"].astype(str) == str(method_name))
                        & (subset["quantum_noise_name"].astype(str) == str(quantum_name))
                    ].sort_values("n_constraints")
                    if curve.empty:
                        continue
                    ax.plot(
                        curve["n_constraints"],
                        np.maximum(curve["capture_rate_scaled_error_mean"].to_numpy(dtype=float), 1e-18),
                        marker="o",
                        linewidth=1.6,
                        markersize=3.8,
                        color=colors[method_name],
                        linestyle=line_map[quantum_name],
                    )

            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
            if row == 0:
                ax.set_title(str(benchmark_label), fontsize=10)
            if col == 0:
                ax.set_ylabel(f"obs={obs:g}\ncapture err.", fontsize=9)
            if row == len(obs_levels) - 1:
                ax.set_xlabel("constraints", fontsize=9)

    method_handles = [
        Line2D([0], [0], color=colors[name], linewidth=1.8, marker="o", markersize=4, label=name)
        for name in method_names
    ]
    noise_handles = [
        Line2D([0], [0], color="black", linewidth=1.8, linestyle=line_map[name], label=name)
        for name in quantum_levels
    ]
    fig.legend(method_handles, [h.get_label() for h in method_handles], loc="upper center", ncol=max(1, len(method_handles)), frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.legend(noise_handles, [h.get_label() for h in noise_handles], loc="upper center", ncol=max(1, len(noise_handles)), frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(rf"Capture-error sensitivity at $\sigma^2_{{\ln K}}={focus_variance:g}$", y=1.04)
    fig.subplots_adjust(top=0.8, bottom=0.13, left=0.08, right=0.98, hspace=0.3, wspace=0.22)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_method_regret_heatmap(normalized_summary: pd.DataFrame, output_base: Path) -> None:
    if normalized_summary.empty or "benchmark_label" not in normalized_summary.columns:
        return

    metric_specs = []
    if "rmse_mean_relative_regret_median" in normalized_summary.columns:
        metric_specs.append(("rmse_mean_relative_regret_median", "RMSE regret"))
    if "capture_rate_scaled_error_mean_relative_regret_median" in normalized_summary.columns:
        metric_specs.append(("capture_rate_scaled_error_mean_relative_regret_median", "Capture regret"))
    if "overall_regret_score_median" in normalized_summary.columns:
        metric_specs.append(("overall_regret_score_median", "Overall score"))
    if not metric_specs:
        return

    data = normalized_summary.copy()
    benchmark_labels = list(dict.fromkeys(data["benchmark_label"].astype(str)))
    method_names = list(dict.fromkeys(data["mcr_method_name"].astype(str)))
    if not benchmark_labels or not method_names:
        return

    fig, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=(3.0 * len(metric_specs), max(2.6, 0.62 * len(benchmark_labels) + 1.6)),
        squeeze=False,
    )

    for col, (metric_col, title) in enumerate(metric_specs):
        ax = axes[0, col]
        grid = (
            data.pivot_table(index="benchmark_label", columns="mcr_method_name", values=metric_col, aggfunc="mean")
            .reindex(index=benchmark_labels, columns=method_names)
            .to_numpy(dtype=float)
        )
        plot_grid = np.log10(1.0 + np.maximum(grid, 0.0))
        image = ax.imshow(np.ma.masked_invalid(plot_grid), aspect="auto", interpolation="nearest", cmap="YlOrRd", origin="upper")
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(benchmark_labels)))
        ax.set_yticklabels(benchmark_labels, fontsize=8)
        ax.set_title(title, fontsize=10)

        for i, benchmark_label in enumerate(benchmark_labels):
            for j, method_name in enumerate(method_names):
                value = grid[i, j]
                if not np.isfinite(value):
                    continue
                ax.text(
                    j,
                    i,
                    _fmt_pct(100.0 * value) if value < 10.0 else _fmt_sci(value),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black",
                )

        colorbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
        colorbar.set_label(r"$\log_{10}(1+\mathrm{median\ regret})$", fontsize=8)

    fig.suptitle("Benchmark-normalized method regret summary", y=0.98)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.15, right=0.98, wspace=0.3)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_capture_tradeoff(normalized_summary: pd.DataFrame, output_base: Path) -> None:
    required = {"benchmark_label", "mcr_method_name", "rmse_mean_relative_regret_median", "capture_rate_scaled_error_mean_relative_regret_median"}
    if normalized_summary.empty or not required.issubset(normalized_summary.columns):
        return

    data = normalized_summary.copy()
    data = data[
        data["rmse_mean_relative_regret_median"].notna()
        & data["capture_rate_scaled_error_mean_relative_regret_median"].notna()
    ].copy()
    if data.empty:
        return

    method_names = list(dict.fromkeys(data["mcr_method_name"].astype(str)))
    benchmark_labels = list(dict.fromkeys(data["benchmark_label"].astype(str)))
    cmap = plt.cm.get_cmap("tab10", max(1, len(method_names)))
    colors = {name: cmap(idx) for idx, name in enumerate(method_names)}
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_map = {name: markers[idx % len(markers)] for idx, name in enumerate(benchmark_labels)}

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for _, row in data.iterrows():
        method_name = str(row["mcr_method_name"])
        benchmark_label = str(row["benchmark_label"])
        x_value = float(row["rmse_mean_relative_regret_median"])
        y_value = float(row["capture_rate_scaled_error_mean_relative_regret_median"])
        ax.scatter(
            max(x_value, 1e-8),
            max(y_value, 1e-8),
            s=64,
            color=colors[method_name],
            marker=marker_map[benchmark_label],
            alpha=0.9,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median RMSE relative regret", fontsize=9)
    ax.set_ylabel("median capture relative regret", fontsize=9)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)

    method_handles = [
        Line2D([0], [0], linestyle="None", marker="o", markersize=6, color=colors[name], label=name)
        for name in method_names
    ]
    benchmark_handles = [
        Line2D([0], [0], linestyle="None", marker=marker_map[name], markersize=6, color="black", label=name)
        for name in benchmark_labels
    ]
    legend1 = ax.legend(method_handles, [h.get_label() for h in method_handles], loc="upper left", frameon=False, title="Method")
    ax.add_artist(legend1)
    ax.legend(benchmark_handles, [h.get_label() for h in benchmark_handles], loc="lower right", frameon=False, title="Benchmark")

    fig.suptitle("RMSE-capture tradeoff in normalized benchmark space", y=0.98)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_figures(
    table: pd.DataFrame,
    diagnostics_table: pd.DataFrame,
    phase_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
    normalized_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    has_benchmark_suite = "benchmark_name" in table.columns and table["benchmark_name"].notna().any()

    if has_benchmark_suite:
        _plot_benchmark_rmse_sensitivity(table, figure_dir / "fig_benchmark_rmse_sensitivity")
        _plot_benchmark_phase_diagram(phase_table, figure_dir / "fig_benchmark_phase_diagram")
        _plot_benchmark_adaptive_gain_heatmap(comparison_table, figure_dir / "fig_benchmark_adaptive_gain")
        _plot_method_regret_heatmap(normalized_summary, figure_dir / "fig_benchmark_regret_heatmap")
        _plot_capture_error_sensitivity(diagnostics_table, figure_dir / "fig_capture_error_sensitivity")
        _plot_capture_tradeoff(normalized_summary, figure_dir / "fig_capture_tradeoff")
        return

    _plot_rmse_grid(
        table,
        field="head",
        metric_column="rmse_mean",
        ylabel="head RMSE",
        output_base=figure_dir / "fig_head_rmse_sensitivity",
    )
    _plot_rmse_grid(
        table,
        field="concentration",
        metric_column="rmse_mean",
        ylabel="concentration RMSE",
        output_base=figure_dir / "fig_concentration_rmse_sensitivity",
    )
    _plot_mass_peak(table, figure_dir / "fig_mass_peak_sensitivity")
    _plot_diagnostics_grid(diagnostics_table, figure_dir / "fig_transport_diagnostics_sensitivity")
    _plot_method_phase_diagram(phase_table, figure_dir / "fig_method_phase_diagram")
    _plot_adaptive_gain_heatmap(comparison_table, figure_dir / "fig_adaptive_gain_phase")


def run_sweep(config: dict[str, Any], *, output_root: str | Path | None = None) -> Path:
    sweep_name = str(config.get("name", "noise_constraint_heterogeneity_sweep"))
    root = Path(output_root or config.get("output", {}).get("root", "outputs/sweeps"))
    sweep_dir = root / sweep_name
    runs_dir = sweep_dir / "runs"
    figures_dir = sweep_dir / "figures"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_config = deepcopy(config["base_experiment"])
    sweep = config["sweep"]
    heterogeneity = sweep.get("heterogeneity", {})
    base_log_variance = float(base_config.get("k_field", {}).get("log_variance", 1.0))
    base_seed = int(base_config.get("k_field", {}).get("seed", 101))
    log_variances = [float(v) for v in heterogeneity.get("log_variances", [base_log_variance])]
    k_seeds = [int(seed) for seed in heterogeneity.get("seeds", [base_seed])]
    benchmark_cases = _resolved_benchmark_cases(sweep)
    quantum_noise_levels = sweep["quantum_noise_levels"]
    obs_noise_levels = [float(v) for v in sweep["observation_noise_relative"]]
    obs_noise_absolute = float(sweep.get("observation_noise_absolute", 0.0))
    constraint_counts = [int(v) for v in sweep["constraint_counts"]]
    constraint_seed_base = int(sweep.get("constraint_seed_base", 1000))

    all_rows: list[dict[str, Any]] = []
    run_index = 0
    total_runs = (
        len(benchmark_cases)
        * len(log_variances)
        * len(k_seeds)
        * len(quantum_noise_levels)
        * len(obs_noise_levels)
    )

    for benchmark_case in benchmark_cases:
        benchmark_meta = benchmark_case.get("benchmark", {}) if benchmark_case else {}
        benchmark_label = str(benchmark_meta.get("label", benchmark_meta.get("name", "baseline")))
        for log_variance in log_variances:
            for k_seed in k_seeds:
                for quantum_noise in quantum_noise_levels:
                    for obs_idx, obs_noise in enumerate(obs_noise_levels):
                        run_index += 1
                        constraint_seed = constraint_seed_base + run_index * 17 + obs_idx
                        run_config = _make_run_config(
                            base_config,
                            sweep_name=sweep_name,
                            log_variance=log_variance,
                            k_seed=k_seed,
                            quantum_noise=quantum_noise,
                            observation_noise_relative=obs_noise,
                            observation_noise_absolute=obs_noise_absolute,
                            constraint_seed=constraint_seed,
                            constraint_counts=constraint_counts,
                            benchmark_case=benchmark_case,
                        )
                        print(
                            f"[{run_index:03d}/{total_runs:03d}] "
                            f"benchmark={benchmark_label}, var={log_variance:g}, seed={k_seed}, "
                            f"q={quantum_noise['name']}, obs={obs_noise:g}"
                        )
                        out_dir = run_experiment(run_config, output_root=runs_dir)
                        rows = _read_metrics(out_dir / "metrics.csv")
                        for row in rows:
                            row.update(
                                {
                                    "sweep_name": sweep_name,
                                    "run_id": run_config["name"],
                                    "log_variance": log_variance,
                                    "k_seed": k_seed,
                                    "quantum_noise_name": quantum_noise["name"],
                                    "observation_noise_relative": obs_noise,
                                    "observation_noise_absolute": obs_noise_absolute,
                                    "constraint_seed": constraint_seed,
                                }
                            )
                            if benchmark_meta:
                                row.setdefault("benchmark_name", benchmark_meta.get("name", ""))
                                row.setdefault("benchmark_label", benchmark_meta.get("label", ""))
                                row.setdefault("benchmark_family", benchmark_meta.get("family", ""))
                        all_rows.extend(rows)

    all_metrics_path = sweep_dir / "all_metrics.csv"
    _write_rows(all_metrics_path, all_rows)

    all_metrics = pd.read_csv(all_metrics_path)
    main_table = build_main_table(all_metrics, sweep_dir / "main_table.csv")
    diagnostics_table = build_diagnostics_table(
        all_metrics,
        sweep_dir / "transport_diagnostics_table.csv",
    )
    manuscript_summary = build_manuscript_summary(
        main_table,
        sweep_dir / "manuscript_summary_table.csv",
    )
    method_comparison = build_method_comparison_table(
        main_table,
        sweep_dir / "method_comparison_table.csv",
    )
    method_phase = build_method_phase_table(
        main_table,
        sweep_dir / "method_phase_table.csv",
    )
    method_winner_summary = build_method_winner_summary(
        method_phase,
        sweep_dir / "method_winner_summary.csv",
    )
    benchmark_winner_summary = build_benchmark_winner_summary(
        method_phase,
        sweep_dir / "benchmark_winner_summary.csv",
    )
    normalized_table = build_benchmark_normalized_table(
        main_table,
        diagnostics_table,
        sweep_dir / "benchmark_normalized_table.csv",
    )
    normalized_summary = build_benchmark_normalized_summary(
        normalized_table,
        sweep_dir / "benchmark_normalized_summary.csv",
    )
    capture_manuscript = build_capture_manuscript_table(
        normalized_table,
        sweep_dir / "capture_manuscript_table.csv",
    )
    build_results_brief(
        manuscript_summary,
        diagnostics_table,
        sweep_dir / "results_brief.md",
        normalized_summary=normalized_summary,
    )
    make_figures(
        main_table,
        diagnostics_table,
        method_phase,
        method_comparison,
        normalized_summary,
        figures_dir,
    )

    summary = {
        "sweep_name": sweep_name,
        "n_runs": total_runs,
        "n_metric_rows": len(all_rows),
        "all_metrics_csv": str(all_metrics_path),
        "main_table_csv": str(sweep_dir / "main_table.csv"),
        "transport_diagnostics_table_csv": str(sweep_dir / "transport_diagnostics_table.csv"),
        "manuscript_summary_table_csv": str(sweep_dir / "manuscript_summary_table.csv"),
        "method_comparison_table_csv": str(sweep_dir / "method_comparison_table.csv"),
        "method_phase_table_csv": str(sweep_dir / "method_phase_table.csv"),
        "method_winner_summary_csv": str(sweep_dir / "method_winner_summary.csv"),
        "benchmark_winner_summary_csv": str(sweep_dir / "benchmark_winner_summary.csv"),
        "benchmark_normalized_table_csv": str(sweep_dir / "benchmark_normalized_table.csv"),
        "benchmark_normalized_summary_csv": str(sweep_dir / "benchmark_normalized_summary.csv"),
        "capture_manuscript_table_csv": str(sweep_dir / "capture_manuscript_table.csv"),
        "results_brief_md": str(sweep_dir / "results_brief.md"),
        "n_manuscript_summary_rows": int(len(manuscript_summary)),
        "n_method_phase_rows": int(len(method_phase)),
        "n_method_winner_summary_rows": int(len(method_winner_summary)),
        "n_benchmark_winner_summary_rows": int(len(benchmark_winner_summary)),
        "n_benchmark_normalized_rows": int(len(normalized_table)),
        "n_benchmark_normalized_summary_rows": int(len(normalized_summary)),
        "n_capture_manuscript_rows": int(len(capture_manuscript)),
        "figures_dir": str(figures_dir),
    }
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (sweep_dir / "sweep_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return sweep_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a quantum-hydro batch sweep.")
    parser.add_argument("--config", required=True, help="Path to sweep JSON/YAML config.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = run_sweep(config, output_root=args.output_root)
    print(f"[OK] Sweep saved to {out_dir}")


if __name__ == "__main__":
    main()
