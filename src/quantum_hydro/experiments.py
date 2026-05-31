"""Configuration-driven experiment runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import linalg as spla

from .anchors import select_anchor_indices
from .config import ensure_output_dir, load_config
from .kfields import generate_k_field
from .mcr import MCRResult, apply_mcr_method
from .metrics import (
    breakthrough_summary,
    concentration_metrics,
    condition_number_estimate,
    field_metrics,
    matrix_metrics,
    stringify_metrics,
    trace_summary_error_metrics,
    transport_trace_row,
)
from .operators import GroundwaterModel, SourceTerm, WellTerm
from .solvers import QuantumStateResult, solver_from_config


@dataclass
class FlowRegime:
    start_step: int
    end_step: int
    wells: list[WellTerm]
    flow_matrix: Any
    flow_rhs: np.ndarray
    head: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    transport_matrix: Any
    solve_transport_step: Any


@dataclass
class SchedulePhase:
    phase_index: int
    start_step: int
    end_step: int
    regime_index: int
    label: str
    net_well_rate: float
    total_source_strength: float


def _parse_sources(raw_sources: list[dict[str, Any]]) -> list[SourceTerm]:
    return [
        SourceTerm(
            i=int(item["i"]),
            j=int(item["j"]),
            value=float(item["value"]),
            name=str(item.get("name", "")),
            start_step=int(item.get("start_step", 1)),
            end_step=None if item.get("end_step") is None else int(item["end_step"]),
            profile=str(item.get("profile", "constant")),
            peak_step=None if item.get("peak_step") is None else int(item["peak_step"]),
            ramp_steps=int(item.get("ramp_steps", 0)),
            decay_tau_steps=None
            if item.get("decay_tau_steps") is None
            else float(item["decay_tau_steps"]),
        )
        for item in raw_sources
    ]


def _parse_flow_wells(raw_wells: list[dict[str, Any]]) -> list[WellTerm]:
    return [
        WellTerm(
            i=int(item["i"]),
            j=int(item["j"]),
            value=float(item["value"]),
            name=str(item.get("name", "")),
            start_step=int(item.get("start_step", 1)),
            end_step=None if item.get("end_step") is None else int(item["end_step"]),
            profile=str(item.get("profile", "constant")),
            peak_step=None if item.get("peak_step") is None else int(item["peak_step"]),
            ramp_steps=int(item.get("ramp_steps", 0)),
            decay_tau_steps=None
            if item.get("decay_tau_steps") is None
            else float(item["decay_tau_steps"]),
        )
        for item in raw_wells
    ]


def _source_points(sources: list[SourceTerm]) -> list[tuple[int, int]]:
    return [(source.i, source.j) for source in sources]


def _active_flow_wells(
    wells: list[WellTerm],
    *,
    step_idx: int,
) -> list[WellTerm]:
    active: list[WellTerm] = []
    for well in wells:
        value = float(well.value_at_step(step_idx))
        if abs(value) < 1e-30:
            continue
        active.append(
            WellTerm(
                i=well.i,
                j=well.j,
                value=value,
                name=well.name,
            )
        )
    return active


def _well_schedule_signature(wells: list[WellTerm], *, step_idx: int) -> tuple[float, ...]:
    return tuple(round(float(well.value_at_step(step_idx)), 12) for well in wells)


def _source_schedule_signature(sources: list[SourceTerm], *, step_idx: int) -> tuple[float, ...]:
    return tuple(round(float(source.value_at_step(step_idx)), 12) for source in sources)


def _sum_well_rate(wells: list[WellTerm], *, step_idx: int) -> float:
    return float(sum(well.value_at_step(step_idx) for well in wells))


def _sum_source_rate(sources: list[SourceTerm], *, step_idx: int) -> float:
    return float(sum(source.value_at_step(step_idx) for source in sources))


def _has_transient_wells(wells: list[WellTerm]) -> bool:
    return any(
        well.end_step is not None
        or str(well.profile).lower() not in {"constant", "boxcar"}
        or int(well.start_step) > 1
        for well in wells
    )


def _build_flow_regimes(
    model: GroundwaterModel,
    *,
    h_left: float,
    h_right: float,
    steps: int,
    flow_wells: list[WellTerm],
    flow_boundary: dict[str, Any] | None,
) -> list[FlowRegime]:
    regimes: list[FlowRegime] = []
    current_signature: tuple[float, ...] | None = None

    for step_idx in range(1, int(steps) + 1):
        signature = _well_schedule_signature(flow_wells, step_idx=step_idx)
        if current_signature == signature and regimes:
            regimes[-1].end_step = step_idx
            continue

        flow_matrix, flow_rhs = model.build_flow_matrix(
            h_left,
            h_right,
            wells=flow_wells,
            flow_boundary=flow_boundary,
            step_idx=step_idx,
        )
        head = spla.spsolve(flow_matrix, flow_rhs).reshape(model.ny, model.nx)
        vx, vy = model.compute_velocity(head)
        transport_matrix, _ = model.build_transport_operators(vx, vy)
        regimes.append(
            FlowRegime(
                start_step=step_idx,
                end_step=step_idx,
                wells=_active_flow_wells(flow_wells, step_idx=step_idx),
                flow_matrix=flow_matrix,
                flow_rhs=np.asarray(flow_rhs, dtype=float),
                head=head,
                vx=vx,
                vy=vy,
                transport_matrix=transport_matrix,
                solve_transport_step=spla.factorized(transport_matrix),
            )
        )
        current_signature = signature

    if not regimes:
        flow_matrix, flow_rhs = model.build_flow_matrix(
            h_left,
            h_right,
            wells=flow_wells,
            flow_boundary=flow_boundary,
            step_idx=1,
        )
        head = spla.spsolve(flow_matrix, flow_rhs).reshape(model.ny, model.nx)
        vx, vy = model.compute_velocity(head)
        transport_matrix, _ = model.build_transport_operators(vx, vy)
        regimes.append(
            FlowRegime(
                start_step=1,
                end_step=max(1, int(steps)),
                wells=_active_flow_wells(flow_wells, step_idx=1),
                flow_matrix=flow_matrix,
                flow_rhs=np.asarray(flow_rhs, dtype=float),
                head=head,
                vx=vx,
                vy=vy,
                transport_matrix=transport_matrix,
                solve_transport_step=spla.factorized(transport_matrix),
            )
        )

    return regimes


def _schedule_phase_label(
    *,
    net_well_rate: float,
    total_source_strength: float,
) -> str:
    well_eps = 1e-12
    source_eps = 1e-12
    source_on = total_source_strength > source_eps
    if net_well_rate > well_eps:
        return "injection" if source_on else "injection_flush"
    if net_well_rate < -well_eps:
        return "pumpback" if not source_on else "extraction_with_source"
    if source_on:
        return "source_release"
    return "background"


def _build_schedule_phases(
    *,
    flow_regimes: list[FlowRegime],
    steps: int,
    sources: list[SourceTerm],
    flow_wells: list[WellTerm],
) -> tuple[list[SchedulePhase], np.ndarray]:
    phases: list[SchedulePhase] = []
    step_to_phase = np.full(max(0, int(steps)) + 1, fill_value=-1, dtype=int)
    regime_index = 0
    current_signature: tuple[Any, ...] | None = None

    for step_idx in range(1, int(steps) + 1):
        while regime_index < len(flow_regimes) - 1 and step_idx > flow_regimes[regime_index].end_step:
            regime_index += 1
        well_signature = _well_schedule_signature(flow_wells, step_idx=step_idx)
        source_signature = _source_schedule_signature(sources, step_idx=step_idx)
        signature: tuple[Any, ...] = (regime_index, well_signature, source_signature)
        if signature != current_signature:
            net_well_rate = _sum_well_rate(flow_wells, step_idx=step_idx)
            total_source_strength = _sum_source_rate(sources, step_idx=step_idx)
            phases.append(
                SchedulePhase(
                    phase_index=len(phases),
                    start_step=step_idx,
                    end_step=step_idx,
                    regime_index=regime_index,
                    label=_schedule_phase_label(
                        net_well_rate=net_well_rate,
                        total_source_strength=total_source_strength,
                    ),
                    net_well_rate=net_well_rate,
                    total_source_strength=total_source_strength,
                )
            )
            current_signature = signature
        else:
            phases[-1].end_step = step_idx
        step_to_phase[step_idx] = phases[-1].phase_index

    if not phases:
        phases.append(
            SchedulePhase(
                phase_index=0,
                start_step=1,
                end_step=max(1, int(steps)),
                regime_index=0,
                label="background",
                net_well_rate=0.0,
                total_source_strength=0.0,
            )
        )
        step_to_phase[:] = 0

    return phases, step_to_phase


def _phase_config_value(mapping: Any, phase_label: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    label = str(phase_label).strip().lower()
    candidates = [label]
    if label.startswith("injection"):
        candidates.append("injection")
    if label.startswith("pumpback") or label.startswith("extraction"):
        candidates.extend(["pumpback", "extraction"])
    if label.startswith("source_release"):
        candidates.append("source_release")
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _observation_stride_for_phase(
    transport_quantum_config: dict[str, Any],
    *,
    phase_label: str,
) -> int:
    default_stride = max(1, int(transport_quantum_config.get("observation_stride", 1)))
    override = _phase_config_value(transport_quantum_config.get("phase_observation_strides"), phase_label)
    if override is None:
        return default_stride
    return max(1, int(override))


def _phase_constraint_config(
    constraint_config: dict[str, Any],
    *,
    transport_quantum_config: dict[str, Any],
    phase_label: str,
) -> dict[str, Any]:
    resolved = dict(constraint_config)
    overrides = _phase_config_value(transport_quantum_config.get("phase_constraint_overrides"), phase_label)
    if not isinstance(overrides, dict):
        return resolved

    constraint_type = str(constraint_config.get("type", "")).lower()
    constraint_name = str(constraint_config.get("name", "")).lower()
    for key in (constraint_type, constraint_name, "*"):
        if key in overrides and isinstance(overrides[key], dict):
            resolved.update(overrides[key])
    return resolved


def _phase_for_step(
    *,
    schedule_phases: list[SchedulePhase],
    step_to_phase: np.ndarray,
    step_idx: int,
) -> SchedulePhase:
    phase_index = int(
        step_to_phase[step_idx]
        if 0 <= step_idx < step_to_phase.size
        else schedule_phases[-1].phase_index
    )
    return schedule_phases[max(0, min(phase_index, len(schedule_phases) - 1))]


def _should_observe_step(
    *,
    step_idx: int,
    steps: int,
    phase_start_step: int,
    stride: int,
) -> bool:
    return (
        step_idx == 1
        or step_idx == int(steps)
        or (step_idx - int(phase_start_step)) % max(1, int(stride)) == 0
    )


def _build_observation_plan(
    *,
    steps: int,
    schedule_phases: list[SchedulePhase],
    step_to_phase: np.ndarray,
    transport_quantum_config: dict[str, Any],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for step_idx in range(1, int(steps) + 1):
        phase_info = _phase_for_step(
            schedule_phases=schedule_phases,
            step_to_phase=step_to_phase,
            step_idx=step_idx,
        )
        phase_stride = _observation_stride_for_phase(
            transport_quantum_config,
            phase_label=str(phase_info.label),
        )
        if not _should_observe_step(
            step_idx=step_idx,
            steps=steps,
            phase_start_step=int(phase_info.start_step),
            stride=phase_stride,
        ):
            continue
        plan.append(
            {
                "step_idx": int(step_idx),
                "phase_info": phase_info,
                "phase_stride": int(phase_stride),
            }
        )
    return plan


def _global_reference_observation_steps(*, steps: int, stride: int) -> list[int]:
    observed: list[int] = []
    for step_idx in range(1, int(steps) + 1):
        if step_idx == 1 or step_idx == int(steps) or (step_idx - 1) % max(1, int(stride)) == 0:
            observed.append(int(step_idx))
    return observed


def _phase_reference_observation_steps(
    *,
    steps: int,
    schedule_phases: list[SchedulePhase],
    step_to_phase: np.ndarray,
    stride: int,
) -> list[int]:
    observed: list[int] = []
    for step_idx in range(1, int(steps) + 1):
        phase_info = _phase_for_step(
            schedule_phases=schedule_phases,
            step_to_phase=step_to_phase,
            step_idx=step_idx,
        )
        if _should_observe_step(
            step_idx=step_idx,
            steps=steps,
            phase_start_step=int(phase_info.start_step),
            stride=stride,
        ):
            observed.append(int(step_idx))
    return observed


def _resolve_step_anchor_budgets(
    *,
    constraint_config: dict[str, Any],
    observation_plan: list[dict[str, Any]],
    transport_quantum_config: dict[str, Any],
    steps: int,
    schedule_phases: list[SchedulePhase],
    step_to_phase: np.ndarray,
) -> tuple[dict[int, int], dict[str, Any]]:
    base_m = constraint_config.get("m")
    if base_m is None:
        return {}, {
            "enabled": False,
            "base_m": np.nan,
            "reference_mode": "",
            "reference_stride": np.nan,
            "reference_observation_steps": np.nan,
            "reference_total_observations": np.nan,
            "phase_summary": {},
        }

    base_m_int = max(1, int(base_m))
    factor_mapping = transport_quantum_config.get("phase_anchor_budget_factors")
    budget_match_enabled = bool(transport_quantum_config.get("budget_match_total_observations", False))
    if not budget_match_enabled and not isinstance(factor_mapping, dict):
        return {}, {
            "enabled": False,
            "base_m": int(base_m_int),
            "reference_mode": "",
            "reference_stride": np.nan,
            "reference_observation_steps": np.nan,
            "reference_total_observations": np.nan,
            "phase_summary": {},
        }

    if not observation_plan:
        return {}, {
            "enabled": budget_match_enabled or isinstance(factor_mapping, dict),
            "base_m": int(base_m_int),
            "reference_mode": "",
            "reference_stride": np.nan,
            "reference_observation_steps": 0,
            "reference_total_observations": 0,
            "phase_summary": {},
        }

    step_indices = [int(item["step_idx"]) for item in observation_plan]
    phase_labels = [str(item["phase_info"].label) for item in observation_plan]
    phase_factors = np.asarray(
        [
            max(
                0.0,
                float(
                    _phase_config_value(factor_mapping, phase_label)
                    if _phase_config_value(factor_mapping, phase_label) is not None
                    else 1.0
                ),
            )
            for phase_label in phase_labels
        ],
        dtype=float,
    )
    if not np.any(phase_factors > 0.0):
        phase_factors[:] = 1.0

    step_budgets: np.ndarray
    reference_mode = ""
    reference_stride = np.nan
    reference_steps: list[int] = []

    if budget_match_enabled:
        reference_mode = str(transport_quantum_config.get("budget_match_reference_mode", "global_stride")).strip().lower()
        reference_stride = max(
            1,
            int(
                transport_quantum_config.get(
                    "budget_match_reference_stride",
                    transport_quantum_config.get("observation_stride", 1),
                )
            ),
        )
        if reference_mode in {"phase_reset_stride", "phase_reset", "phase_aware_stride"}:
            reference_steps = _phase_reference_observation_steps(
                steps=steps,
                schedule_phases=schedule_phases,
                step_to_phase=step_to_phase,
                stride=int(reference_stride),
            )
        else:
            reference_mode = "global_stride"
            reference_steps = _global_reference_observation_steps(
                steps=steps,
                stride=int(reference_stride),
            )

        target_total_observations = int(base_m_int * len(reference_steps))
        min_per_observation = max(0, int(transport_quantum_config.get("phase_anchor_budget_minimum", 1)))
        if min_per_observation * len(step_indices) > target_total_observations:
            min_per_observation = max(0, target_total_observations // max(1, len(step_indices)))

        effective_weights = phase_factors / max(float(np.sum(phase_factors)), 1e-30)
        discretionary_total = max(0, int(target_total_observations - min_per_observation * len(step_indices)))
        raw_discretionary = discretionary_total * effective_weights
        step_budgets = np.full(len(step_indices), fill_value=min_per_observation, dtype=int)
        step_budgets += np.floor(raw_discretionary).astype(int)
        remainder = int(target_total_observations - int(np.sum(step_budgets)))
        if remainder > 0:
            fractional = raw_discretionary - np.floor(raw_discretionary)
            order = sorted(
                range(len(step_indices)),
                key=lambda idx: (float(fractional[idx]), float(phase_factors[idx]), -step_indices[idx]),
                reverse=True,
            )
            for idx in order[:remainder]:
                step_budgets[idx] += 1
    else:
        reference_steps = []
        step_budgets = np.maximum(
            1,
            np.rint(base_m_int * phase_factors).astype(int),
        )

    phase_summary: dict[str, dict[str, Any]] = {}
    for step_idx, phase_label, factor, budget in zip(step_indices, phase_labels, phase_factors, step_budgets):
        summary = phase_summary.setdefault(
            phase_label,
            {
                "factor": float(factor),
                "n_observation_steps": 0,
                "total_budget": 0,
                "min_budget": int(budget),
                "max_budget": int(budget),
            },
        )
        summary["n_observation_steps"] = int(summary["n_observation_steps"]) + 1
        summary["total_budget"] = int(summary["total_budget"]) + int(budget)
        summary["min_budget"] = min(int(summary["min_budget"]), int(budget))
        summary["max_budget"] = max(int(summary["max_budget"]), int(budget))

    for summary in phase_summary.values():
        n_phase_steps = max(1, int(summary["n_observation_steps"]))
        summary["mean_budget"] = float(summary["total_budget"]) / n_phase_steps

    return (
        {int(step_idx): int(budget) for step_idx, budget in zip(step_indices, step_budgets)},
        {
            "enabled": bool(budget_match_enabled or isinstance(factor_mapping, dict)),
            "base_m": int(base_m_int),
            "reference_mode": str(reference_mode),
            "reference_stride": reference_stride,
            "reference_observation_steps": int(len(reference_steps)),
            "reference_total_observations": int(base_m_int * len(reference_steps))
            if budget_match_enabled
            else np.nan,
            "phase_summary": phase_summary,
        },
    )


def _summarize_shared_observations(
    shared_observations: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for record in shared_observations.values():
        metadata = dict(record.get("anchor_metadata", {}))
        phase_label = str(metadata.get("phase_label", "")).strip().lower()
        if not phase_label:
            continue
        n_obs = int(np.asarray(record.get("indices", []), dtype=int).size)
        phase_summary = summary.setdefault(
            phase_label,
            {
                "n_observation_steps": 0,
                "total_observations": 0,
                "min_budget": n_obs,
                "max_budget": n_obs,
            },
        )
        phase_summary["n_observation_steps"] = int(phase_summary["n_observation_steps"]) + 1
        phase_summary["total_observations"] = int(phase_summary["total_observations"]) + n_obs
        phase_summary["min_budget"] = min(int(phase_summary["min_budget"]), n_obs)
        phase_summary["max_budget"] = max(int(phase_summary["max_budget"]), n_obs)

    for phase_summary in summary.values():
        n_phase_steps = max(1, int(phase_summary["n_observation_steps"]))
        phase_summary["mean_budget"] = float(phase_summary["total_observations"]) / n_phase_steps

    return summary


def _apply_observation_noise(
    values: np.ndarray,
    noise_config: dict[str, Any],
    rng: np.random.Generator,
    *,
    noise_key: str | None = None,
) -> np.ndarray:
    relative_std = float(noise_config.get("relative_std", 0.0))
    absolute_std = float(noise_config.get("absolute_std", 0.0))
    if relative_std <= 0.0 and absolute_std <= 0.0:
        return values
    local_rng = rng
    if noise_key is not None:
        base_seed = int(noise_config.get("seed", 0))
        token = f"observation_noise|{base_seed}|{noise_key}".encode("utf-8")
        digest = hashlib.blake2b(token, digest_size=16).digest()
        local_rng = np.random.default_rng(int.from_bytes(digest, "little", signed=False))
    scale = absolute_std + relative_std * np.maximum(np.abs(values), 1e-15)
    return values + local_rng.normal(0.0, scale, size=values.shape)


def _observation_scale(values: np.ndarray, noise_config: dict[str, Any]) -> np.ndarray:
    relative_std = float(noise_config.get("relative_std", 0.0))
    absolute_std = float(noise_config.get("absolute_std", 0.0))
    floor = float(noise_config.get("noise_floor", 1e-12))
    return absolute_std + relative_std * np.maximum(np.abs(values), floor)


def _method_coordinate_field(
    model: GroundwaterModel,
    method_config: dict[str, Any],
) -> np.ndarray:
    basis = str(method_config.get("basis", "")).lower()
    axis = str(method_config.get("axis", "x")).lower()
    if basis.startswith("y_"):
        axis = "y"
    elif basis.startswith("x_"):
        axis = "x"
    elif basis.startswith("radial"):
        axis = "radial"

    if axis == "y":
        return np.repeat(np.linspace(0.0, 1.0, model.ny), model.nx)
    if axis == "radial":
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, model.ny),
            np.linspace(0.0, 1.0, model.nx),
            indexing="ij",
        )
        center_i = method_config.get("center_i")
        center_j = method_config.get("center_j")
        center_fraction_x = method_config.get("center_fraction_x")
        center_fraction_y = method_config.get("center_fraction_y")
        if center_i is not None:
            cy = 0.0 if model.ny <= 1 else float(center_i) / float(model.ny - 1)
        elif center_fraction_y is not None:
            cy = float(center_fraction_y)
        else:
            cy = 0.5
        if center_j is not None:
            cx = 0.0 if model.nx <= 1 else float(center_j) / float(model.nx - 1)
        elif center_fraction_x is not None:
            cx = float(center_fraction_x)
        else:
            cx = 0.5
        radius = np.sqrt((xx - np.clip(cx, 0.0, 1.0)) ** 2 + (yy - np.clip(cy, 0.0, 1.0)) ** 2)
        return (radius / max(float(radius.max()), 1e-15)).reshape(-1)
    return np.tile(np.linspace(0.0, 1.0, model.nx), model.ny)


def _anchor_window_field(
    model: GroundwaterModel,
    indices: np.ndarray,
    *,
    bandwidth_cells: float | None,
    power: float,
) -> tuple[np.ndarray, float]:
    idx = np.asarray(indices, dtype=int).reshape(-1)
    if idx.size == 0:
        return np.ones(model.n_nodes, dtype=float), float(max(model.nx, model.ny))

    anchor_points = np.asarray([model.unravel(int(value)) for value in idx], dtype=float)
    if bandwidth_cells is None:
        if anchor_points.shape[0] >= 2:
            deltas = anchor_points[:, None, :] - anchor_points[None, :, :]
            distances = np.sqrt(np.sum(deltas**2, axis=2))
            distances[distances < 1e-12] = np.inf
            nearest = np.min(distances, axis=1)
            finite = nearest[np.isfinite(nearest)]
            auto_bandwidth = np.median(finite) * 1.25 if finite.size else max(model.nx, model.ny) / 6.0
        else:
            auto_bandwidth = max(model.nx, model.ny) / 6.0
        bandwidth = max(float(auto_bandwidth), 1.5)
    else:
        bandwidth = max(float(bandwidth_cells), 1.0)

    yy, xx = np.meshgrid(np.arange(model.ny, dtype=float), np.arange(model.nx, dtype=float), indexing="ij")
    grid = np.column_stack([yy.reshape(-1), xx.reshape(-1)])
    deltas = grid[:, None, :] - anchor_points[None, :, :]
    nearest_distance = np.min(np.sqrt(np.sum(deltas**2, axis=2)), axis=1)
    window = np.exp(-0.5 * (nearest_distance / bandwidth) ** 2)
    return np.clip(window ** max(float(power), 1e-6), 0.0, 1.0), float(bandwidth)


def _positive_holdout_support(
    ratio_value: Any,
    *,
    threshold: float,
    scale: float,
) -> float:
    if ratio_value is None:
        return 0.0
    try:
        ratio = float(ratio_value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(ratio):
        return 0.0
    margin = max(float(threshold) - ratio, 0.0)
    if scale <= 0.0:
        return 1.0 if margin > 0.0 else 0.0
    return float(np.clip(1.0 - np.exp(-margin / max(float(scale), 1e-9)), 0.0, 1.0))


def _phase_bootstrap_support(
    method_config: dict[str, Any],
    *,
    phase_label: str,
) -> float:
    label = str(phase_label).strip().lower()
    candidate_keys = [
        f"local_evidence_phase_bootstrap_support_{label}",
    ]
    if label.startswith("injection"):
        candidate_keys.append("local_evidence_phase_bootstrap_support_injection")
    if label.startswith("pumpback") or label.startswith("extraction"):
        candidate_keys.append("local_evidence_phase_bootstrap_support_pumpback")
        candidate_keys.append("local_evidence_phase_bootstrap_support_extraction")
    if label.startswith("source_release"):
        candidate_keys.append("local_evidence_phase_bootstrap_support_source_release")
    for key in candidate_keys:
        if key in method_config:
            return float(np.clip(method_config.get(key, 0.0), 0.0, 1.0))
    return float(np.clip(method_config.get("local_evidence_phase_bootstrap_support", 0.0), 0.0, 1.0))


def _group_support_profile(
    coordinate_field: np.ndarray,
    *,
    group_ranges: list[Any] | None,
    group_factors: list[Any] | None,
    group_ratios: list[Any] | None,
    smoothing_scale: float,
    floor: float,
    ratio_support_threshold: float,
    ratio_support_scale: float,
    default_value: float = 0.0,
) -> np.ndarray:
    coord = np.asarray(coordinate_field, dtype=float).reshape(-1)
    ranges = list(group_ranges or [])
    factors = list(group_factors or [])
    ratios = list(group_ratios or [])
    if not ranges or not factors:
        baseline = max(float(default_value), float(np.clip(floor, 0.0, 1.0)))
        return np.full(coord.shape, baseline, dtype=float)

    profile = np.full(
        coord.shape,
        max(float(default_value), float(np.clip(floor, 0.0, 1.0))),
        dtype=float,
    )
    min_width = 1.0 / max(8, coord.size)
    for idx, (raw_range, raw_factor) in enumerate(zip(ranges, factors)):
        if raw_range is None or len(raw_range) != 2:
            continue
        left = float(raw_range[0])
        right = float(raw_range[1])
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        lo, hi = sorted((left, right))
        factor = float(np.clip(raw_factor, 0.0, 1.0))
        ratio = ratios[idx] if idx < len(ratios) else None
        positive_support = _positive_holdout_support(
            ratio,
            threshold=ratio_support_threshold,
            scale=ratio_support_scale,
        )
        local_strength = float(np.clip(factor * positive_support, 0.0, 1.0))
        if local_strength <= 0.0:
            continue
        width = max(hi - lo, min_width)
        smooth = max(width * max(float(smoothing_scale), 1e-6), min_width)
        distance = np.where(coord < lo, lo - coord, np.where(coord > hi, coord - hi, 0.0))
        local = local_strength * np.exp(-0.5 * (distance / smooth) ** 2)
        profile = np.maximum(profile, local)
    return np.clip(profile, 0.0, 1.0)


def _apply_local_evidence_window(
    *,
    model: GroundwaterModel,
    state: np.ndarray,
    indices: np.ndarray,
    noisy_targets: np.ndarray,
    adaptive_result: MCRResult,
    global_same_state_result: MCRResult,
    method_config: dict[str, Any],
) -> MCRResult:
    coordinate_field = _method_coordinate_field(model, method_config)
    ratio_support_threshold = float(method_config.get("local_evidence_ratio_support_threshold", 1.0))
    ratio_support_scale = float(method_config.get("local_evidence_ratio_support_scale", 0.08))
    anchor_window, anchor_bandwidth = _anchor_window_field(
        model,
        indices,
        bandwidth_cells=method_config.get("local_evidence_anchor_bandwidth_cells"),
        power=float(method_config.get("local_evidence_anchor_power", 1.0)),
    )
    group_profile = _group_support_profile(
        coordinate_field,
        group_ranges=adaptive_result.metadata.get("adaptive_stress_holdout_group_ranges"),
        group_factors=adaptive_result.metadata.get("adaptive_stress_holdout_group_factors"),
        group_ratios=adaptive_result.metadata.get("adaptive_stress_holdout_group_ratios"),
        smoothing_scale=float(method_config.get("local_evidence_group_smoothing_scale", 0.75)),
        floor=float(method_config.get("local_evidence_group_floor", 0.0)),
        ratio_support_threshold=ratio_support_threshold,
        ratio_support_scale=ratio_support_scale,
        default_value=0.0,
    )
    validation_factor = float(
        np.clip(
            np.nan_to_num(adaptive_result.metadata.get("adaptive_validation_factor", 0.0), nan=0.0),
            0.0,
            1.0,
        )
    )
    jackknife_ratio = adaptive_result.metadata.get("adaptive_anchor_jackknife_ratio_tail")
    if jackknife_ratio is None or not np.isfinite(float(np.nan_to_num(jackknife_ratio, nan=np.nan))):
        jackknife_ratio = adaptive_result.metadata.get("adaptive_anchor_jackknife_ratio")
    stress_ratio = adaptive_result.metadata.get("adaptive_stress_holdout_ratio_tail")
    if stress_ratio is None or not np.isfinite(float(np.nan_to_num(stress_ratio, nan=np.nan))):
        stress_ratio = adaptive_result.metadata.get("adaptive_stress_holdout_ratio")
    jackknife_support = _positive_holdout_support(
        jackknife_ratio,
        threshold=ratio_support_threshold,
        scale=ratio_support_scale,
    )
    stress_support = _positive_holdout_support(
        stress_ratio,
        threshold=ratio_support_threshold,
        scale=ratio_support_scale,
    )
    shared_holdout_support = min(jackknife_support, stress_support)
    phase_aware = bool(method_config.get("local_evidence_phase_aware", False))
    phase_support_default = (
        float(method_config.get("local_evidence_phase_bootstrap_support", 0.0))
        if phase_aware
        else 1.0
    )
    phase_support = float(
        np.clip(
            np.nan_to_num(
                adaptive_result.metadata.get("adaptive_phase_holdout_support", phase_support_default),
                nan=phase_support_default,
            ),
            0.0,
            1.0,
        )
    )
    temporal_factor = (
        validation_factor ** max(float(method_config.get("local_evidence_validation_power", 1.0)), 1e-6)
    ) * (
        shared_holdout_support ** max(float(method_config.get("local_evidence_temporal_power", 1.0)), 1e-6)
    ) * (
        phase_support ** max(float(method_config.get("local_evidence_phase_power", 1.0)), 1e-6)
    )
    local_window = np.clip(temporal_factor * anchor_window * group_profile, 0.0, 1.0)

    adaptive_scale = np.asarray(adaptive_result.scale_field, dtype=float).reshape(-1)
    global_scale = np.asarray(global_same_state_result.scale_field, dtype=float).reshape(-1)
    blended_scale = global_scale + local_window * (adaptive_scale - global_scale)
    flat_state = np.asarray(state, dtype=float).reshape(-1)
    blended_reconstructed = flat_state * blended_scale
    idx = np.asarray(indices, dtype=int)
    targets = np.asarray(noisy_targets, dtype=float)
    residual = blended_reconstructed[idx] - targets
    residual_rmse = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0

    blended_prediction_std = None
    if adaptive_result.prediction_std is not None and global_same_state_result.prediction_std is not None:
        adaptive_std = np.asarray(adaptive_result.prediction_std, dtype=float).reshape(-1)
        global_std = np.asarray(global_same_state_result.prediction_std, dtype=float).reshape(-1)
        blended_prediction_std = (
            global_std + local_window * (adaptive_std - global_std)
        ).reshape(model.ny, model.nx)

    metadata = dict(adaptive_result.metadata)
    metadata.update(
        {
            "adaptive_local_window_applied": True,
            "adaptive_local_window_temporal_factor": float(temporal_factor),
            "adaptive_local_window_peak": float(np.max(local_window)),
            "adaptive_local_window_mean": float(np.mean(local_window)),
            "adaptive_local_window_anchor_bandwidth_cells": float(anchor_bandwidth),
            "adaptive_local_window_anchor_peak": float(np.max(anchor_window)),
            "adaptive_local_window_group_peak": float(np.max(group_profile)),
            "adaptive_local_window_group_mean": float(np.mean(group_profile)),
            "adaptive_local_window_validation_factor": float(validation_factor),
            "adaptive_local_window_jackknife_support": float(jackknife_support),
            "adaptive_local_window_stress_support": float(stress_support),
            "adaptive_local_window_shared_holdout_support": float(shared_holdout_support),
            "adaptive_local_window_phase_support": float(phase_support),
        }
    )

    return MCRResult(
        reconstructed=blended_reconstructed.reshape(model.ny, model.nx),
        alpha=float(np.mean(blended_scale)),
        residual_rmse=residual_rmse,
        denominator=float(adaptive_result.denominator),
        n_constraints=int(adaptive_result.n_constraints),
        method=adaptive_result.method,
        scale_field=blended_scale.reshape(model.ny, model.nx),
        prediction_std=blended_prediction_std,
        coefficients=adaptive_result.coefficients,
        coefficient_std=adaptive_result.coefficient_std,
        posterior_trace=adaptive_result.posterior_trace,
        metadata=metadata,
    )


def _reset_result_to_backbone(
    target_result: MCRResult,
    backbone_result: MCRResult,
    *,
    metadata_updates: dict[str, Any] | None = None,
) -> None:
    target_result.reconstructed = np.asarray(backbone_result.reconstructed, dtype=float)
    target_result.alpha = float(backbone_result.alpha)
    target_result.residual_rmse = float(backbone_result.residual_rmse)
    target_result.denominator = float(backbone_result.denominator)
    target_result.scale_field = (
        None
        if backbone_result.scale_field is None
        else np.asarray(backbone_result.scale_field, dtype=float)
    )
    target_result.prediction_std = (
        None
        if backbone_result.prediction_std is None
        else np.asarray(backbone_result.prediction_std, dtype=float)
    )
    target_result.coefficients = (
        None
        if backbone_result.coefficients is None
        else np.asarray(backbone_result.coefficients, dtype=float)
    )
    target_result.coefficient_std = (
        None
        if backbone_result.coefficient_std is None
        else np.asarray(backbone_result.coefficient_std, dtype=float)
    )
    target_result.posterior_trace = backbone_result.posterior_trace
    if metadata_updates:
        target_result.metadata.update(metadata_updates)


def _resolve_mcr_methods(mcr_config: dict[str, Any], field_name: str) -> list[dict[str, Any]]:
    specific_key = f"{field_name}_methods"
    methods = mcr_config.get(specific_key)
    if methods is None:
        methods = mcr_config.get("methods")
    if methods is None:
        return [{"name": "global", "type": "global"}]
    return [dict(item) for item in methods]


def _transport_quantum_mode(config: dict[str, Any]) -> str:
    transport_quantum = config.get("transport_quantum", {})
    return str(transport_quantum.get("mode", "operator_stepwise_hybrid")).lower()


def _constraint_indices_and_values(
    model: GroundwaterModel,
    constraint_config: dict[str, Any],
    reference_field: np.ndarray,
    *,
    h_left: float,
    h_right: float,
    flow_boundary: dict[str, Any] | None,
    sources: list[SourceTerm],
    rng: np.random.Generator,
    vx: np.ndarray | None = None,
    vy: np.ndarray | None = None,
    plane_columns: dict[str, int] | None = None,
    flow_wells: list[WellTerm] | None = None,
    phase_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    constraint_type = str(constraint_config.get("type", "sources")).lower()

    if constraint_type == "dirichlet_boundary":
        indices, values = model.flow_boundary_indices_and_values(
            h_left,
            h_right,
            flow_boundary=flow_boundary,
        )
        return indices, values, {"strategy": "dirichlet_boundary", "truth_independent": True}

    placement = select_anchor_indices(
        model,
        constraint_config,
        reference_field=reference_field,
        sources=sources,
        rng=rng,
        vx=vx,
        vy=vy,
        plane_columns=plane_columns,
        flow_wells=flow_wells,
        phase_label=phase_label,
    )
    indices = np.asarray(placement.indices, dtype=int)
    values = np.asarray(reference_field, dtype=float).reshape(-1)[indices]
    metadata = dict(placement.metadata)
    metadata["truth_independent"] = constraint_type != "random_active"
    return indices, values, metadata


def _should_record(step_idx: int, steps: int, stride: int) -> bool:
    return step_idx == steps or step_idx % stride == 0


def _simulate_transport_reference(
    model: GroundwaterModel,
    *,
    flow_regimes: list[FlowRegime],
    steps: int,
    sources: list[SourceTerm],
    record_stride: int,
    plane_columns: dict[str, int],
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float]]]:
    initial_regime = flow_regimes[0]
    concentration = np.zeros(model.n_nodes, dtype=float)
    history: list[np.ndarray] = [concentration.reshape(model.ny, model.nx).copy()]
    trace: list[dict[str, float]] = [
        transport_trace_row(
            history[0],
            time=0.0,
            x_coords=model.x_coords,
            y_coords=model.y_coords,
            dx=model.dx,
            dy=model.dy,
            vx=initial_regime.vx,
            plane_columns=plane_columns,
        )
    ]

    stride = max(1, int(record_stride))
    regime_index = 0
    for step_idx in range(1, int(steps) + 1):
        while regime_index < len(flow_regimes) - 1 and step_idx > flow_regimes[regime_index].end_step:
            regime_index += 1
        regime = flow_regimes[regime_index]
        source_vec = model.source_vector(sources, step_idx=step_idx)
        concentration = regime.solve_transport_step(concentration + model.dt * source_vec)
        field = concentration.reshape(model.ny, model.nx).copy()
        history.append(field)
        if _should_record(step_idx, int(steps), stride):
            trace.append(
                transport_trace_row(
                    field,
                    time=step_idx * model.dt,
                    x_coords=model.x_coords,
                    y_coords=model.y_coords,
                    dx=model.dx,
                    dy=model.dy,
                    vx=regime.vx,
                    plane_columns=plane_columns,
                )
            )

    return history[-1], history, trace


def _result_row(
    *,
    field_name: str,
    constraint: dict[str, Any],
    method_config: dict[str, Any],
    anchor_metadata: dict[str, Any],
    result: MCRResult,
    reference_field: np.ndarray,
    model: GroundwaterModel,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    vx: np.ndarray | None = None,
    plane_columns: dict[str, int] | None = None,
    flow_wells: list[WellTerm] | None = None,
    state_result: QuantumStateResult | None = None,
) -> dict[str, Any]:
    method_name = str(method_config.get("name", method_config.get("type", "global")))
    method_type = str(method_config.get("type", "global"))
    row: dict[str, Any] = {
        "field": field_name,
        "constraint_name": str(constraint.get("name", constraint.get("type", "constraint"))),
        "constraint_type": str(constraint.get("type", "")),
        "anchor_strategy": str(anchor_metadata.get("strategy", constraint.get("type", ""))),
        "anchor_phase_label": str(anchor_metadata.get("phase_label", "")),
        "anchor_phase_constraint_type": str(anchor_metadata.get("phase_constraint_type", "")),
        "anchor_phase_observation_stride": anchor_metadata.get("phase_observation_stride", np.nan),
        "anchor_phase_budget": anchor_metadata.get("phase_anchor_budget", np.nan),
        "anchor_transport_budget_match_enabled": anchor_metadata.get(
            "transport_budget_match_enabled",
            np.nan,
        ),
        "anchor_transport_budget_reference_mode": anchor_metadata.get(
            "transport_budget_reference_mode",
            "",
        ),
        "anchor_transport_budget_reference_stride": anchor_metadata.get(
            "transport_budget_reference_stride",
            np.nan,
        ),
        "anchor_transport_budget_reference_observation_steps": anchor_metadata.get(
            "transport_budget_reference_observation_steps",
            np.nan,
        ),
        "anchor_transport_budget_reference_total_observations": anchor_metadata.get(
            "transport_budget_reference_total_observations",
            np.nan,
        ),
        "constraint_truth_independent": bool(anchor_metadata.get("truth_independent", True)),
        "n_constraints": result.n_constraints,
        "alpha": result.alpha,
        "mcr_method_name": method_name,
        "mcr_method_type": method_type,
        "mcr_residual_rmse": result.residual_rmse,
        "mcr_denominator": result.denominator,
        "mcr_posterior_trace": result.posterior_trace,
        "mcr_n_coefficients": np.nan if result.coefficients is None else int(result.coefficients.size),
        "mcr_scale_range": np.nan
        if result.coefficients is None
        else float(np.ptp(result.coefficients)),
        "mcr_scale_cv": np.nan
        if result.coefficients is None or abs(float(np.mean(result.coefficients))) < 1e-30
        else float(np.std(result.coefficients) / abs(float(np.mean(result.coefficients)))),
        "mcr_coefficient_std_mean": np.nan
        if result.coefficient_std is None
        else float(np.mean(result.coefficient_std)),
        "mcr_basis_type": str(result.metadata.get("basis_type", "")),
        "mcr_effective_regions": result.metadata.get("n_regions", np.nan),
        "mcr_requested_regions": result.metadata.get("requested_n_regions", np.nan),
        "mcr_basis_rank": result.metadata.get("basis_rank", np.nan),
        "mcr_raw_feature_count": result.metadata.get("raw_feature_count", np.nan),
        "mcr_retained_energy": result.metadata.get("retained_energy", np.nan),
        "mcr_hierarchical_levels": result.metadata.get("hierarchical_levels", np.nan),
        "mcr_low_rank_rank": result.metadata.get("low_rank_rank", np.nan),
        "mcr_low_rank_energy": result.metadata.get("low_rank_energy", np.nan),
        "mcr_low_rank_singular_power": result.metadata.get("low_rank_singular_power", np.nan),
        "mcr_low_rank_min_prior_scale": result.metadata.get("low_rank_min_prior_scale", np.nan),
        "mcr_prior_std_scale": result.metadata.get("prior_std_scale", np.nan),
        "mcr_min_constraints_per_region": result.metadata.get("min_constraints_per_region", np.nan),
        "mcr_shrinkage_weight": result.metadata.get("adaptive_shrinkage_weight", np.nan),
        "mcr_signal_variance": result.metadata.get("adaptive_signal_variance", np.nan),
        "mcr_noise_variance": result.metadata.get("adaptive_noise_variance", np.nan),
        "mcr_effective_noise_variance": result.metadata.get("adaptive_effective_noise_variance", np.nan),
        "mcr_validation_noise_variance": result.metadata.get("adaptive_validation_noise_variance", np.nan),
        "mcr_support_ratio": result.metadata.get("adaptive_support_ratio", np.nan),
        "mcr_stability_factor": result.metadata.get("adaptive_stability_factor", np.nan),
        "mcr_level_stability_factor": result.metadata.get("adaptive_level_stability_factor", np.nan),
        "mcr_validation_factor": result.metadata.get("adaptive_validation_factor", np.nan),
        "mcr_external_validation_factor": result.metadata.get("adaptive_external_validation_factor", np.nan),
        "mcr_hard_fallback_applied": result.metadata.get("adaptive_hard_fallback_applied", np.nan),
        "mcr_validation_hard_gate_threshold": result.metadata.get("adaptive_validation_hard_gate_threshold", np.nan),
        "mcr_local_window_applied": result.metadata.get("adaptive_local_window_applied", np.nan),
        "mcr_local_window_temporal_factor": result.metadata.get("adaptive_local_window_temporal_factor", np.nan),
        "mcr_local_window_peak": result.metadata.get("adaptive_local_window_peak", np.nan),
        "mcr_local_window_mean": result.metadata.get("adaptive_local_window_mean", np.nan),
        "mcr_local_window_anchor_bandwidth_cells": result.metadata.get("adaptive_local_window_anchor_bandwidth_cells", np.nan),
        "mcr_local_window_anchor_peak": result.metadata.get("adaptive_local_window_anchor_peak", np.nan),
        "mcr_local_window_group_peak": result.metadata.get("adaptive_local_window_group_peak", np.nan),
        "mcr_local_window_group_mean": result.metadata.get("adaptive_local_window_group_mean", np.nan),
        "mcr_local_window_validation_factor": result.metadata.get("adaptive_local_window_validation_factor", np.nan),
        "mcr_local_window_jackknife_support": result.metadata.get("adaptive_local_window_jackknife_support", np.nan),
        "mcr_local_window_stress_support": result.metadata.get("adaptive_local_window_stress_support", np.nan),
        "mcr_local_window_shared_holdout_support": result.metadata.get("adaptive_local_window_shared_holdout_support", np.nan),
        "mcr_local_window_phase_support": result.metadata.get("adaptive_local_window_phase_support", np.nan),
        "mcr_local_window_backbone_fallback": result.metadata.get("adaptive_local_window_backbone_fallback", np.nan),
        "mcr_schedule_phase_index": result.metadata.get("adaptive_schedule_phase_index", np.nan),
        "mcr_schedule_phase_label": result.metadata.get("adaptive_schedule_phase_label", ""),
        "mcr_schedule_phase_progress": result.metadata.get("adaptive_schedule_phase_progress", np.nan),
        "mcr_schedule_phase_start_step": result.metadata.get("adaptive_schedule_phase_start_step", np.nan),
        "mcr_schedule_phase_end_step": result.metadata.get("adaptive_schedule_phase_end_step", np.nan),
        "mcr_schedule_phase_transition_reset": result.metadata.get("adaptive_schedule_phase_transition_reset", np.nan),
        "mcr_phase_holdout_support": result.metadata.get("adaptive_phase_holdout_support", np.nan),
        "mcr_phase_holdout_bootstrap_support": result.metadata.get(
            "adaptive_phase_holdout_bootstrap_support",
            np.nan,
        ),
        "mcr_phase_holdout_instant_support": result.metadata.get(
            "adaptive_phase_holdout_instant_support",
            np.nan,
        ),
        "mcr_phase_holdout_ratio": result.metadata.get("adaptive_phase_holdout_ratio", np.nan),
        "mcr_phase_holdout_adaptive_forecast_rmse": result.metadata.get(
            "adaptive_phase_holdout_adaptive_forecast_rmse",
            np.nan,
        ),
        "mcr_phase_holdout_global_forecast_rmse": result.metadata.get(
            "adaptive_phase_holdout_global_forecast_rmse",
            np.nan,
        ),
        "mcr_phase_holdout_first_observation": result.metadata.get(
            "adaptive_phase_holdout_first_observation",
            np.nan,
        ),
        "mcr_anchor_jackknife_factor": result.metadata.get("adaptive_anchor_jackknife_factor", np.nan),
        "mcr_anchor_jackknife_ratio": result.metadata.get("adaptive_anchor_jackknife_ratio", np.nan),
        "mcr_anchor_jackknife_ratio_tail": result.metadata.get("adaptive_anchor_jackknife_ratio_tail", np.nan),
        "mcr_anchor_jackknife_regional_rmse": result.metadata.get("adaptive_anchor_jackknife_regional_rmse", np.nan),
        "mcr_anchor_jackknife_global_rmse": result.metadata.get("adaptive_anchor_jackknife_global_rmse", np.nan),
        "mcr_anchor_jackknife_excursion_ratio": result.metadata.get("adaptive_anchor_jackknife_excursion_ratio", np.nan),
        "mcr_anchor_jackknife_excursion_ratio_tail": result.metadata.get("adaptive_anchor_jackknife_excursion_ratio_tail", np.nan),
        "mcr_anchor_jackknife_excursion_factor": result.metadata.get("adaptive_anchor_jackknife_excursion_factor", np.nan),
        "mcr_anchor_jackknife_scale_field_variance": result.metadata.get("adaptive_anchor_jackknife_scale_field_variance", np.nan),
        "mcr_anchor_jackknife_folds_used": result.metadata.get("adaptive_anchor_jackknife_folds_used", np.nan),
        "mcr_stress_holdout_factor": result.metadata.get("adaptive_stress_holdout_factor", np.nan),
        "mcr_stress_holdout_ratio": result.metadata.get("adaptive_stress_holdout_ratio", np.nan),
        "mcr_stress_holdout_ratio_tail": result.metadata.get("adaptive_stress_holdout_ratio_tail", np.nan),
        "mcr_stress_holdout_regional_rmse": result.metadata.get("adaptive_stress_holdout_regional_rmse", np.nan),
        "mcr_stress_holdout_global_rmse": result.metadata.get("adaptive_stress_holdout_global_rmse", np.nan),
        "mcr_stress_holdout_excursion_ratio": result.metadata.get("adaptive_stress_holdout_excursion_ratio", np.nan),
        "mcr_stress_holdout_excursion_ratio_tail": result.metadata.get("adaptive_stress_holdout_excursion_ratio_tail", np.nan),
        "mcr_stress_holdout_excursion_factor": result.metadata.get("adaptive_stress_holdout_excursion_factor", np.nan),
        "mcr_stress_holdout_scale_field_variance": result.metadata.get("adaptive_stress_holdout_scale_field_variance", np.nan),
        "mcr_stress_holdout_folds_used": result.metadata.get("adaptive_stress_holdout_folds_used", np.nan),
        "mcr_temporal_holdout_factor": result.metadata.get("adaptive_temporal_holdout_factor", np.nan),
        "mcr_temporal_holdout_ratio": result.metadata.get("adaptive_temporal_holdout_ratio", np.nan),
        "mcr_temporal_holdout_adaptive_forecast_rmse": result.metadata.get("adaptive_temporal_holdout_adaptive_forecast_rmse", np.nan),
        "mcr_temporal_holdout_global_forecast_rmse": result.metadata.get("adaptive_temporal_holdout_global_forecast_rmse", np.nan),
        "mcr_temporal_holdout_transition": result.metadata.get("adaptive_temporal_holdout_transition", np.nan),
        "mcr_scale_cv_raw": result.metadata.get("adaptive_scale_cv_raw", np.nan),
        "mcr_scale_level_ratio_raw": result.metadata.get("adaptive_scale_level_ratio_raw", np.nan),
    }

    if state_result is not None:
        row["state_true_norm"] = state_result.true_norm
        row["state_label"] = state_result.metadata.get("label", "")

    if field_name == "concentration":
        row.update(
            concentration_metrics(
                field_name,
                result.reconstructed,
                reference_field,
                dx=model.dx,
                dy=model.dy,
                prediction_std=result.prediction_std,
                x_coords=x_coords,
                y_coords=y_coords,
                vx=vx,
                plane_columns=plane_columns,
                flow_wells=flow_wells,
            )
        )
    else:
        row.update(
            field_metrics(
                field_name,
                result.reconstructed,
                reference_field,
                prediction_std=result.prediction_std,
            )
        )

    return row


def _save_result_fields(
    fields: dict[str, np.ndarray],
    *,
    field_name: str,
    constraint: dict[str, Any],
    method_config: dict[str, Any],
    result: MCRResult,
) -> str:
    name = str(constraint.get("name", constraint.get("type", "constraint")))
    method_name = str(method_config.get("name", method_config.get("type", "global")))
    key = f"{field_name}_{name}_{method_name}".replace(" ", "_")
    fields[key] = np.asarray(result.reconstructed, dtype=float)
    if result.scale_field is not None:
        fields[f"{key}_scale_field"] = np.asarray(result.scale_field, dtype=float)
    if result.prediction_std is not None:
        fields[f"{key}_prediction_std"] = np.asarray(result.prediction_std, dtype=float)
    return key


def _run_final_state_mcr_family(
    *,
    field_name: str,
    state_result: QuantumStateResult,
    reference_field: np.ndarray,
    model: GroundwaterModel,
    constraint_sets: list[dict[str, Any]],
    method_configs: list[dict[str, Any]],
    h_left: float,
    h_right: float,
    flow_boundary: dict[str, Any] | None,
    sources: list[SourceTerm],
    observation_noise: dict[str, Any],
    rng: np.random.Generator,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    anchor_vx: np.ndarray | None = None,
    anchor_vy: np.ndarray | None = None,
    metric_vx: np.ndarray | None = None,
    plane_columns: dict[str, int] | None = None,
    flow_wells: list[WellTerm] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    fields: dict[str, np.ndarray] = {}

    for constraint in constraint_sets:
        constraint_name = str(constraint.get("name", constraint.get("type", "constraint")))
        indices, targets, anchor_metadata = _constraint_indices_and_values(
            model,
            constraint,
            reference_field,
            h_left=h_left,
            h_right=h_right,
            flow_boundary=flow_boundary,
            sources=sources,
            rng=rng,
            vx=anchor_vx,
            vy=anchor_vy,
            plane_columns=plane_columns,
            flow_wells=flow_wells,
        )
        noisy_targets = _apply_observation_noise(
            targets,
            observation_noise,
            rng,
            noise_key=f"{field_name}|{constraint_name}|final_observation",
        )
        observation_std = _observation_scale(targets, observation_noise)

        for method_config in method_configs:
            result = apply_mcr_method(
                state_result.state,
                indices,
                noisy_targets,
                output_shape=reference_field.shape,
                method_config=method_config,
                observation_std=observation_std,
            )
            _save_result_fields(
                fields,
                field_name=field_name,
                constraint=constraint,
                method_config=method_config,
                result=result,
            )
            rows.append(
                _result_row(
                    field_name=field_name,
                    constraint=constraint,
                    method_config=method_config,
                    anchor_metadata=anchor_metadata,
                    result=result,
                    reference_field=reference_field,
                    model=model,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    vx=metric_vx if metric_vx is not None else anchor_vx,
                    plane_columns=plane_columns,
                    flow_wells=flow_wells,
                    state_result=state_result,
                )
            )

    return rows, fields


def _run_operator_transport_family(
    *,
    solver: Any,
    model: GroundwaterModel,
    reference_history: list[np.ndarray],
    reference_trace_summary: dict[str, float],
    flow_regimes: list[FlowRegime],
    steps: int,
    sources: list[SourceTerm],
    anchor_vx: np.ndarray,
    anchor_vy: np.ndarray,
    metric_vx: np.ndarray,
    record_stride: int,
    plane_columns: dict[str, int],
    constraint_sets: list[dict[str, Any]],
    method_configs: list[dict[str, Any]],
    h_left: float,
    h_right: float,
    flow_boundary: dict[str, Any] | None,
    observation_noise: dict[str, Any],
    rng: np.random.Generator,
    transport_quantum_config: dict[str, Any],
    flow_wells: list[WellTerm] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], QuantumStateResult | None]:
    rows: list[dict[str, Any]] = []
    fields: dict[str, np.ndarray] = {}
    observation_stride = max(1, int(transport_quantum_config.get("observation_stride", 1)))
    store_trace_metrics = bool(transport_quantum_config.get("store_trace_metrics", True))
    representative_state: QuantumStateResult | None = None
    initial_regime = flow_regimes[0]
    schedule_phases, step_to_phase = _build_schedule_phases(
        flow_regimes=flow_regimes,
        steps=steps,
        sources=sources,
        flow_wells=flow_wells or [],
    )

    for constraint in constraint_sets:
        constraint_name = str(constraint.get("name", constraint.get("type", "constraint")))
        shared_observations: dict[int, dict[str, Any]] = {}
        observation_plan = _build_observation_plan(
            steps=steps,
            schedule_phases=schedule_phases,
            step_to_phase=step_to_phase,
            transport_quantum_config=transport_quantum_config,
        )
        step_anchor_budgets, observation_budget_metadata = _resolve_step_anchor_budgets(
            constraint_config=constraint,
            observation_plan=observation_plan,
            transport_quantum_config=transport_quantum_config,
            steps=steps,
            schedule_phases=schedule_phases,
            step_to_phase=step_to_phase,
        )
        for observation_item in observation_plan:
            step_idx = int(observation_item["step_idx"])
            phase_info = observation_item["phase_info"]
            phase_stride = int(observation_item["phase_stride"])
            reference_field = reference_history[step_idx]
            phase_constraint = _phase_constraint_config(
                constraint,
                transport_quantum_config=transport_quantum_config,
                phase_label=str(phase_info.label),
            )
            if step_idx in step_anchor_budgets:
                phase_constraint = dict(phase_constraint)
                phase_constraint["m"] = int(step_anchor_budgets[step_idx])
            anchor_regime = flow_regimes[max(0, min(int(phase_info.regime_index), len(flow_regimes) - 1))]
            indices, targets, anchor_metadata = _constraint_indices_and_values(
                model,
                phase_constraint,
                reference_field,
                h_left=h_left,
                h_right=h_right,
                flow_boundary=flow_boundary,
                sources=sources,
                rng=rng,
                vx=anchor_regime.vx,
                vy=anchor_regime.vy,
                plane_columns=plane_columns,
                flow_wells=anchor_regime.wells,
                phase_label=str(phase_info.label),
            )
            anchor_metadata = dict(anchor_metadata)
            anchor_metadata["phase_label"] = str(phase_info.label)
            anchor_metadata["phase_constraint_type"] = str(phase_constraint.get("type", constraint.get("type", "")))
            anchor_metadata["phase_observation_stride"] = int(phase_stride)
            anchor_metadata["phase_anchor_budget"] = int(step_anchor_budgets.get(step_idx, int(indices.size)))
            anchor_metadata["transport_budget_match_enabled"] = bool(observation_budget_metadata.get("enabled", False))
            anchor_metadata["transport_budget_reference_mode"] = str(
                observation_budget_metadata.get("reference_mode", "")
            )
            anchor_metadata["transport_budget_reference_stride"] = observation_budget_metadata.get(
                "reference_stride",
                np.nan,
            )
            anchor_metadata["transport_budget_reference_observation_steps"] = observation_budget_metadata.get(
                "reference_observation_steps",
                np.nan,
            )
            anchor_metadata["transport_budget_reference_total_observations"] = observation_budget_metadata.get(
                "reference_total_observations",
                np.nan,
            )
            shared_observations[step_idx] = {
                "reference_field": reference_field,
                "indices": np.asarray(indices, dtype=int),
                "targets": targets,
                "noisy_targets": _apply_observation_noise(
                    targets,
                    observation_noise,
                    rng,
                    noise_key=f"transport_operator|{constraint_name}|step={step_idx}|observation",
                ),
                "observation_std": _observation_scale(targets, observation_noise),
                "anchor_metadata": anchor_metadata,
            }
        actual_observation_summary = _summarize_shared_observations(shared_observations)

        for method_config in method_configs:
            method_type = str(method_config.get("type", "global")).lower()
            local_window_enabled = method_type in {
                "regional_adaptive_eb",
                "adaptive_regional",
                "regional_shrinkage_eb",
            } and bool(method_config.get("local_evidence_window", False))
            phase_aware_local_window = local_window_enabled and bool(
                method_config.get("local_evidence_phase_aware", False)
            )
            use_temporal_holdout = method_type in {
                "regional_adaptive_eb",
                "adaptive_regional",
                "regional_shrinkage_eb",
            } and method_config.get("stress_period_temporal_holdout_threshold") is not None
            track_global_backbone = method_type in {
                "regional_adaptive_eb",
                "adaptive_regional",
                "regional_shrinkage_eb",
            } and (
                use_temporal_holdout
                or local_window_enabled
                or method_config.get("validation_hard_gate_threshold") is not None
            )
            estimated = np.zeros(model.n_nodes, dtype=float)
            estimated_trace: list[dict[str, float]] = [
                transport_trace_row(
                    estimated.reshape(model.ny, model.nx),
                    time=0.0,
                    x_coords=model.x_coords,
                    y_coords=model.y_coords,
                    dx=model.dx,
                    dy=model.dy,
                    vx=initial_regime.vx,
                    plane_columns=plane_columns,
                )
            ]
            last_state_result: QuantumStateResult | None = None
            last_result: MCRResult | None = None
            last_scale_field = np.ones(model.n_nodes, dtype=float)
            estimated_global_backbone = np.zeros(model.n_nodes, dtype=float)
            global_backbone_scale = np.ones(model.n_nodes, dtype=float)
            last_observed_regime_index: int | None = None
            n_observations = 0
            regime_index = 0
            hard_fallback_latched = False
            fallback_latch_persistent = (not local_window_enabled) or bool(
                method_config.get("local_evidence_latch_fallback", False)
            )
            current_phase_index: int | None = None
            current_phase_observations = 0
            phase_transition_reset_pending = False
            current_phase_support_state = 0.0
            latest_anchor_metadata: dict[str, Any] = {
                "strategy": str(constraint.get("type", "constraint")),
                "truth_independent": True,
            }

            for step_idx in range(1, int(steps) + 1):
                while regime_index < len(flow_regimes) - 1 and step_idx > flow_regimes[regime_index].end_step:
                    regime_index += 1
                regime = flow_regimes[regime_index]
                phase_index = int(
                    step_to_phase[step_idx]
                    if 0 <= step_idx < step_to_phase.size
                    else schedule_phases[-1].phase_index
                )
                phase_info = schedule_phases[max(0, min(phase_index, len(schedule_phases) - 1))]
                phase_length = max(1, int(phase_info.end_step) - int(phase_info.start_step) + 1)
                phase_progress = (
                    1.0
                    if phase_length <= 1
                    else float(step_idx - int(phase_info.start_step)) / float(phase_length - 1)
                )
                phase_transition_reset = False
                if current_phase_index is None:
                    current_phase_index = phase_index
                elif phase_index != current_phase_index:
                    if (
                        phase_aware_local_window
                        and track_global_backbone
                        and bool(method_config.get("local_evidence_phase_transition_reset", True))
                    ):
                        estimated = estimated_global_backbone.copy()
                        last_scale_field = global_backbone_scale.copy()
                        hard_fallback_latched = False
                        phase_transition_reset = True
                        phase_transition_reset_pending = True
                    current_phase_index = phase_index
                    current_phase_observations = 0
                    current_phase_support_state = 0.0
                source_vec = model.source_vector(sources, step_idx=step_idx)
                rhs_est = estimated + model.dt * source_vec
                solved = regime.solve_transport_step(rhs_est)
                state_result = solver.normalize_vector(
                    solved,
                    label=(
                        f"transport_operator_{constraint_name}_"
                        f"{method_config.get('name', method_config.get('type', 'global'))}_step{step_idx}"
                    ),
                    noise_key=f"transport_operator|{constraint_name}|step={step_idx}|primary_state",
                )
                global_state_result: QuantumStateResult | None = None
                if track_global_backbone:
                    rhs_global = estimated_global_backbone + model.dt * source_vec
                    solved_global = regime.solve_transport_step(rhs_global)
                    global_state_result = solver.normalize_vector(
                        solved_global,
                        label=(
                            f"transport_operator_{constraint_name}_"
                            f"{method_config.get('name', method_config.get('type', 'global'))}_globalbackbone_step{step_idx}"
                        ),
                        # Use the same keyed quantum-noise realization as the standalone
                        # method trajectory for this constraint/step so hard fallback
                        # truly collapses onto the reproducible global backbone.
                        noise_key=f"transport_operator|{constraint_name}|step={step_idx}|primary_state",
                    )

                should_observe = step_idx in shared_observations
                if should_observe:
                    observation_record = shared_observations[step_idx]
                    reference_field = np.asarray(observation_record["reference_field"], dtype=float)
                    indices = np.asarray(observation_record["indices"], dtype=int)
                    targets = np.asarray(observation_record["targets"], dtype=float)
                    noisy_targets = np.asarray(observation_record["noisy_targets"], dtype=float)
                    observation_std = np.asarray(observation_record["observation_std"], dtype=float)
                    latest_anchor_metadata = dict(observation_record.get("anchor_metadata", latest_anchor_metadata))
                    runtime_method_config = dict(method_config)
                    requested_hard_gate_threshold = method_config.get("validation_hard_gate_threshold")
                    if local_window_enabled:
                        runtime_method_config["validation_hard_gate_threshold"] = None
                    temporal_holdout_factor = 1.0
                    temporal_holdout_ratio = np.nan
                    temporal_adaptive_forecast_rmse = np.nan
                    temporal_global_forecast_rmse = np.nan
                    temporal_transition = False
                    phase_holdout_ratio = np.nan
                    phase_adaptive_forecast_rmse = np.nan
                    phase_global_forecast_rmse = np.nan
                    phase_first_observation = phase_aware_local_window and current_phase_observations == 0
                    phase_bootstrap_support = (
                        _phase_bootstrap_support(
                            method_config,
                            phase_label=str(phase_info.label),
                        )
                        if phase_aware_local_window
                        else 0.0
                    )
                    phase_instant_support = np.nan
                    phase_holdout_support = (
                        phase_bootstrap_support
                        if phase_aware_local_window
                        else 1.0
                    )
                    obs_weights = _observation_scale(targets, observation_noise)
                    if obs_weights is None:
                        local_weights = np.ones_like(noisy_targets, dtype=float)
                    else:
                        local_weights = 1.0 / np.maximum(np.asarray(obs_weights, dtype=float), 1e-12) ** 2

                    if (use_temporal_holdout or phase_aware_local_window) and global_state_result is not None:
                        adaptive_forecast = state_result.state[indices] * last_scale_field[indices]
                        global_forecast = global_state_result.state[indices] * global_backbone_scale[indices]
                        forecast_adaptive_rmse = float(
                            np.sqrt(
                                np.sum(local_weights * (adaptive_forecast - noisy_targets) ** 2)
                                / max(np.sum(local_weights), 1e-30)
                            )
                        )
                        forecast_global_rmse = float(
                            np.sqrt(
                                np.sum(local_weights * (global_forecast - noisy_targets) ** 2)
                                / max(np.sum(local_weights), 1e-30)
                            )
                        )
                        ratio_floor = max(float(np.sqrt(np.mean(noisy_targets**2))), 1.0) * 1e-9
                        if (
                            forecast_global_rmse <= ratio_floor
                            and forecast_adaptive_rmse <= ratio_floor
                        ):
                            forecast_ratio = 1.0
                        else:
                            forecast_ratio = forecast_adaptive_rmse / max(
                                forecast_global_rmse,
                                ratio_floor,
                            )
                        if use_temporal_holdout:
                            temporal_adaptive_forecast_rmse = float(forecast_adaptive_rmse)
                            temporal_global_forecast_rmse = float(forecast_global_rmse)
                            temporal_holdout_ratio = float(forecast_ratio)
                            temporal_transition = (
                                last_observed_regime_index is not None
                                and regime_index != last_observed_regime_index
                            )
                            apply_temporal_gate = (
                                last_observed_regime_index is not None
                                and (
                                    not bool(method_config.get("stress_period_temporal_holdout_transition_only", False))
                                    or temporal_transition
                                )
                            )
                            if apply_temporal_gate:
                                threshold = float(method_config.get("stress_period_temporal_holdout_threshold", 1.05))
                                scale = float(method_config.get("stress_period_temporal_holdout_scale", 0.2))
                                excess = max(float(temporal_holdout_ratio) - threshold, 0.0)
                                temporal_holdout_factor = 1.0 / (1.0 + excess / max(scale, 1e-9))
                            runtime_method_config["external_validation_factor"] = float(
                                np.clip(temporal_holdout_factor, 0.0, 1.0)
                            )
                        if phase_aware_local_window:
                            phase_adaptive_forecast_rmse = float(forecast_adaptive_rmse)
                            phase_global_forecast_rmse = float(forecast_global_rmse)
                            phase_holdout_ratio = float(forecast_ratio)
                            if phase_first_observation:
                                current_phase_support_state = float(np.clip(phase_bootstrap_support, 0.0, 1.0))
                            else:
                                phase_instant_support = _positive_holdout_support(
                                    phase_holdout_ratio,
                                    threshold=float(
                                        method_config.get("local_evidence_phase_ratio_support_threshold", 1.0)
                                    ),
                                    scale=float(
                                        method_config.get("local_evidence_phase_ratio_support_scale", 0.05)
                                    ),
                                )
                                decay = float(
                                    np.clip(
                                        method_config.get("local_evidence_phase_support_decay", 0.6),
                                        0.0,
                                        1.0,
                                    )
                                )
                                current_phase_support_state = max(
                                    float(np.clip(phase_instant_support, 0.0, 1.0)),
                                    decay * float(np.clip(current_phase_support_state, 0.0, 1.0)),
                                )
                            phase_holdout_support = float(
                                np.clip(current_phase_support_state, 0.0, 1.0)
                            )

                    last_result = apply_mcr_method(
                        state_result.state,
                        indices,
                        noisy_targets,
                        output_shape=(model.ny, model.nx),
                        method_config=runtime_method_config,
                        observation_std=observation_std,
                    )
                    if use_temporal_holdout:
                        last_result.metadata["adaptive_temporal_holdout_factor"] = float(temporal_holdout_factor)
                        last_result.metadata["adaptive_temporal_holdout_ratio"] = float(temporal_holdout_ratio)
                        last_result.metadata["adaptive_temporal_holdout_adaptive_forecast_rmse"] = float(
                            temporal_adaptive_forecast_rmse
                        )
                        last_result.metadata["adaptive_temporal_holdout_global_forecast_rmse"] = float(
                            temporal_global_forecast_rmse
                        )
                        last_result.metadata["adaptive_temporal_holdout_transition"] = bool(temporal_transition)
                    if local_window_enabled:
                        last_result.metadata["adaptive_schedule_phase_index"] = int(phase_index)
                        last_result.metadata["adaptive_schedule_phase_label"] = str(phase_info.label)
                        last_result.metadata["adaptive_schedule_phase_progress"] = float(phase_progress)
                        last_result.metadata["adaptive_schedule_phase_start_step"] = int(phase_info.start_step)
                        last_result.metadata["adaptive_schedule_phase_end_step"] = int(phase_info.end_step)
                        last_result.metadata["adaptive_schedule_phase_transition_reset"] = bool(
                            phase_transition_reset_pending or phase_transition_reset
                        )
                    if phase_aware_local_window:
                        last_result.metadata["adaptive_phase_holdout_support"] = float(
                            np.clip(phase_holdout_support, 0.0, 1.0)
                        )
                        last_result.metadata["adaptive_phase_holdout_bootstrap_support"] = float(
                            np.clip(phase_bootstrap_support, 0.0, 1.0)
                        )
                        last_result.metadata["adaptive_phase_holdout_instant_support"] = float(
                            np.nan_to_num(phase_instant_support, nan=np.nan)
                        )
                        last_result.metadata["adaptive_phase_holdout_ratio"] = float(phase_holdout_ratio)
                        last_result.metadata["adaptive_phase_holdout_adaptive_forecast_rmse"] = float(
                            phase_adaptive_forecast_rmse
                        )
                        last_result.metadata["adaptive_phase_holdout_global_forecast_rmse"] = float(
                            phase_global_forecast_rmse
                        )
                        last_result.metadata["adaptive_phase_holdout_first_observation"] = bool(
                            phase_first_observation
                        )
                    global_same_state_result: MCRResult | None = None
                    if local_window_enabled:
                        global_same_state_result = apply_mcr_method(
                            state_result.state,
                            indices,
                            noisy_targets,
                            output_shape=(model.ny, model.nx),
                            method_config={"name": "local_window_global_same_state", "type": "global"},
                            observation_std=observation_std,
                        )
                        last_result = _apply_local_evidence_window(
                            model=model,
                            state=state_result.state,
                            indices=indices,
                            noisy_targets=noisy_targets,
                            adaptive_result=last_result,
                            global_same_state_result=global_same_state_result,
                            method_config=method_config,
                        )
                    global_backbone_result: MCRResult | None = None
                    if track_global_backbone and global_state_result is not None:
                        global_backbone_result = apply_mcr_method(
                            global_state_result.state,
                            indices,
                            noisy_targets,
                            output_shape=(model.ny, model.nx),
                            method_config={"name": "temporal_holdout_global", "type": "global"},
                            observation_std=observation_std,
                        )
                        estimated_global_backbone = global_backbone_result.reconstructed.reshape(-1)
                        if global_backbone_result.scale_field is not None:
                            global_backbone_scale = np.asarray(global_backbone_result.scale_field, dtype=float).reshape(-1)
                        else:
                            global_backbone_scale = np.full(
                                model.n_nodes,
                                float(global_backbone_result.alpha),
                                dtype=float,
                            )
                        last_observed_regime_index = regime_index
                    fallback_this_step = bool(last_result.metadata.get("adaptive_hard_fallback_applied", False))
                    if local_window_enabled and requested_hard_gate_threshold is not None:
                        validation_factor = float(last_result.metadata.get("adaptive_validation_factor", 1.0))
                        local_peak = float(last_result.metadata.get("adaptive_local_window_peak", 0.0))
                        min_peak = float(method_config.get("local_evidence_min_peak", 0.05))
                        fallback_this_step = (
                            validation_factor <= float(requested_hard_gate_threshold)
                            and local_peak < min_peak
                        )
                        if fallback_latch_persistent:
                            hard_fallback_latched = hard_fallback_latched or fallback_this_step
                        else:
                            hard_fallback_latched = fallback_this_step
                    elif fallback_this_step:
                        hard_fallback_latched = True
                    if (hard_fallback_latched or fallback_this_step) and global_backbone_result is not None:
                        _reset_result_to_backbone(
                            last_result,
                            global_backbone_result,
                            metadata_updates={
                                "adaptive_hard_fallback_reset_to_backbone": True,
                                "adaptive_hard_fallback_latched": bool(hard_fallback_latched),
                                "adaptive_local_window_backbone_fallback": True,
                            },
                        )
                    elif local_window_enabled and not fallback_latch_persistent:
                        hard_fallback_latched = False
                    estimated = last_result.reconstructed.reshape(-1)
                    if last_result.scale_field is not None:
                        last_scale_field = np.asarray(last_result.scale_field, dtype=float).reshape(-1)
                    else:
                        last_scale_field = np.full(model.n_nodes, float(last_result.alpha), dtype=float)
                    current_phase_observations += 1
                    phase_transition_reset_pending = False
                    n_observations += int(indices.size)
                else:
                    if track_global_backbone and global_state_result is not None:
                        estimated_global_backbone = global_state_result.state * global_backbone_scale
                    if hard_fallback_latched and track_global_backbone:
                        estimated = estimated_global_backbone.copy()
                    else:
                        estimated = state_result.state * last_scale_field

                last_state_result = state_result

                if _should_record(step_idx, int(steps), int(record_stride)):
                    estimated_trace.append(
                        transport_trace_row(
                            estimated.reshape(model.ny, model.nx),
                            time=step_idx * model.dt,
                            x_coords=model.x_coords,
                            y_coords=model.y_coords,
                            dx=model.dx,
                            dy=model.dy,
                            vx=regime.vx,
                            plane_columns=plane_columns,
                        )
                    )

            if last_result is None or last_state_result is None:
                raise RuntimeError("Transport operator surrogate did not produce a final MCR result")

            representative_state = representative_state or last_state_result
            row = _result_row(
                field_name="concentration",
                constraint=constraint,
                method_config=method_config,
                anchor_metadata=latest_anchor_metadata,
                result=last_result,
                reference_field=reference_history[-1],
                model=model,
                x_coords=model.x_coords,
                y_coords=model.y_coords,
                vx=metric_vx,
                plane_columns=plane_columns,
                flow_wells=flow_wells,
                state_result=last_state_result,
            )
            row["transport_assimilation_mode"] = "operator_stepwise_hybrid"
            row["transport_observation_stride"] = observation_stride
            row["transport_total_observations"] = n_observations
            row["transport_budget_match_enabled"] = bool(observation_budget_metadata.get("enabled", False))
            row["transport_budget_reference_mode"] = str(observation_budget_metadata.get("reference_mode", ""))
            row["transport_budget_reference_stride"] = observation_budget_metadata.get("reference_stride", np.nan)
            row["transport_budget_reference_observation_steps"] = observation_budget_metadata.get(
                "reference_observation_steps",
                np.nan,
            )
            row["transport_budget_reference_total_observations"] = observation_budget_metadata.get(
                "reference_total_observations",
                np.nan,
            )
            injection_summary = actual_observation_summary.get("injection", {})
            pumpback_summary = actual_observation_summary.get("pumpback", {})
            row["transport_budget_injection_mean_anchors"] = injection_summary.get("mean_budget", np.nan)
            row["transport_budget_injection_observation_steps"] = injection_summary.get(
                "n_observation_steps",
                np.nan,
            )
            row["transport_budget_pumpback_mean_anchors"] = pumpback_summary.get("mean_budget", np.nan)
            row["transport_budget_pumpback_observation_steps"] = pumpback_summary.get(
                "n_observation_steps",
                np.nan,
            )

            if store_trace_metrics:
                row.update(
                    trace_summary_error_metrics(
                        "estimated_trace",
                        estimated_trace,
                        reference_trace_summary,
                        plane_names=list(plane_columns.keys()),
                    )
                )

            _save_result_fields(
                fields,
                field_name="concentration",
                constraint=constraint,
                method_config=method_config,
                result=last_result,
            )
            rows.append(row)

    return rows, fields, representative_state


def run_experiment(config: dict[str, Any], *, output_root: str | Path | None = None) -> Path:
    name = str(config.get("name", "quantum_hydro_experiment"))
    root = output_root or config.get("output", {}).get("root", "outputs")
    out_dir = ensure_output_dir(root, name)

    domain = config["domain"]
    physics = config["physics"]
    sources = _parse_sources(physics["sources"])
    flow_wells = _parse_flow_wells(physics.get("flow_wells", []))
    flow_boundary = physics.get("flow_boundary")
    k_field = generate_k_field(
        int(domain["nx"]),
        int(domain["ny"]),
        **config.get("k_field", {}),
    )

    model = GroundwaterModel(
        nx=int(domain["nx"]),
        ny=int(domain["ny"]),
        lx=float(domain["lx"]),
        ly=float(domain["ly"]),
        k_field=k_field,
        dispersion=float(physics["dispersion"]),
        dt=float(physics["dt"]),
    )

    h_left = float(physics["h_left"])
    h_right = float(physics["h_right"])
    steps = int(physics["steps"])
    diagnostics_config = config.get("transport_diagnostics", {})
    control_plane_fractions = diagnostics_config.get("control_plane_fractions", [0.25, 0.5, 0.75])
    record_stride = int(diagnostics_config.get("record_stride", max(1, steps // 50)))
    plane_columns = model.control_plane_columns(control_plane_fractions)
    resolved_flow_boundary = model.resolve_flow_boundary(h_left, h_right, flow_boundary=flow_boundary)
    flow_regimes = _build_flow_regimes(
        model,
        h_left=h_left,
        h_right=h_right,
        steps=steps,
        flow_wells=flow_wells,
        flow_boundary=flow_boundary,
    )
    initial_flow_regime = flow_regimes[0]
    final_flow_regime = flow_regimes[-1]
    flow_matrix = final_flow_regime.flow_matrix
    flow_rhs = final_flow_regime.flow_rhs
    head_reference = final_flow_regime.head
    anchor_vx = initial_flow_regime.vx
    anchor_vy = initial_flow_regime.vy
    final_metric_vx = final_flow_regime.vx

    concentration_reference, reference_history, reference_trace = _simulate_transport_reference(
        model,
        flow_regimes=flow_regimes,
        steps=steps,
        sources=sources,
        record_stride=record_stride,
        plane_columns=plane_columns,
    )
    reference_trace_summary = breakthrough_summary(reference_trace, plane_names=list(plane_columns.keys()))

    solver = solver_from_config(config.get("quantum_solver", {"type": "exact"}))
    head_state = solver.solve_state(
        flow_matrix,
        flow_rhs,
        label="flow_head",
        noise_key="flow_head",
    )

    mcr_config = config.get("mcr", {})
    observation_noise = mcr_config.get("observation_noise", {})
    rng = np.random.default_rng(int(observation_noise.get("seed", 0)))
    flow_methods = _resolve_mcr_methods(mcr_config, "flow")
    transport_methods = _resolve_mcr_methods(mcr_config, "transport")

    flow_constraint_sets = mcr_config.get(
        "flow_constraint_sets",
        [{"name": "dirichlet_boundary", "type": "dirichlet_boundary"}],
    )
    transport_constraint_sets = mcr_config.get(
        "transport_constraint_sets",
        [{"name": "sources", "type": "sources"}],
    )

    rows: list[dict[str, Any]] = []
    fields: dict[str, np.ndarray] = {
        "k_field": k_field,
        "head_reference": head_reference,
        "concentration_reference": concentration_reference,
        "head_state_normalized": head_state.state.reshape(model.ny, model.nx),
    }
    if len(flow_regimes) > 1:
        fields["head_reference_initial"] = initial_flow_regime.head
        fields["head_reference_final"] = final_flow_regime.head

    flow_rows, flow_fields = _run_final_state_mcr_family(
        field_name="head",
        state_result=head_state,
        reference_field=head_reference,
        model=model,
        constraint_sets=flow_constraint_sets,
        method_configs=flow_methods,
        h_left=h_left,
        h_right=h_right,
        flow_boundary=flow_boundary,
        sources=sources,
        observation_noise=observation_noise,
        rng=rng,
        anchor_vx=final_flow_regime.vx,
        anchor_vy=final_flow_regime.vy,
        metric_vx=final_metric_vx,
        plane_columns=plane_columns,
    )
    rows.extend(flow_rows)
    fields.update(flow_fields)

    transport_quantum = config.get("transport_quantum", {})
    transport_mode = _transport_quantum_mode(config)
    concentration_state_result: QuantumStateResult | None = None

    if transport_mode in {"final_field_normalized", "final_state_baseline", "legacy_final_field"}:
        concentration_state_result = solver.normalize_vector(
            concentration_reference.reshape(-1),
            label="transport_final_field_normalized",
            noise_key="transport_final_field_normalized",
        )
        fields["concentration_state_normalized"] = concentration_state_result.state.reshape(model.ny, model.nx)
        transport_rows, transport_fields = _run_final_state_mcr_family(
            field_name="concentration",
            state_result=concentration_state_result,
            reference_field=concentration_reference,
            model=model,
            constraint_sets=transport_constraint_sets,
            method_configs=transport_methods,
            h_left=h_left,
            h_right=h_right,
            flow_boundary=flow_boundary,
            sources=sources,
            observation_noise=observation_noise,
            rng=rng,
            x_coords=model.x_coords,
            y_coords=model.y_coords,
            anchor_vx=anchor_vx,
            anchor_vy=anchor_vy,
            metric_vx=final_metric_vx,
            plane_columns=plane_columns,
            flow_wells=final_flow_regime.wells,
        )
    elif transport_mode in {"operator_stepwise_hybrid", "operator_level", "stepwise_operator_mcr"}:
        transport_rows, transport_fields, concentration_state_result = _run_operator_transport_family(
            solver=solver,
            model=model,
            reference_history=reference_history,
            reference_trace_summary=reference_trace_summary,
            flow_regimes=flow_regimes,
            steps=steps,
            sources=sources,
            anchor_vx=anchor_vx,
            anchor_vy=anchor_vy,
            metric_vx=final_metric_vx,
            record_stride=record_stride,
            plane_columns=plane_columns,
            constraint_sets=transport_constraint_sets,
            method_configs=transport_methods,
            h_left=h_left,
            h_right=h_right,
            flow_boundary=flow_boundary,
            observation_noise=observation_noise,
            rng=rng,
            transport_quantum_config=transport_quantum,
            flow_wells=final_flow_regime.wells,
        )
        if concentration_state_result is not None:
            fields["concentration_state_normalized"] = concentration_state_result.state.reshape(model.ny, model.nx)
    else:
        raise ValueError(f"Unknown transport quantum mode: {transport_mode}")

    for row in transport_rows:
        row["transport_quantum_mode"] = transport_mode
    rows.extend(transport_rows)
    fields.update(transport_fields)

    benchmark = config.get("benchmark", {})
    k_field_config = config.get("k_field", {})
    common: dict[str, Any] = {
        "experiment": name,
        "nx": model.nx,
        "ny": model.ny,
        "n_nodes": model.n_nodes,
        "dx": model.dx,
        "dy": model.dy,
        "dt": model.dt,
        "steps": steps,
        "h_left": h_left,
        "h_right": h_right,
        "dispersion": model.dispersion,
        "k_min": float(k_field.min()),
        "k_max": float(k_field.max()),
        "k_mean": float(k_field.mean()),
        "k_std": float(k_field.std()),
        "k_field_model": str(k_field_config.get("model", "lognormal")),
        "k_field_seed": k_field_config.get("seed", np.nan),
        "k_field_correlation_length": k_field_config.get("correlation_length", np.nan),
        "k_field_log_mean": k_field_config.get("log_mean", np.nan),
        "k_field_log_variance": k_field_config.get("log_variance", np.nan),
        "source_count": len(sources),
        "flow_well_count": len(flow_wells),
        "flow_well_active_count_initial": len(initial_flow_regime.wells),
        "flow_well_active_count_final": len(final_flow_regime.wells),
        "flow_well_total_rate": _sum_well_rate(flow_wells, step_idx=steps) if flow_wells else 0.0,
        "flow_well_total_rate_initial": _sum_well_rate(flow_wells, step_idx=1) if flow_wells else 0.0,
        "flow_well_total_rate_peak_abs": max(
            (abs(_sum_well_rate(flow_wells, step_idx=step_idx)) for step_idx in range(1, steps + 1)),
            default=0.0,
        ),
        "flow_has_transient_wells": _has_transient_wells(flow_wells),
        "flow_schedule_n_regimes": len(flow_regimes),
        "flow_boundary_mode": str((flow_boundary or {}).get("mode", "left_right_dirichlet")),
        "flow_boundary_left_type": resolved_flow_boundary["left"]["type"],
        "flow_boundary_right_type": resolved_flow_boundary["right"]["type"],
        "flow_boundary_top_type": resolved_flow_boundary["top"]["type"],
        "flow_boundary_bottom_type": resolved_flow_boundary["bottom"]["type"],
        "transport_has_transient_sources": any(
            source.end_step is not None
            or str(source.profile).lower() not in {"constant", "boxcar"}
            or int(source.start_step) > 1
            for source in sources
        ),
        "transport_record_stride": record_stride,
        "head_state_true_norm": head_state.true_norm,
        "transport_quantum_mode": transport_mode,
        "concentration_reference_l2_norm": float(np.linalg.norm(concentration_reference.reshape(-1))),
        "observation_noise_seed": int(observation_noise.get("seed", 0)),
        "observation_noise_protocol": "keyed_shared_v1",
        "shared_operator_observation_noise": True,
    }
    if benchmark:
        common.update({f"benchmark_{key}": value for key, value in benchmark.items()})
    solver_metadata = {k: v for k, v in head_state.metadata.items() if k != "label"}
    common.update({f"quantum_{k}": v for k, v in solver_metadata.items()})
    common["head_state_label"] = head_state.metadata.get("label", "flow_head")
    common.update(matrix_metrics("flow_matrix", flow_matrix))

    common.update(matrix_metrics("transport_matrix", final_flow_regime.transport_matrix))
    common.update({f"reference_trace_{key}": value for key, value in reference_trace_summary.items()})

    if config.get("metrics", {}).get("estimate_condition_numbers", False):
        common["flow_matrix_cond_est"] = condition_number_estimate(flow_matrix)
        common["transport_matrix_cond_est"] = condition_number_estimate(final_flow_regime.transport_matrix)

    rows = [{**common, **row} for row in rows]

    metrics_path = out_dir / "metrics.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(stringify_metrics(row))

    trace_path = out_dir / "transport_reference_trace.csv"
    if reference_trace:
        fieldnames = sorted({key for row in reference_trace for key in row.keys()})
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reference_trace:
                writer.writerow(stringify_metrics(row))

    np.savez_compressed(out_dir / "fields.npz", **fields)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "transport_trace_summary.json").write_text(
        json.dumps(reference_trace_summary, indent=2),
        encoding="utf-8",
    )

    summary = {
        "experiment": name,
        "output_dir": str(out_dir),
        "metrics_csv": str(metrics_path),
        "transport_reference_trace_csv": str(trace_path),
        "n_metric_rows": len(rows),
        "fields": sorted(fields.keys()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a quantum-hydro experiment from config.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = run_experiment(config, output_root=args.output_root)
    print(f"[OK] Experiment saved to {out_dir}")


if __name__ == "__main__":
    main()
