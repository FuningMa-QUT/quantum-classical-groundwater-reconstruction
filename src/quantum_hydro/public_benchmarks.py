"""Adapters for official public groundwater benchmark cases."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .benchmarks import merge_nested_dicts, resolve_benchmark_case


PUBLIC_GROUNDWATER_BENCHMARK_LIBRARY: dict[str, dict[str, Any]] = {
    "modflow6_mt3dms_p05": {
        "benchmark": {
            "name": "modflow6_mt3dms_p05",
            "label": "MODFLOW 6 P05 Radial Flow",
            "family": "public_modflow6_mt3dms",
            "description": "Official MODFLOW 6 MT3DMS Problem 5 radial-flow transport benchmark adapted to the quantum-hydro surrogate.",
            "source_type": "official_modflow6_example",
            "source_id": "ex-gwt-mt3dms-p05",
            "source_title": "MT3DMS Problem 5",
            "source_url": "https://modflow6-examples.readthedocs.io/en/latest/_examples/ex-gwt-mt3dms-p05.html",
            "reference_citation": "MODFLOW 6 Examples ex-gwt-mt3dms-p05; benchmark origin in Moench and Ogata (1981).",
            "adaptation_type": "structured_public_adapter",
            "adaptation_notes": "Preserves the official grid, perimeter constant-head drainage, center injection topology, and 27-day radial transport setting; concentration source strength is normalized by cell pore volume for the operator-level surrogate.",
        },
        "domain": {
            "nx": 31,
            "ny": 31,
            "lx": 300.0,
            "ly": 300.0,
        },
        "physics": {
            "h_left": 0.0,
            "h_right": 0.0,
            "flow_boundary": {
                "mode": "perimeter_dirichlet",
                "value": 0.0,
            },
            "dispersion": 5.0,
            "dt": 1.0,
            "steps": 27,
            "sources": [
                {
                    "name": "central_injection",
                    "i": 15,
                    "j": 15,
                    "value": 3.3333333333333335,
                    "profile": "boxcar",
                    "start_step": 1,
                    "end_step": 27,
                }
            ],
            "flow_wells": [
                {
                    "name": "injector_center",
                    "i": 15,
                    "j": 15,
                    "value": 100.0,
                }
            ],
        },
        "k_field": {
            "model": "constant",
            "k_min": 1.0,
        },
        "transport_diagnostics": {
            "control_plane_fractions": [0.58, 0.7, 0.82, 0.9],
            "record_stride": 1,
        },
        "mcr": {
            "transport_methods": [
                {"name": "global", "type": "global"},
                {
                    "name": "adaptive_eb_radial4",
                    "type": "regional_adaptive_eb",
                    "basis": "radial_gaussian",
                    "axis": "radial",
                    "center_i": 15,
                    "center_j": 15,
                    "n_regions": 4,
                    "basis_width_scale": 1.5,
                    "prior_std_scale": 1.5,
                    "min_constraints_per_region": 8,
                    "max_shrinkage": 0.95,
                    "stability_cv_threshold": 0.45,
                    "stability_cv_scale": 2.0
                },
                {
                    "name": "regional_radial4",
                    "type": "regional_bayesian",
                    "basis": "radial_gaussian",
                    "axis": "radial",
                    "center_i": 15,
                    "center_j": 15,
                    "n_regions": 4,
                    "basis_width_scale": 1.5,
                    "min_constraints_per_region": 8,
                    "prior_std_scale": 1.5
                }
            ]
        },
        "sweep_options": {
            "transport_constraint_strategy": "pathline_monitoring"
        },
        "transport_quantum": {
            "observation_stride": 1
        }
    },
    "modflow6_mt3dms_p06": {
        "benchmark": {
            "name": "modflow6_mt3dms_p06",
            "label": "MODFLOW 6 P06 Injection-Extraction Well",
            "family": "public_modflow6_mt3dms",
            "description": "Official MODFLOW 6 MT3DMS Problem 6 benchmark with well reversal from injection to extraction.",
            "source_type": "official_modflow6_example",
            "source_id": "ex-gwt-mt3dms-p06",
            "source_title": "MT3DMS Problem 6",
            "source_url": "https://modflow6-examples.readthedocs.io/en/latest/_examples/ex-gwt-mt3dms-p06.html",
            "reference_citation": "MODFLOW 6 Examples ex-gwt-mt3dms-p06; benchmark origin in El-Kadi (1988) and Zheng (1993).",
            "adaptation_type": "structured_public_adapter",
            "adaptation_notes": "Preserves the official 31x31 radial-domain geometry, perimeter constant-head drainage, 2.5-year injection followed by 7.5-year extraction at the same well, and the immediate steady-flow assumption within each stress period; the surrogate uses a piecewise-steady transient well schedule and a pore-volume-normalized concentration source during the injection phase.",
        },
        "domain": {
            "nx": 31,
            "ny": 31,
            "lx": 27900.0,
            "ly": 27900.0,
        },
        "physics": {
            "h_left": 0.0,
            "h_right": 0.0,
            "flow_boundary": {
                "mode": "perimeter_dirichlet",
                "value": 0.0
            },
            "dispersion": 100.0,
            "dt": 2.5,
            "steps": 1460,
            "sources": [
                {
                    "name": "injection_concentration",
                    "i": 15,
                    "j": 15,
                    "value": 1.5238095238095237,
                    "profile": "boxcar",
                    "start_step": 1,
                    "end_step": 365
                }
            ],
            "flow_wells": [
                {
                    "name": "inject_phase",
                    "i": 15,
                    "j": 15,
                    "value": 86400.0,
                    "profile": "boxcar",
                    "start_step": 1,
                    "end_step": 365
                },
                {
                    "name": "extract_phase",
                    "i": 15,
                    "j": 15,
                    "value": -86400.0,
                    "profile": "boxcar",
                    "start_step": 366,
                    "end_step": 1460
                }
            ]
        },
        "k_field": {
            "model": "constant",
            "k_min": 432.0
        },
        "transport_diagnostics": {
            "control_plane_fractions": [0.58, 0.7, 0.82, 0.9],
            "record_stride": 20
        },
        "transport_quantum": {
            "observation_stride": 20,
            "phase_observation_strides": {
                "injection": 40,
                "pumpback": 14,
            },
            "phase_constraint_overrides": {
                "pumpback": {
                    "pathline_monitoring": {
                        "type": "capture_monitoring",
                        "include_capture_wells": True,
                        "capture_ring_budget_fraction": 0.55,
                        "capture_corridor_budget_fraction": 0.3,
                        "capture_ring_radii_cells": [2, 4, 7, 10],
                        "capture_ring_angle_count": 20
                    }
                }
            }
        },
        "mcr": {
            "transport_methods": [
                {"name": "global", "type": "global"},
                {
                    "name": "adaptive_eb_radial4",
                    "type": "regional_adaptive_eb",
                    "basis": "radial_gaussian",
                    "axis": "radial",
                    "center_i": 15,
                    "center_j": 15,
                    "n_regions": 4,
                    "basis_width_scale": 1.5,
                    "prior_std_scale": 1.5,
                    "min_constraints_per_region": 8,
                    "max_shrinkage": 0.95,
                    "stability_cv_threshold": 0.45,
                    "stability_cv_scale": 2.0
                },
                {
                    "name": "adaptive_eb_hlradial4",
                    "type": "regional_adaptive_eb",
                    "basis": "radial_hierarchical_lowrank",
                    "axis": "radial",
                    "center_i": 15,
                    "center_j": 15,
                    "n_regions": 4,
                    "basis_width_scale": 1.35,
                    "prior_std_scale": 1.5,
                    "hierarchical_levels": 4,
                    "low_rank_rank": 4,
                    "low_rank_energy": 0.97,
                    "low_rank_singular_power": 1.5,
                    "low_rank_min_prior_scale": 0.2,
                    "min_constraints_per_region": 8,
                    "max_shrinkage": 0.25,
                    "support_power": 1.25,
                    "stability_cv_threshold": 0.3,
                    "stability_cv_scale": 0.35,
                    "stability_level_threshold": 0.45,
                    "stability_level_scale": 0.15,
                    "anchor_jackknife_folds": 4,
                    "anchor_jackknife_min_train_constraints": 8,
                    "anchor_jackknife_ratio_threshold": 1.15,
                    "anchor_jackknife_ratio_scale": 0.18,
                    "anchor_jackknife_tail_quantile": 0.9,
                    "anchor_jackknife_tail_weight": 0.65,
                    "anchor_jackknife_excursion_threshold": 0.16,
                    "anchor_jackknife_excursion_scale": 0.06,
                    "stress_period_holdout_folds": 2,
                    "stress_period_holdout_min_train_constraints": 8,
                    "stress_period_holdout_ratio_threshold": 1.1,
                    "stress_period_holdout_ratio_scale": 0.12,
                    "stress_period_holdout_tail_quantile": 1.0,
                    "stress_period_holdout_tail_weight": 0.8,
                    "stress_period_holdout_excursion_threshold": 0.14,
                    "stress_period_holdout_excursion_scale": 0.05,
                    "validation_variance_scale": 1.5,
                    "validation_hard_gate_threshold": 1.0e-4,
                    "local_evidence_window": True,
                    "local_evidence_anchor_power": 1.25,
                    "local_evidence_temporal_power": 1.0,
                    "local_evidence_validation_power": 1.25,
                    "local_evidence_phase_aware": True,
                    "local_evidence_phase_power": 1.0,
                    "local_evidence_phase_bootstrap_support": 0.0,
                    "local_evidence_phase_bootstrap_support_injection": 1.0,
                    "local_evidence_phase_bootstrap_support_pumpback": 0.35,
                    "local_evidence_phase_ratio_support_threshold": 1.0,
                    "local_evidence_phase_ratio_support_scale": 0.05,
                    "local_evidence_phase_support_decay": 0.6,
                    "local_evidence_phase_transition_reset": True,
                    "local_evidence_group_smoothing_scale": 0.75,
                    "local_evidence_group_floor": 0.0,
                    "local_evidence_ratio_support_threshold": 1.0,
                    "local_evidence_ratio_support_scale": 0.05,
                    "local_evidence_min_peak": 0.045,
                    "local_evidence_latch_fallback": False
                },
                {
                    "name": "regional_radial4",
                    "type": "regional_bayesian",
                    "basis": "radial_gaussian",
                    "axis": "radial",
                    "center_i": 15,
                    "center_j": 15,
                    "n_regions": 4,
                    "basis_width_scale": 1.5,
                    "min_constraints_per_region": 8,
                    "prior_std_scale": 1.5
                },
                {
                    "name": "regional_hlradial4",
                    "type": "regional_bayesian",
                    "basis": "radial_hierarchical_lowrank",
                    "axis": "radial",
                    "center_i": 15,
                    "center_j": 15,
                    "n_regions": 4,
                    "basis_width_scale": 1.35,
                    "prior_std_scale": 1.5,
                    "hierarchical_levels": 4,
                    "low_rank_rank": 4,
                    "low_rank_energy": 0.97,
                    "low_rank_singular_power": 1.5,
                    "low_rank_min_prior_scale": 0.2,
                    "min_constraints_per_region": 8
                }
            ]
        },
        "sweep_options": {
            "transport_constraint_strategy": "pathline_monitoring"
        }
    },
    "modflow6_mt3dms_p09": {
        "benchmark": {
            "name": "modflow6_mt3dms_p09",
            "label": "MODFLOW 6 P09 Two-Dimensional Application",
            "family": "public_modflow6_mt3dms",
            "description": "Official MODFLOW 6 MT3DMS Problem 9 heterogeneous transport benchmark with a low-K block and coupled injection-extraction wells.",
            "source_type": "official_modflow6_example",
            "source_id": "ex-gwt-mt3dms-p09",
            "source_title": "MT3DMS Problem 9",
            "source_url": "https://modflow6-examples.readthedocs.io/en/latest/_examples/ex-gwt-mt3dms-p09.html",
            "reference_citation": "MODFLOW 6 Examples ex-gwt-mt3dms-p09; original benchmark in Zheng and Wang (1999).",
            "adaptation_type": "structured_public_adapter",
            "adaptation_notes": "Preserves the official 18x14 plan-view grid, north-south constant-head gradient, low-K block geometry, and two-period source schedule; the concentration forcing is cell-volume normalized for the operator-level surrogate.",
        },
        "domain": {
            "nx": 14,
            "ny": 18,
            "lx": 1300.0,
            "ly": 1700.0,
        },
        "physics": {
            "h_left": 250.0,
            "h_right": 20.0,
            "flow_boundary": {
                "mode": "top_bottom_dirichlet",
                "top": 250.0,
                "bottom": {
                    "type": "dirichlet",
                    "profile": "linear_ramp",
                    "start": 20.0,
                    "end": 52.5,
                },
                "left": {"type": "no_flow"},
                "right": {"type": "no_flow"},
            },
            "dispersion": 4.0e-4,
            "dt": 86400.0,
            "steps": 730,
            "sources": [
                {
                    "name": "injection_well_concentration",
                    "i": 3,
                    "j": 6,
                    "value": 1.929e-6,
                    "profile": "boxcar",
                    "start_step": 1,
                    "end_step": 365,
                }
            ],
            "flow_wells": [
                {
                    "name": "injector",
                    "i": 3,
                    "j": 6,
                    "value": 0.001,
                },
                {
                    "name": "pump",
                    "i": 10,
                    "j": 6,
                    "value": -0.0189,
                },
            ],
        },
        "k_field": {
            "model": "constant",
            "k_min": 1.474e-4,
            "rectangular_overlays": [
                {
                    "mode": "multiply",
                    "i_start": 5,
                    "i_stop": 8,
                    "j_start": 1,
                    "j_stop": 8,
                    "multiplier": 0.001,
                }
            ],
        },
        "transport_diagnostics": {
            "control_plane_fractions": [0.35, 0.5, 0.7, 0.86],
            "record_stride": 10,
        },
        "mcr": {
            "transport_methods": [
                {"name": "global", "type": "global"},
                {
                    "name": "adaptive_eb_y4",
                    "type": "regional_adaptive_eb",
                    "basis": "y_gaussian",
                    "axis": "y",
                    "n_regions": 4,
                    "basis_width_scale": 1.5,
                    "prior_std_scale": 1.5,
                    "min_constraints_per_region": 8,
                    "max_shrinkage": 0.95,
                    "stability_cv_threshold": 0.45,
                    "stability_cv_scale": 2.0
                },
                {
                    "name": "regional_y4",
                    "type": "regional_bayesian",
                    "basis": "y_gaussian",
                    "axis": "y",
                    "n_regions": 4,
                    "basis_width_scale": 1.5,
                    "min_constraints_per_region": 8,
                    "prior_std_scale": 1.5
                }
            ]
        },
        "sweep_options": {
            "transport_constraint_strategy": "pathline_monitoring"
        }
    },
}


def _register_public_benchmark_variant(
    *,
    preset: str,
    name: str,
    label: str,
    description: str,
    adaptation_notes: str,
    transport_quantum: dict[str, Any],
) -> None:
    variant = deepcopy(PUBLIC_GROUNDWATER_BENCHMARK_LIBRARY[preset])
    variant["name"] = str(name)
    variant["benchmark"] = deepcopy(variant.get("benchmark", {}))
    variant["benchmark"]["name"] = str(name)
    variant["benchmark"]["label"] = str(label)
    variant["benchmark"]["description"] = str(description)
    variant["benchmark"]["adaptation_notes"] = str(adaptation_notes)
    variant["transport_quantum"] = deepcopy(transport_quantum)
    PUBLIC_GROUNDWATER_BENCHMARK_LIBRARY[str(name)] = variant


_register_public_benchmark_variant(
    preset="modflow6_mt3dms_p06",
    name="modflow6_mt3dms_p06_uniform_reference",
    label="MODFLOW 6 P06 Uniform Monitoring Reference",
    description=(
        "P06 reference protocol with a uniform per-phase operator-observation stride and "
        "pathline monitoring anchors."
    ),
    adaptation_notes=(
        "Matches the public P06 adaptation geometry and transient injection-extraction schedule, "
        "but uses a single uniform operator-observation stride that resets at phase boundaries, "
        "without phase-specific monitoring overrides; this serves as the budget reference for "
        "the phase-aware protocol ablation."
    ),
    transport_quantum={
        "observation_stride": 20,
    },
)


_register_public_benchmark_variant(
    preset="modflow6_mt3dms_p06",
    name="modflow6_mt3dms_p06_phaseprotocol_budgetmatched",
    label="MODFLOW 6 P06 Phase Protocol Budget-Matched",
    description=(
        "P06 phase-aware monitoring protocol with pumpback capture anchors and an exact "
        "budget-matched operator-observation allocation."
    ),
    adaptation_notes=(
        "Preserves the phase-aware pumpback capture monitoring design, but redistributes the "
        "same total operator-observation budget as the uniform-stride reference toward the "
        "pumpback phase via denser stride and matched per-observation anchor budgets."
    ),
    transport_quantum={
        "observation_stride": 20,
        "phase_observation_strides": {
            "injection": 40,
            "pumpback": 14,
        },
        "phase_constraint_overrides": {
            "pumpback": {
                "pathline_monitoring": {
                    "type": "capture_monitoring",
                    "include_capture_wells": True,
                    "capture_ring_budget_fraction": 0.55,
                    "capture_corridor_budget_fraction": 0.3,
                    "capture_ring_radii_cells": [2, 4, 7, 10],
                    "capture_ring_angle_count": 20,
                }
            }
        },
        "phase_anchor_budget_factors": {
            "injection": 0.5,
            "pumpback": 1.0,
        },
        "budget_match_total_observations": True,
        "budget_match_reference_mode": "phase_reset_stride",
        "budget_match_reference_stride": 20,
        "phase_anchor_budget_minimum": 1,
    },
)


def available_public_benchmark_cases() -> list[str]:
    return sorted(PUBLIC_GROUNDWATER_BENCHMARK_LIBRARY)


def get_public_benchmark_case(name: str) -> dict[str, Any]:
    case_name = str(name)
    if case_name not in PUBLIC_GROUNDWATER_BENCHMARK_LIBRARY:
        known = ", ".join(available_public_benchmark_cases())
        raise ValueError(f"Unknown public benchmark case: {case_name}. Known cases: {known}")
    return deepcopy(PUBLIC_GROUNDWATER_BENCHMARK_LIBRARY[case_name])


def resolve_public_benchmark_case(case: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(case, str):
        resolved = get_public_benchmark_case(case)
    elif isinstance(case, dict):
        preset = case.get("preset")
        if preset is None:
            resolved = deepcopy(case)
        else:
            resolved = merge_nested_dicts(
                get_public_benchmark_case(str(preset)),
                {key: value for key, value in case.items() if key != "preset"},
            )
    else:
        raise TypeError(f"Public benchmark case must be a string or mapping, got {type(case)!r}")

    benchmark = deepcopy(resolved.get("benchmark", {}))
    benchmark.setdefault("name", str(resolved.get("name", "public_groundwater_benchmark")))
    benchmark.setdefault("label", str(benchmark["name"]).replace("_", " ").title())
    benchmark.setdefault("family", "public_modflow6_mt3dms")
    resolved["benchmark"] = benchmark
    resolved.setdefault("name", str(benchmark["name"]))
    return resolved


def resolve_any_benchmark_case(case: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(case, str):
        if case in PUBLIC_GROUNDWATER_BENCHMARK_LIBRARY:
            return resolve_public_benchmark_case(case)
        return resolve_benchmark_case(case)

    if isinstance(case, dict):
        preset = case.get("preset")
        if preset is not None and str(preset) in PUBLIC_GROUNDWATER_BENCHMARK_LIBRARY:
            return resolve_public_benchmark_case(case)
        return resolve_benchmark_case(case)

    raise TypeError(f"Benchmark case must be a string or mapping, got {type(case)!r}")
