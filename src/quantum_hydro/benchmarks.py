"""Named groundwater benchmark presets for structured hydrogeologic scenarios."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def merge_nested_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings without mutating either input."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


GROUNDWATER_BENCHMARK_LIBRARY: dict[str, dict[str, Any]] = {
    "layered_aquifer": {
        "benchmark": {
            "name": "layered_aquifer",
            "label": "Layered Aquifer",
            "family": "structured_layered",
            "description": "Alternating aquifer-aquitard layering with superimposed lognormal sublayer variability.",
        },
        "domain": {
            "nx": 48,
            "ny": 48,
            "lx": 48.0,
            "ly": 48.0,
        },
        "physics": {
            "h_left": 24.0,
            "h_right": 10.0,
            "dispersion": 0.18,
            "dt": 0.12,
            "steps": 140,
            "sources": [
                {"i": 14, "j": 5, "value": 10.0},
                {"i": 33, "j": 7, "value": 8.0},
            ],
        },
        "k_field": {
            "model": "layered_lognormal",
            "correlation_length": 1.8,
            "log_mean": 1.9,
            "clip_min": 0.02,
            "clip_max": 500.0,
            "n_layers": 7,
            "layer_axis": "horizontal",
            "layer_multipliers": [0.12, 0.45, 2.4, 0.25, 1.5, 3.8, 0.6],
            "transition_sigma": 0.9,
            "jitter_strength": 0.06,
        },
        "transport_diagnostics": {
            "control_plane_fractions": [0.2, 0.45, 0.7, 0.85],
            "record_stride": 3,
        },
    },
    "meandering_channel": {
        "benchmark": {
            "name": "meandering_channel",
            "label": "Meandering Channel",
            "family": "structured_channelized",
            "description": "Connected high-K paleochannel network embedded in a lower-K matrix.",
        },
        "domain": {
            "nx": 56,
            "ny": 40,
            "lx": 56.0,
            "ly": 40.0,
        },
        "physics": {
            "h_left": 22.0,
            "h_right": 9.0,
            "dispersion": 0.14,
            "dt": 0.09,
            "steps": 170,
            "sources": [
                {"i": 10, "j": 4, "value": 8.0},
                {"i": 28, "j": 5, "value": 12.0},
                {"i": 34, "j": 6, "value": 7.0},
            ],
        },
        "k_field": {
            "model": "channelized_lognormal",
            "correlation_length": 1.6,
            "log_mean": 1.8,
            "clip_min": 0.02,
            "clip_max": 700.0,
            "channel_multiplier": 8.0,
            "background_multiplier": 0.75,
            "channel_width_fraction": 0.075,
            "channel_center_fraction": 0.48,
            "amplitude_fraction": 0.17,
            "meander_wavelength_fraction": 0.85,
            "phase_fraction": 0.12,
            "secondary_channel_scale": 0.55,
            "secondary_channel_offset_fraction": 0.22,
            "secondary_channel_shift_fraction": -0.18,
        },
        "transport_diagnostics": {
            "control_plane_fractions": [0.18, 0.42, 0.68, 0.86],
            "record_stride": 4,
        },
    },
    "barrier_lens": {
        "benchmark": {
            "name": "barrier_lens",
            "label": "Barrier Lens",
            "family": "structured_barrier",
            "description": "Low-permeability clay lenses that force plume diversion and bypassing.",
        },
        "domain": {
            "nx": 52,
            "ny": 44,
            "lx": 52.0,
            "ly": 44.0,
        },
        "physics": {
            "h_left": 23.0,
            "h_right": 10.0,
            "dispersion": 0.16,
            "dt": 0.1,
            "steps": 160,
            "sources": [
                {"i": 18, "j": 5, "value": 10.0},
                {"i": 24, "j": 6, "value": 7.5},
            ],
        },
        "k_field": {
            "model": "barrier_lens_lognormal",
            "correlation_length": 1.7,
            "log_mean": 1.9,
            "clip_min": 0.01,
            "clip_max": 600.0,
            "background_multiplier": 1.05,
            "transition_sigma": 0.9,
            "lenses": [
                {
                    "center_x_fraction": 0.52,
                    "center_y_fraction": 0.42,
                    "radius_x_fraction": 0.18,
                    "radius_y_fraction": 0.08,
                    "orientation_deg": 18.0,
                    "multiplier": 0.06,
                },
                {
                    "center_x_fraction": 0.63,
                    "center_y_fraction": 0.63,
                    "radius_x_fraction": 0.14,
                    "radius_y_fraction": 0.09,
                    "orientation_deg": -24.0,
                    "multiplier": 0.10,
                },
            ],
        },
        "transport_diagnostics": {
            "control_plane_fractions": [0.2, 0.5, 0.74, 0.9],
            "record_stride": 4,
        },
    },
    "patchy_facies": {
        "benchmark": {
            "name": "patchy_facies",
            "label": "Patchy Facies",
            "family": "structured_facies",
            "description": "Multi-facies conductivity mosaic with connected and disconnected preferential pathways.",
        },
        "domain": {
            "nx": 48,
            "ny": 48,
            "lx": 48.0,
            "ly": 48.0,
        },
        "physics": {
            "h_left": 22.0,
            "h_right": 9.0,
            "dispersion": 0.15,
            "dt": 0.1,
            "steps": 155,
            "sources": [
                {"i": 11, "j": 5, "value": 7.0},
                {"i": 24, "j": 4, "value": 10.0},
                {"i": 36, "j": 6, "value": 8.0},
            ],
        },
        "k_field": {
            "model": "facies_lognormal",
            "correlation_length": 1.8,
            "log_mean": 1.85,
            "clip_min": 0.015,
            "clip_max": 800.0,
            "transition_sigma": 0.7,
            "facies_correlation_length": 3.2,
            "facies_quantiles": [0.18, 0.5, 0.8],
            "facies_multipliers": [0.07, 0.3, 1.2, 5.8],
        },
        "transport_diagnostics": {
            "control_plane_fractions": [0.2, 0.4, 0.65, 0.85],
            "record_stride": 4,
        },
    },
    "pumped_channel_capture": {
        "benchmark": {
            "name": "pumped_channel_capture",
            "label": "Pumped Channel Capture",
            "family": "capture_channelized",
            "description": "Transient release in a channelized aquifer under downgradient extraction control.",
        },
        "domain": {
            "nx": 64,
            "ny": 48,
            "lx": 64.0,
            "ly": 48.0,
        },
        "physics": {
            "h_left": 18.0,
            "h_right": 11.0,
            "dispersion": 0.12,
            "dt": 0.08,
            "steps": 220,
            "sources": [
                {
                    "name": "spill_primary",
                    "i": 15,
                    "j": 6,
                    "value": 14.0,
                    "profile": "boxcar",
                    "start_step": 1,
                    "end_step": 42
                },
                {
                    "name": "spill_secondary",
                    "i": 31,
                    "j": 8,
                    "value": 9.0,
                    "profile": "exponential_decay",
                    "start_step": 26,
                    "end_step": 110,
                    "decay_tau_steps": 22.0
                }
            ],
            "flow_wells": [
                {"name": "pump_main", "i": 23, "j": 53, "value": -28.0},
                {"name": "pump_guard", "i": 33, "j": 50, "value": -18.0}
            ]
        },
        "k_field": {
            "model": "channelized_lognormal",
            "correlation_length": 1.5,
            "log_mean": 1.8,
            "clip_min": 0.015,
            "clip_max": 900.0,
            "channel_multiplier": 9.0,
            "background_multiplier": 0.65,
            "channel_width_fraction": 0.07,
            "channel_center_fraction": 0.46,
            "amplitude_fraction": 0.18,
            "meander_wavelength_fraction": 0.8,
            "phase_fraction": 0.08,
            "secondary_channel_scale": 0.5,
            "secondary_channel_offset_fraction": 0.18,
            "secondary_channel_shift_fraction": -0.14
        },
        "transport_diagnostics": {
            "control_plane_fractions": [0.18, 0.36, 0.58, 0.78, 0.9],
            "record_stride": 4,
        },
    },
    "doublet_barrier_remediation": {
        "benchmark": {
            "name": "doublet_barrier_remediation",
            "label": "Doublet Barrier Remediation",
            "family": "capture_barrier",
            "description": "Barrier-lens containment with dual extraction wells and pulsed source loading.",
        },
        "domain": {
            "nx": 58,
            "ny": 46,
            "lx": 58.0,
            "ly": 46.0,
        },
        "physics": {
            "h_left": 19.0,
            "h_right": 10.5,
            "dispersion": 0.14,
            "dt": 0.09,
            "steps": 210,
            "sources": [
                {
                    "name": "pulse_release_a",
                    "i": 17,
                    "j": 7,
                    "value": 11.0,
                    "profile": "triangular",
                    "start_step": 1,
                    "end_step": 70,
                    "peak_step": 28
                },
                {
                    "name": "pulse_release_b",
                    "i": 28,
                    "j": 9,
                    "value": 8.5,
                    "profile": "boxcar",
                    "start_step": 48,
                    "end_step": 104
                }
            ],
            "flow_wells": [
                {"name": "pump_north", "i": 15, "j": 47, "value": -20.0},
                {"name": "pump_south", "i": 31, "j": 45, "value": -24.0}
            ]
        },
        "k_field": {
            "model": "barrier_lens_lognormal",
            "correlation_length": 1.7,
            "log_mean": 1.85,
            "clip_min": 0.01,
            "clip_max": 700.0,
            "background_multiplier": 1.0,
            "transition_sigma": 0.8,
            "lenses": [
                {
                    "center_x_fraction": 0.46,
                    "center_y_fraction": 0.39,
                    "radius_x_fraction": 0.18,
                    "radius_y_fraction": 0.08,
                    "orientation_deg": 12.0,
                    "multiplier": 0.07
                },
                {
                    "center_x_fraction": 0.58,
                    "center_y_fraction": 0.61,
                    "radius_x_fraction": 0.16,
                    "radius_y_fraction": 0.1,
                    "orientation_deg": -20.0,
                    "multiplier": 0.09
                }
            ]
        },
        "transport_diagnostics": {
            "control_plane_fractions": [0.2, 0.42, 0.64, 0.82, 0.92],
            "record_stride": 4,
        },
    },
}


def available_benchmark_cases() -> list[str]:
    return sorted(GROUNDWATER_BENCHMARK_LIBRARY)


def get_benchmark_case(name: str) -> dict[str, Any]:
    case_name = str(name)
    if case_name not in GROUNDWATER_BENCHMARK_LIBRARY:
        known = ", ".join(available_benchmark_cases())
        raise ValueError(f"Unknown benchmark case: {case_name}. Known cases: {known}")
    return deepcopy(GROUNDWATER_BENCHMARK_LIBRARY[case_name])


def resolve_benchmark_case(case: str | dict[str, Any]) -> dict[str, Any]:
    """Resolve a benchmark preset name or preset+override mapping."""

    if isinstance(case, str):
        resolved = get_benchmark_case(case)
    elif isinstance(case, dict):
        preset = case.get("preset")
        if preset is None:
            resolved = deepcopy(case)
        else:
            resolved = merge_nested_dicts(
                get_benchmark_case(str(preset)),
                {key: value for key, value in case.items() if key != "preset"},
            )
    else:
        raise TypeError(f"Benchmark case must be a string or mapping, got {type(case)!r}")

    benchmark = deepcopy(resolved.get("benchmark", {}))
    benchmark.setdefault("name", str(resolved.get("name", "groundwater_benchmark")))
    benchmark.setdefault("label", str(benchmark["name"]).replace("_", " ").title())
    benchmark.setdefault("family", str(resolved.get("k_field", {}).get("model", "lognormal")))
    resolved["benchmark"] = benchmark
    resolved.setdefault("name", str(benchmark["name"]))
    return resolved
