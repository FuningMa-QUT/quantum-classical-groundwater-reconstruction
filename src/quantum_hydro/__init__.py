"""Reusable components for quantum-ready groundwater experiments."""

from .anchors import AnchorPlacementResult, select_anchor_indices
from .kfields import generate_k_field
from .mcr import MCRResult, apply_mcr, apply_mcr_method, apply_regional_mcr, estimate_alpha
from .operators import GroundwaterModel, SourceTerm, WellTerm
from .public_benchmarks import available_public_benchmark_cases, resolve_any_benchmark_case
from .solvers import ExactStateSolver, NoisyStateSolver, QuantumStateResult

__all__ = [
    "AnchorPlacementResult",
    "ExactStateSolver",
    "GroundwaterModel",
    "MCRResult",
    "NoisyStateSolver",
    "QuantumStateResult",
    "SourceTerm",
    "WellTerm",
    "apply_mcr",
    "apply_mcr_method",
    "apply_regional_mcr",
    "available_public_benchmark_cases",
    "estimate_alpha",
    "generate_k_field",
    "resolve_any_benchmark_case",
    "select_anchor_indices",
]
