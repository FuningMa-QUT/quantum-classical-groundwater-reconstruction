"""Idealized and noisy quantum-state solver interfaces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass
class QuantumStateResult:
    state: np.ndarray
    true_norm: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ExactStateSolver:
    """Classical emulation of an ideal normalized QLSA output."""

    def solve_vector(self, matrix: sparse.spmatrix | np.ndarray, rhs: np.ndarray) -> np.ndarray:
        if sparse.issparse(matrix):
            return spla.spsolve(matrix.tocsc(), rhs)
        return np.linalg.solve(np.asarray(matrix, dtype=float), rhs)

    def normalize_vector(
        self,
        vector: np.ndarray,
        *,
        label: str = "vector",
        noise_key: str | None = None,
    ) -> QuantumStateResult:
        vec = np.asarray(vector, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-15:
            state = vec.copy()
        else:
            state = vec / norm
        return QuantumStateResult(
            state=state,
            true_norm=norm,
            metadata={
                "solver": "exact",
                "label": label,
                "noise_protocol": "deterministic_exact",
                "noise_key": label if noise_key is None else str(noise_key),
            },
        )

    def solve_state(
        self,
        matrix: sparse.spmatrix | np.ndarray,
        rhs: np.ndarray,
        *,
        label: str = "linear_solve",
        noise_key: str | None = None,
    ) -> QuantumStateResult:
        return self.normalize_vector(self.solve_vector(matrix, rhs), label=label, noise_key=noise_key)


class NoisyStateSolver(ExactStateSolver):
    """Noisy normalized-state emulator.

    The solver first computes the ideal normalized state, then optionally applies
    finite-shot amplitude sampling, additive amplitude noise, and random sign
    flips. This keeps the first upgraded codebase hardware-agnostic while making
    the MCR protocol testable under realistic readout imperfections.
    """

    def __init__(
        self,
        *,
        shots: int | None = None,
        amplitude_noise_std: float = 0.0,
        relative_noise_std: float = 0.0,
        sign_flip_probability: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.shots = None if shots is None else int(shots)
        self.amplitude_noise_std = float(amplitude_noise_std)
        self.relative_noise_std = float(relative_noise_std)
        self.sign_flip_probability = float(sign_flip_probability)
        self.seed = 0 if seed is None else int(seed)

    def _rng_for_key(self, *, label: str, noise_key: str | None) -> np.random.Generator:
        resolved_key = label if noise_key is None else str(noise_key)
        token = f"quantum_state_noise|{self.seed}|{resolved_key}".encode("utf-8")
        digest = hashlib.blake2b(token, digest_size=16).digest()
        return np.random.default_rng(int.from_bytes(digest, "little", signed=False))

    def normalize_vector(
        self,
        vector: np.ndarray,
        *,
        label: str = "vector",
        noise_key: str | None = None,
    ) -> QuantumStateResult:
        ideal = super().normalize_vector(vector, label=label, noise_key=noise_key)
        state = ideal.state.copy()
        rng = self._rng_for_key(label=label, noise_key=noise_key)

        if self.shots is not None and self.shots > 0:
            probabilities = state**2
            prob_sum = float(probabilities.sum())
            if prob_sum > 1e-15:
                probabilities = probabilities / prob_sum
                counts = rng.multinomial(self.shots, probabilities)
                magnitudes = np.sqrt(counts / float(self.shots))
                state = magnitudes * np.sign(state)

        if self.relative_noise_std > 0.0:
            scale = np.maximum(np.abs(state), 1.0 / max(1, state.size))
            state = state + rng.normal(0.0, self.relative_noise_std, size=state.size) * scale

        if self.amplitude_noise_std > 0.0:
            state = state + rng.normal(0.0, self.amplitude_noise_std, size=state.size)

        if self.sign_flip_probability > 0.0:
            flips = rng.random(state.size) < self.sign_flip_probability
            state[flips] *= -1.0

        norm = float(np.linalg.norm(state))
        if norm > 1e-15:
            state = state / norm

        metadata = dict(ideal.metadata)
        metadata.update(
            {
                "solver": "noisy",
                "shots": self.shots,
                "amplitude_noise_std": self.amplitude_noise_std,
                "relative_noise_std": self.relative_noise_std,
                "sign_flip_probability": self.sign_flip_probability,
                "seed": self.seed,
                "noise_protocol": "keyed_deterministic_v1",
                "noise_key": label if noise_key is None else str(noise_key),
            }
        )
        return QuantumStateResult(state=state, true_norm=ideal.true_norm, metadata=metadata)


def solver_from_config(config: dict[str, Any]) -> ExactStateSolver:
    solver_type = str(config.get("type", "exact")).lower()
    if solver_type in {"exact", "exact_state", "ideal"}:
        return ExactStateSolver()
    if solver_type in {"noisy", "noisy_state"}:
        return NoisyStateSolver(
            shots=config.get("shots"),
            amplitude_noise_std=config.get("amplitude_noise_std", 0.0),
            relative_noise_std=config.get("relative_noise_std", 0.0),
            sign_flip_probability=config.get("sign_flip_probability", 0.0),
            seed=config.get("seed"),
        )
    raise ValueError(f"Unknown quantum solver type: {solver_type}")
