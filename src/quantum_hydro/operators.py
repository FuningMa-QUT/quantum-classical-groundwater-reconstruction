"""Finite-difference operators for flow and solute transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def _scheduled_value_at_step(
    base_value: float,
    *,
    step_idx: int,
    start_step: int = 1,
    end_step: int | None = None,
    profile: str = "constant",
    peak_step: int | None = None,
    ramp_steps: int = 0,
    decay_tau_steps: float | None = None,
) -> float:
    step = int(step_idx)
    start = max(1, int(start_step))
    end = None if end_step is None else int(end_step)

    if step < start:
        return 0.0
    if end is not None and step > end:
        return 0.0

    kind = str(profile).lower()
    base = float(base_value)

    if kind in {"constant", "boxcar"}:
        return base

    if kind == "ramp":
        ramp = max(1, int(ramp_steps))
        phase = min(max(step - start + 1, 0), ramp)
        return base * phase / ramp

    if kind == "triangular":
        final = end if end is not None else start + max(2, int(ramp_steps) * 2)
        peak = int(peak_step) if peak_step is not None else start + max(1, (final - start) // 2)
        if peak <= start:
            peak = start + 1
        if step <= peak:
            return base * (step - start + 1) / max(1, peak - start + 1)
        return base * max(final - step + 1, 0) / max(1, final - peak + 1)

    if kind in {"exponential_decay", "exp_decay"}:
        tau = float(decay_tau_steps) if decay_tau_steps is not None else max(
            1.0,
            ((end - start + 1) / 3.0) if end is not None else 12.0,
        )
        return base * float(np.exp(-(step - start) / tau))

    raise ValueError(f"Unknown temporal profile: {profile}")


@dataclass(frozen=True)
class SourceTerm:
    i: int
    j: int
    value: float
    name: str = ""
    start_step: int = 1
    end_step: int | None = None
    profile: str = "constant"
    peak_step: int | None = None
    ramp_steps: int = 0
    decay_tau_steps: float | None = None

    def value_at_step(self, step_idx: int) -> float:
        return _scheduled_value_at_step(
            self.value,
            step_idx=step_idx,
            start_step=self.start_step,
            end_step=self.end_step,
            profile=self.profile,
            peak_step=self.peak_step,
            ramp_steps=self.ramp_steps,
            decay_tau_steps=self.decay_tau_steps,
        )


@dataclass(frozen=True)
class WellTerm:
    i: int
    j: int
    value: float
    name: str = ""
    start_step: int = 1
    end_step: int | None = None
    profile: str = "constant"
    peak_step: int | None = None
    ramp_steps: int = 0
    decay_tau_steps: float | None = None

    def value_at_step(self, step_idx: int) -> float:
        return _scheduled_value_at_step(
            self.value,
            step_idx=step_idx,
            start_step=self.start_step,
            end_step=self.end_step,
            profile=self.profile,
            peak_step=self.peak_step,
            ramp_steps=self.ramp_steps,
            decay_tau_steps=self.decay_tau_steps,
        )


class GroundwaterModel:
    """Structured-grid groundwater flow and ADE transport model."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        k_field: np.ndarray,
        dispersion: float,
        dt: float,
    ) -> None:
        if k_field.shape != (ny, nx):
            raise ValueError(f"k_field shape {k_field.shape} != ({ny}, {nx})")

        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)
        self.k_field = np.asarray(k_field, dtype=float)
        self.dispersion = float(dispersion)
        self.dt = float(dt)
        self.n_nodes = self.nx * self.ny
        self.x_coords = np.linspace(0.0, self.lx, self.nx)
        self.y_coords = np.linspace(0.0, self.ly, self.ny)

    def idx(self, i: int, j: int) -> int:
        return int(i) * self.nx + int(j)

    def unravel(self, index: int) -> tuple[int, int]:
        return divmod(int(index), self.nx)

    @staticmethod
    def harmonic_mean(a: float, b: float) -> float:
        return 2.0 * a * b / (a + b + 1e-15)

    def _boundary_axis_length(self, side: str) -> int:
        return self.ny if side in {"left", "right"} else self.nx

    def _normalize_boundary_side_spec(self, side: str, raw_spec: Any) -> dict[str, Any]:
        length = self._boundary_axis_length(side)

        if raw_spec is None:
            return {"type": "no_flow"}

        if np.isscalar(raw_spec):
            return {"type": "dirichlet", "values": np.full(length, float(raw_spec), dtype=float)}

        if isinstance(raw_spec, (list, tuple, np.ndarray)):
            values = np.asarray(raw_spec, dtype=float).reshape(-1)
            if values.size != length:
                raise ValueError(f"Boundary side {side!r} expects {length} values, got {values.size}")
            return {"type": "dirichlet", "values": values.astype(float, copy=False)}

        if not isinstance(raw_spec, dict):
            raise TypeError(f"Unsupported boundary specification for side {side!r}: {type(raw_spec)!r}")

        bc_type = str(raw_spec.get("type", "dirichlet")).lower()
        if bc_type in {"no_flow", "neumann_zero", "zero_gradient"}:
            return {"type": "no_flow"}
        if bc_type != "dirichlet":
            raise ValueError(f"Unsupported boundary type {bc_type!r} on side {side!r}")

        if "values" in raw_spec:
            values = np.asarray(raw_spec["values"], dtype=float).reshape(-1)
            if values.size != length:
                raise ValueError(f"Boundary side {side!r} expects {length} values, got {values.size}")
            return {"type": "dirichlet", "values": values.astype(float, copy=False)}

        profile = str(raw_spec.get("profile", "constant")).lower()
        if profile in {"linear", "linear_ramp"}:
            start = float(raw_spec.get("start", raw_spec.get("value", 0.0)))
            end = float(raw_spec.get("end", raw_spec.get("value", start)))
            return {"type": "dirichlet", "values": np.linspace(start, end, num=length, dtype=float)}

        value = float(raw_spec.get("value", 0.0))
        return {"type": "dirichlet", "values": np.full(length, value, dtype=float)}

    def resolve_flow_boundary(
        self,
        h_left: float,
        h_right: float,
        *,
        flow_boundary: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        raw = dict(flow_boundary or {})
        mode = str(raw.get("mode", "left_right_dirichlet")).lower()

        if mode in {"default", "legacy"}:
            mode = "left_right_dirichlet"

        if mode == "left_right_dirichlet":
            defaults: dict[str, Any] = {
                "left": {"type": "dirichlet", "value": h_left},
                "right": {"type": "dirichlet", "value": h_right},
                "top": {"type": "no_flow"},
                "bottom": {"type": "no_flow"},
            }
        elif mode == "perimeter_dirichlet":
            perimeter_value = raw.get("value", 0.0)
            defaults = {
                "left": {"type": "dirichlet", "value": perimeter_value},
                "right": {"type": "dirichlet", "value": perimeter_value},
                "top": {"type": "dirichlet", "value": perimeter_value},
                "bottom": {"type": "dirichlet", "value": perimeter_value},
            }
        elif mode == "top_bottom_dirichlet":
            defaults = {
                "left": {"type": "no_flow"},
                "right": {"type": "no_flow"},
                "top": {"type": "dirichlet", "value": h_left},
                "bottom": {"type": "dirichlet", "value": h_right},
            }
        elif mode == "custom":
            defaults = {
                "left": {"type": "no_flow"},
                "right": {"type": "no_flow"},
                "top": {"type": "no_flow"},
                "bottom": {"type": "no_flow"},
            }
        else:
            raise ValueError(f"Unknown flow boundary mode: {mode}")

        resolved: dict[str, dict[str, Any]] = {}
        for side, default_spec in defaults.items():
            resolved[side] = self._normalize_boundary_side_spec(side, raw.get(side, default_spec))
        return resolved

    def _boundary_value_at(self, side: str, spec: dict[str, Any], i: int, j: int) -> float:
        values = np.asarray(spec["values"], dtype=float)
        if side in {"left", "right"}:
            return float(values[int(i)])
        return float(values[int(j)])

    def _dirichlet_boundary_value(
        self,
        i: int,
        j: int,
        boundary: dict[str, dict[str, Any]],
    ) -> float | None:
        side_values: list[float] = []
        if j == 0 and boundary["left"]["type"] == "dirichlet":
            side_values.append(self._boundary_value_at("left", boundary["left"], i, j))
        if j == self.nx - 1 and boundary["right"]["type"] == "dirichlet":
            side_values.append(self._boundary_value_at("right", boundary["right"], i, j))
        if i == 0 and boundary["top"]["type"] == "dirichlet":
            side_values.append(self._boundary_value_at("top", boundary["top"], i, j))
        if i == self.ny - 1 and boundary["bottom"]["type"] == "dirichlet":
            side_values.append(self._boundary_value_at("bottom", boundary["bottom"], i, j))
        if not side_values:
            return None
        return float(np.mean(side_values))

    def flow_boundary_indices_and_values(
        self,
        h_left: float,
        h_right: float,
        *,
        flow_boundary: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        boundary = self.resolve_flow_boundary(h_left, h_right, flow_boundary=flow_boundary)
        indices: list[int] = []
        values: list[float] = []

        for i in range(self.ny):
            for j in range(self.nx):
                if i not in {0, self.ny - 1} and j not in {0, self.nx - 1}:
                    continue
                value = self._dirichlet_boundary_value(i, j, boundary)
                if value is None:
                    continue
                indices.append(self.idx(i, j))
                values.append(value)

        return np.asarray(indices, dtype=int), np.asarray(values, dtype=float)

    def build_flow_matrix(
        self,
        h_left: float,
        h_right: float,
        *,
        wells: Iterable[WellTerm] | None = None,
        flow_boundary: dict[str, Any] | None = None,
        step_idx: int | None = None,
    ) -> tuple[sparse.csc_matrix, np.ndarray]:
        """Build steady groundwater-flow matrix with configurable side boundaries."""

        mat = sparse.lil_matrix((self.n_nodes, self.n_nodes), dtype=float)
        rhs = np.zeros(self.n_nodes, dtype=float)
        well_rhs = self.well_vector([] if wells is None else wells, step_idx=step_idx)
        boundary = self.resolve_flow_boundary(h_left, h_right, flow_boundary=flow_boundary)

        for i in range(self.ny):
            for j in range(self.nx):
                row = self.idx(i, j)

                dirichlet_value = self._dirichlet_boundary_value(i, j, boundary)
                if dirichlet_value is not None:
                    mat[row, row] = 1.0
                    rhs[row] = dirichlet_value
                    continue

                if i == 0:
                    mat[row, row] = 1.0
                    mat[row, self.idx(i + 1, j)] = -1.0
                    continue
                if i == self.ny - 1:
                    mat[row, row] = 1.0
                    mat[row, self.idx(i - 1, j)] = -1.0
                    continue
                if j == 0:
                    mat[row, row] = 1.0
                    mat[row, self.idx(i, j + 1)] = -1.0
                    continue
                if j == self.nx - 1:
                    mat[row, row] = 1.0
                    mat[row, self.idx(i, j - 1)] = -1.0
                    continue

                k_center = float(self.k_field[i, j])
                k_left = self.harmonic_mean(k_center, float(self.k_field[i, j - 1]))
                k_right = self.harmonic_mean(k_center, float(self.k_field[i, j + 1]))
                k_down = self.harmonic_mean(k_center, float(self.k_field[i - 1, j]))
                k_up = self.harmonic_mean(k_center, float(self.k_field[i + 1, j]))

                mat[row, self.idx(i, j - 1)] = -k_left
                mat[row, self.idx(i, j + 1)] = -k_right
                mat[row, self.idx(i - 1, j)] = -k_down
                mat[row, self.idx(i + 1, j)] = -k_up
                mat[row, row] = k_left + k_right + k_down + k_up
                rhs[row] = well_rhs[row]

        return mat.tocsc(), rhs

    def build_transport_operators(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        *,
        dt: float | None = None,
    ) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
        """Build implicit upwind ADE matrix and continuous-time operator."""

        current_dt = self.dt if dt is None else float(dt)
        mat = sparse.lil_matrix((self.n_nodes, self.n_nodes), dtype=float)
        coeff_x = self.dispersion * current_dt / (self.dx**2)
        coeff_y = self.dispersion * current_dt / (self.dy**2)

        for i in range(self.ny):
            for j in range(self.nx):
                row = self.idx(i, j)

                if i == 0 or i == self.ny - 1 or j == 0 or j == self.nx - 1:
                    mat[row, row] = 1.0
                    continue

                u = float(vx[i, j])
                v = float(vy[i, j])
                courant_x = u * current_dt / self.dx
                courant_y = v * current_dt / self.dy

                upwind_x = self.idx(i, j - 1) if u > 0.0 else self.idx(i, j + 1)
                upwind_y = self.idx(i - 1, j) if v > 0.0 else self.idx(i + 1, j)

                mat[row, row] = 1.0 + 2.0 * (coeff_x + coeff_y) + abs(courant_x) + abs(courant_y)
                mat[row, self.idx(i, j - 1)] -= coeff_x
                mat[row, self.idx(i, j + 1)] -= coeff_x
                mat[row, self.idx(i - 1, j)] -= coeff_y
                mat[row, self.idx(i + 1, j)] -= coeff_y
                mat[row, upwind_x] -= abs(courant_x)
                mat[row, upwind_y] -= abs(courant_y)

        step = mat.tocsc()
        generator = (sparse.eye(self.n_nodes, format="csc") - step) / current_dt
        return step, generator

    def solve_flow(
        self,
        h_left: float,
        h_right: float,
        *,
        wells: Iterable[WellTerm] | None = None,
        flow_boundary: dict[str, Any] | None = None,
        step_idx: int | None = None,
    ) -> np.ndarray:
        mat, rhs = self.build_flow_matrix(
            h_left,
            h_right,
            wells=wells,
            flow_boundary=flow_boundary,
            step_idx=step_idx,
        )
        return spla.spsolve(mat, rhs).reshape(self.ny, self.nx)

    def compute_velocity(self, head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grad_y, grad_x = np.gradient(head, self.dy, self.dx)
        vx = -self.k_field * grad_x
        vy = -self.k_field * grad_y
        return vx, vy

    def source_vector(
        self,
        sources: Iterable[SourceTerm],
        *,
        step_idx: int | None = None,
    ) -> np.ndarray:
        values = np.zeros(self.n_nodes, dtype=float)
        for source in sources:
            amplitude = float(source.value) if step_idx is None else float(source.value_at_step(step_idx))
            if abs(amplitude) < 1e-30:
                continue
            values[self.idx(source.i, source.j)] += amplitude
        return values

    def well_vector(
        self,
        wells: Iterable[WellTerm],
        *,
        step_idx: int | None = None,
    ) -> np.ndarray:
        values = np.zeros(self.n_nodes, dtype=float)
        for well in wells:
            amplitude = float(well.value) if step_idx is None else float(well.value_at_step(step_idx))
            if abs(amplitude) < 1e-30:
                continue
            values[self.idx(well.i, well.j)] += amplitude
        return values

    def control_plane_columns(self, fractions: Iterable[float]) -> dict[str, int]:
        planes: dict[str, int] = {}
        for fraction in fractions:
            clipped = min(max(float(fraction), 0.0), 1.0)
            column = int(round(clipped * (self.nx - 1)))
            column = min(max(column, 0), self.nx - 1)
            name = f"cp{int(round(clipped * 100.0)):02d}"
            planes[name] = column
        return planes

    def solve_transport_with_diagnostics(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        *,
        steps: int,
        sources: Iterable[SourceTerm],
        record_stride: int = 1,
        diagnostics_fn: Callable[[float, np.ndarray], dict[str, float]] | None = None,
        include_initial: bool = True,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        step, _ = self.build_transport_operators(vx, vy)
        solve_step = spla.factorized(step)
        concentration = np.zeros(self.n_nodes, dtype=float)
        trace: list[dict[str, float]] = []
        stride = max(1, int(record_stride))

        if diagnostics_fn is not None and include_initial:
            trace.append(diagnostics_fn(0.0, concentration.reshape(self.ny, self.nx)))

        for step_idx in range(1, int(steps) + 1):
            source_vec = self.source_vector(sources, step_idx=step_idx)
            concentration = solve_step(concentration + self.dt * source_vec)
            should_record = step_idx == int(steps) or (step_idx % stride == 0)
            if diagnostics_fn is not None and should_record:
                trace.append(
                    diagnostics_fn(
                        step_idx * self.dt,
                        concentration.reshape(self.ny, self.nx),
                    )
                )

        return concentration.reshape(self.ny, self.nx), trace

    def solve_transport(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        *,
        steps: int,
        sources: Iterable[SourceTerm],
    ) -> np.ndarray:
        concentration, _ = self.solve_transport_with_diagnostics(
            vx,
            vy,
            steps=steps,
            sources=sources,
            diagnostics_fn=None,
        )
        return concentration
