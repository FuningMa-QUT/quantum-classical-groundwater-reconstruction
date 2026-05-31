"""Truth-independent anchor placement strategies for hybrid quantum transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .operators import GroundwaterModel, SourceTerm, WellTerm


@dataclass
class AnchorPlacementResult:
    indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def _internal_indices(model: GroundwaterModel) -> list[int]:
    return [
        model.idx(i, j)
        for i in range(1, model.ny - 1)
        for j in range(1, model.nx - 1)
    ]


def _unique_indices(indices: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in indices:
        idx = int(value)
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def _source_influence_scores(
    model: GroundwaterModel,
    sources: Sequence[SourceTerm],
    *,
    vx: np.ndarray | None = None,
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(model.ny), np.arange(model.nx), indexing="ij")
    score = np.zeros((model.ny, model.nx), dtype=float)
    band = max(1.0, model.ny / 10.0)

    if not sources:
        score[1:-1, 1:-1] = 1.0
    else:
        for source in sources:
            downstream = np.maximum(xx - int(source.j), 0.0)
            cross_stream = (yy - int(source.i)) / band
            local = np.exp(-0.5 * cross_stream**2) / (1.0 + downstream)
            local[:, : int(source.j)] *= 0.2
            score = np.maximum(score, local)

    if vx is not None:
        positive_flux = np.maximum(np.asarray(vx, dtype=float), 0.0)
        flux_scale = float(np.mean(positive_flux[1:-1, 1:-1])) if positive_flux.size else 0.0
        score = score * (positive_flux + max(flux_scale, 1e-6))

    score[0, :] = 0.0
    score[-1, :] = 0.0
    score[:, 0] = 0.0
    score[:, -1] = 0.0
    return score


def _ranked_internal_candidates(
    model: GroundwaterModel,
    *,
    sources: Sequence[SourceTerm],
    vx: np.ndarray | None,
) -> list[int]:
    scores = _source_influence_scores(model, sources, vx=vx).reshape(-1)
    candidates = _internal_indices(model)
    return sorted(candidates, key=lambda idx: (scores[idx], -idx), reverse=True)


def _trace_pathline_cells(
    model: GroundwaterModel,
    source: SourceTerm,
    *,
    vx: np.ndarray,
    vy: np.ndarray,
    step_size_cells: float,
    max_steps: int,
) -> list[int]:
    x = float(source.j)
    y = float(source.i)
    cells: list[int] = []

    for _ in range(max_steps):
        j = int(round(x))
        i = int(round(y))
        if i <= 0 or i >= model.ny - 1 or j <= 0 or j >= model.nx - 1:
            break

        idx = model.idx(i, j)
        if not cells or idx != cells[-1]:
            cells.append(idx)

        u = float(vx[i, j])
        v = float(vy[i, j])
        speed = float(np.hypot(u, v))
        if speed < 1e-12:
            break

        x += step_size_cells * u / speed
        y += step_size_cells * v / speed

        if x <= 0.5 or x >= model.nx - 1.5 or y <= 0.5 or y >= model.ny - 1.5:
            break

    return _unique_indices(cells)


def _subsample_uniform(indices: Sequence[int], m: int) -> list[int]:
    ordered = _unique_indices(indices)
    if m <= 0 or not ordered:
        return []
    if len(ordered) <= m:
        return ordered
    picks = np.linspace(0, len(ordered) - 1, num=m)
    return [ordered[int(round(position))] for position in picks]


def _quotas(total: int, bins: int) -> list[int]:
    if bins <= 0:
        return []
    base = max(total, 0) // bins
    remainder = max(total, 0) % bins
    return [base + (1 if k < remainder else 0) for k in range(bins)]


def _supplement_indices(
    current: list[int],
    *,
    model: GroundwaterModel,
    m: int,
    sources: Sequence[SourceTerm],
    vx: np.ndarray | None,
    rng: np.random.Generator,
    random_fill: bool = False,
) -> list[int]:
    if len(current) >= m:
        return current[:m]

    ranked = _ranked_internal_candidates(model, sources=sources, vx=vx)
    used = set(current)
    supplemented = list(current)

    for idx in ranked:
        if idx not in used:
            supplemented.append(idx)
            used.add(idx)
        if len(supplemented) >= m:
            return supplemented

    if random_fill:
        remaining = [idx for idx in _internal_indices(model) if idx not in used]
        if remaining:
            count = min(m - len(supplemented), len(remaining))
            supplemented.extend(rng.choice(remaining, size=count, replace=False).tolist())

    return supplemented[:m]


def _pathline_monitoring_indices(
    model: GroundwaterModel,
    constraint_config: dict[str, Any],
    *,
    sources: Sequence[SourceTerm],
    vx: np.ndarray | None,
    vy: np.ndarray | None,
    rng: np.random.Generator,
) -> AnchorPlacementResult:
    m = int(constraint_config.get("m", 8))
    include_sources = bool(constraint_config.get("include_sources", True))
    step_size_cells = float(constraint_config.get("step_size_cells", 0.75))
    max_steps = int(constraint_config.get("max_path_steps", 4 * max(model.nx, model.ny)))

    selected: list[int] = []
    if include_sources:
        selected.extend(model.idx(source.i, source.j) for source in sources)

    traced_cells: list[int] = []
    if vx is not None and vy is not None and sources:
        paths = [
            _trace_pathline_cells(
                model,
                source,
                vx=vx,
                vy=vy,
                step_size_cells=step_size_cells,
                max_steps=max_steps,
            )
            for source in sources
        ]
        quotas = _quotas(max(m - len(selected), 0), len(paths))
        for quota, path in zip(quotas, paths):
            traced_cells.extend(_subsample_uniform(path, quota))

    selected.extend(traced_cells)
    selected = _supplement_indices(
        _unique_indices(selected),
        model=model,
        m=m,
        sources=sources,
        vx=vx,
        rng=rng,
        random_fill=True,
    )
    return AnchorPlacementResult(
        indices=np.asarray(selected, dtype=int),
        metadata={"strategy": "pathline_monitoring"},
    )


def _control_plane_monitoring_indices(
    model: GroundwaterModel,
    constraint_config: dict[str, Any],
    *,
    sources: Sequence[SourceTerm],
    vx: np.ndarray | None,
    plane_columns: dict[str, int] | None,
    rng: np.random.Generator,
) -> AnchorPlacementResult:
    del rng
    m = int(constraint_config.get("m", 8))
    fractions = constraint_config.get("plane_fractions", [0.25, 0.5, 0.75])
    planes = plane_columns or model.control_plane_columns(fractions)
    scores = _source_influence_scores(model, sources, vx=vx)
    selected: list[int] = []
    quotas = _quotas(m, max(1, len(planes)))

    for quota, column in zip(quotas, planes.values()):
        col = int(column)
        if col <= 0 or col >= model.nx - 1 or quota <= 0:
            continue
        candidates = [model.idx(i, col) for i in range(1, model.ny - 1)]
        ordered = sorted(
            candidates,
            key=lambda idx: scores[model.unravel(idx)],
            reverse=True,
        )
        selected.extend(ordered[:quota])

    selected = _supplement_indices(
        _unique_indices(selected),
        model=model,
        m=m,
        sources=sources,
        vx=vx,
        rng=np.random.default_rng(0),
        random_fill=False,
    )
    return AnchorPlacementResult(
        indices=np.asarray(selected, dtype=int),
        metadata={"strategy": "control_plane_monitoring", "n_planes": len(planes)},
    )


def _hybrid_monitoring_indices(
    model: GroundwaterModel,
    constraint_config: dict[str, Any],
    *,
    sources: Sequence[SourceTerm],
    vx: np.ndarray | None,
    vy: np.ndarray | None,
    plane_columns: dict[str, int] | None,
    rng: np.random.Generator,
) -> AnchorPlacementResult:
    m = int(constraint_config.get("m", 8))
    source_budget = min(len(sources), max(1, int(constraint_config.get("source_budget", max(1, m // 4)))))
    path_budget = max(1, int(constraint_config.get("pathline_budget", max(1, m // 2))))
    plane_budget = max(1, m - source_budget - path_budget)

    selected = [model.idx(source.i, source.j) for source in sources[:source_budget]]

    pathline = _pathline_monitoring_indices(
        model,
        {**constraint_config, "m": path_budget, "include_sources": False},
        sources=sources,
        vx=vx,
        vy=vy,
        rng=rng,
    ).indices.tolist()
    control = _control_plane_monitoring_indices(
        model,
        {**constraint_config, "m": plane_budget},
        sources=sources,
        vx=vx,
        plane_columns=plane_columns,
        rng=rng,
    ).indices.tolist()

    selected.extend(pathline)
    selected.extend(control)
    selected = _supplement_indices(
        _unique_indices(selected),
        model=model,
        m=m,
        sources=sources,
        vx=vx,
        rng=rng,
        random_fill=True,
    )
    return AnchorPlacementResult(
        indices=np.asarray(selected, dtype=int),
        metadata={"strategy": "source_control_plane_hybrid"},
    )


def _capture_monitoring_indices(
    model: GroundwaterModel,
    constraint_config: dict[str, Any],
    *,
    sources: Sequence[SourceTerm],
    flow_wells: Sequence[WellTerm] | None,
    vx: np.ndarray | None,
    vy: np.ndarray | None,
    rng: np.random.Generator,
) -> AnchorPlacementResult:
    m = int(constraint_config.get("m", 8))
    wells = list(flow_wells or [])
    capture_wells = [well for well in wells if float(well.value) < 0.0]
    if not capture_wells:
        capture_wells = wells
    if not capture_wells and sources:
        capture_wells = [
            WellTerm(i=int(source.i), j=int(source.j), value=-abs(float(source.value)), name=source.name)
            for source in sources
        ]

    include_wells = bool(constraint_config.get("include_capture_wells", True))
    ring_fraction = float(np.clip(constraint_config.get("capture_ring_budget_fraction", 0.5), 0.0, 1.0))
    corridor_fraction = float(
        np.clip(constraint_config.get("capture_corridor_budget_fraction", 0.35), 0.0, 1.0)
    )
    default_radii = [
        max(1, int(round(scale * max(model.nx, model.ny))))
        for scale in (0.06, 0.12, 0.2, 0.3)
    ]
    ring_radii = [int(radius) for radius in constraint_config.get("capture_ring_radii_cells", default_radii)]
    ring_radii = sorted({radius for radius in ring_radii if radius >= 1})
    angle_count = max(4, int(constraint_config.get("capture_ring_angle_count", 16)))

    selected: list[int] = []
    if include_wells:
        selected.extend(model.idx(well.i, well.j) for well in capture_wells)

    remaining = max(0, m - len(selected))
    ring_budget = min(remaining, max(0, int(round(remaining * ring_fraction))))
    corridor_budget = min(max(0, remaining - ring_budget), max(0, int(round(remaining * corridor_fraction))))

    ring_candidates: list[int] = []
    for well in capture_wells:
        for radius in ring_radii:
            for angle in np.linspace(0.0, 2.0 * np.pi, num=angle_count, endpoint=False):
                i = int(round(float(well.i) + float(radius) * float(np.sin(angle))))
                j = int(round(float(well.j) + float(radius) * float(np.cos(angle))))
                if 0 < i < model.ny - 1 and 0 < j < model.nx - 1:
                    ring_candidates.append(model.idx(i, j))
    selected.extend(_subsample_uniform(ring_candidates, ring_budget))

    corridor_candidates: list[int] = []
    if vx is not None and vy is not None and capture_wells:
        speed = np.hypot(np.asarray(vx, dtype=float), np.asarray(vy, dtype=float))
        yy, xx = np.meshgrid(np.arange(model.ny, dtype=float), np.arange(model.nx, dtype=float), indexing="ij")
        score = np.zeros((model.ny, model.nx), dtype=float)
        for well in capture_wells:
            distance = np.sqrt((yy - float(well.i)) ** 2 + (xx - float(well.j)) ** 2)
            local_score = speed / np.maximum(distance, 1.0)
            score = np.maximum(score, local_score)
        score[0, :] = 0.0
        score[-1, :] = 0.0
        score[:, 0] = 0.0
        score[:, -1] = 0.0
        candidates = _internal_indices(model)
        corridor_candidates = sorted(
            candidates,
            key=lambda idx: (score[model.unravel(idx)], -idx),
            reverse=True,
        )
    used = set(_unique_indices(selected))
    corridor_selected: list[int] = []
    for idx in corridor_candidates:
        if idx in used:
            continue
        corridor_selected.append(idx)
        used.add(idx)
        if len(corridor_selected) >= corridor_budget:
            break
    selected.extend(corridor_selected)

    selected = _supplement_indices(
        _unique_indices(selected),
        model=model,
        m=m,
        sources=sources,
        vx=vx,
        rng=rng,
        random_fill=True,
    )
    return AnchorPlacementResult(
        indices=np.asarray(selected, dtype=int),
        metadata={
            "strategy": "capture_monitoring",
            "n_capture_wells": len(capture_wells),
        },
    )


def select_anchor_indices(
    model: GroundwaterModel,
    constraint_config: dict[str, Any],
    *,
    reference_field: np.ndarray | None,
    sources: Sequence[SourceTerm],
    rng: np.random.Generator,
    vx: np.ndarray | None = None,
    vy: np.ndarray | None = None,
    plane_columns: dict[str, int] | None = None,
    flow_wells: Sequence[WellTerm] | None = None,
    phase_label: str | None = None,
) -> AnchorPlacementResult:
    constraint_type = str(constraint_config.get("type", "sources")).lower()

    if constraint_type == "corners":
        points = [
            (0, 0),
            (0, model.nx - 1),
            (model.ny - 1, 0),
            (model.ny - 1, model.nx - 1),
        ]
        indices = np.asarray([model.idx(i, j) for i, j in points], dtype=int)
        return AnchorPlacementResult(indices=indices, metadata={"strategy": "corners"})

    if constraint_type == "sources":
        indices = np.asarray([model.idx(i, j) for i, j in [(s.i, s.j) for s in sources]], dtype=int)
        return AnchorPlacementResult(indices=indices, metadata={"strategy": "sources"})

    if constraint_type == "custom_points":
        points = [(int(point["i"]), int(point["j"])) for point in constraint_config.get("points", [])]
        indices = np.asarray([model.idx(i, j) for i, j in points], dtype=int)
        return AnchorPlacementResult(indices=indices, metadata={"strategy": "custom_points"})

    if constraint_type == "random_internal":
        m = int(constraint_config.get("m", 8))
        candidates = _internal_indices(model)
        if m > len(candidates):
            raise ValueError(f"Requested {m} internal constraints, only {len(candidates)} available")
        indices = np.asarray(rng.choice(candidates, size=m, replace=False), dtype=int)
        return AnchorPlacementResult(indices=indices, metadata={"strategy": "random_internal"})

    if constraint_type == "random_active":
        if reference_field is None:
            raise ValueError("random_active requires a reference field for legacy truth-based selection")
        m = int(constraint_config.get("m", 8))
        threshold_fraction = float(constraint_config.get("threshold_fraction", 1e-4))
        flat = np.asarray(reference_field, dtype=float).reshape(-1)
        max_abs = float(np.max(np.abs(flat)))
        threshold = threshold_fraction * max_abs
        candidates = [
            model.idx(i, j)
            for i in range(1, model.ny - 1)
            for j in range(1, model.nx - 1)
            if abs(flat[model.idx(i, j)]) >= threshold
        ]
        if len(candidates) < m:
            ordered = sorted(_internal_indices(model), key=lambda idx: abs(flat[idx]), reverse=True)
            candidates = ordered[: max(m, len(candidates))]
        indices = np.asarray(rng.choice(candidates, size=m, replace=False), dtype=int)
        return AnchorPlacementResult(indices=indices, metadata={"strategy": "random_active"})

    if constraint_type == "pathline_monitoring":
        return _pathline_monitoring_indices(
            model,
            constraint_config,
            sources=sources,
            vx=vx,
            vy=vy,
            rng=rng,
        )

    if constraint_type == "control_plane_monitoring":
        return _control_plane_monitoring_indices(
            model,
            constraint_config,
            sources=sources,
            vx=vx,
            plane_columns=plane_columns,
            rng=rng,
        )

    if constraint_type in {"source_control_plane_hybrid", "hybrid_monitoring"}:
        return _hybrid_monitoring_indices(
            model,
            constraint_config,
            sources=sources,
            vx=vx,
            vy=vy,
            plane_columns=plane_columns,
            rng=rng,
        )

    if constraint_type in {"capture_monitoring", "pumpback_capture_monitoring"}:
        result = _capture_monitoring_indices(
            model,
            constraint_config,
            sources=sources,
            flow_wells=flow_wells,
            vx=vx,
            vy=vy,
            rng=rng,
        )
        if phase_label:
            result.metadata["phase_label"] = str(phase_label)
        return result

    raise ValueError(f"Unknown anchor/constraint type: {constraint_type}")
