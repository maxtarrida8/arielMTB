"""Tiny registry of exploration / exploration+locomotion fitnesses.

All entries are written for **maximisation** (higher is better) so they are
directly compatible with the existing exploration scripts (`is_maximisation=True`
for the EA script, and the Nevergrad script negates internally).

Public API:
    FITNESS_REGISTRY: dict[str, FitnessSpec]
    evaluate_fitness(name, *, N, xy, grid, dt, forward_xy) -> float

Each fitness in the registry declares which inputs it requires
(`needs_xy`, `needs_grid`, `needs_dt`, `needs_forward`); `evaluate_fitness`
dispatches with only the inputs the chosen fitness actually needs so callers
can pass the same kwargs every time.

Designed to be imported with a `sys.path` insert in each script:

    sys.path.insert(0, str(Path(__file__).parent))
    from _fitness_registry import (
        FITNESS_REGISTRY,
        evaluate_fitness,
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ariel.simulation.tasks.exploration import (
    GridSpec,
    fitness_f1_pure_coverage,
    fitness_f2_meaningful_coverage,
    fitness_f3_unique_cells_per_step,
    fitness_f4_unknown_area_reduction,
    fitness_f6_coverage_integral,
    fitness_f7_coverage_efficiency,
)
from ariel.simulation.tasks.exploration_locomotion import (
    fitness_f1_plus_forward,
    fitness_f1_plus_mean_forward_speed,
    fitness_f2_plus_forward,
    fitness_gated_f6_by_forward_speed,
    fitness_scalarized_explore_locomote,
)


@dataclass(frozen=True)
class FitnessSpec:
    """Metadata describing how to call a fitness function."""

    name: str
    description: str
    func: Callable[..., float]
    # Required inputs (passed via kwargs):
    needs_N: bool = True
    needs_xy: bool = False
    needs_grid: bool = False
    needs_dt: bool = False
    needs_forward: bool = False


def _f1(N: np.ndarray, **_: Any) -> float:
    return float(fitness_f1_pure_coverage(N))


def _f2(N: np.ndarray, **_: Any) -> float:
    return float(fitness_f2_meaningful_coverage(N, lambda_=0.5))


def _f3(N: np.ndarray, **_: Any) -> float:
    return float(fitness_f3_unique_cells_per_step(N))


def _f4(N: np.ndarray, **_: Any) -> float:
    return float(fitness_f4_unknown_area_reduction(N))


def _f6(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    **_: Any,
) -> float:
    return float(fitness_f6_coverage_integral(xy, grid))


def _f7(N: np.ndarray, **_: Any) -> float:
    return float(fitness_f7_coverage_efficiency(N, alpha=0.5))


def _f1_plus_forward(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    forward_xy: tuple[float, float] | None,
    **_: Any,
) -> float:
    return float(
        fitness_f1_plus_forward(N, xy, forward_xy=forward_xy, alpha=0.2)
    )


def _f1_plus_speed(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    dt: float,
    forward_xy: tuple[float, float] | None,
    **_: Any,
) -> float:
    return float(
        fitness_f1_plus_mean_forward_speed(
            N,
            xy,
            dt=float(dt),
            forward_xy=forward_xy,
            beta=0.05,
        )
    )


def _f2_plus_forward(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    forward_xy: tuple[float, float] | None,
    **_: Any,
) -> float:
    return float(
        fitness_f2_plus_forward(N, xy, forward_xy=forward_xy, alpha=0.2)
    )


def _scalarized(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    forward_xy: tuple[float, float] | None,
    **_: Any,
) -> float:
    return float(
        fitness_scalarized_explore_locomote(N, xy, forward_xy=forward_xy, a=0.5, b=0.3, c=0.2)
    )


def _f6_gated_speed(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    dt: float,
    forward_xy: tuple[float, float] | None,
    **_: Any,
) -> float:
    return float(
        fitness_gated_f6_by_forward_speed(
            xy,
            grid,
            dt=float(dt),
            v_min=0.01,
            forward_xy=forward_xy,
        )
    )


FITNESS_REGISTRY: dict[str, FitnessSpec] = {
    "f1": FitnessSpec(
        name="f1",
        description="Pure cell coverage: |V_T| / |C|.",
        func=_f1,
    ),
    "f2": FitnessSpec(
        name="f2",
        description="Meaningful coverage: cov(T) - 0.5 * R(T).",
        func=_f2,
    ),
    "f3": FitnessSpec(
        name="f3",
        description="Path efficiency: |V_T| / sum_c N(c).",
        func=_f3,
    ),
    "f4": FitnessSpec(
        name="f4",
        description="Unknown-area reduction (== coverage with U=unvisited).",
        func=_f4,
    ),
    "f6": FitnessSpec(
        name="f6",
        description="Coverage integral over time: (1/K) * sum_k cov(t_k).",
        func=_f6,
        needs_N=False,
        needs_xy=True,
        needs_grid=True,
    ),
    "f7": FitnessSpec(
        name="f7",
        description="Coverage + efficiency: 0.5 * cov(T) + 0.5 * E(T). No forward bias.",
        func=_f7,
    ),
    "f1_plus_forward": FitnessSpec(
        name="f1_plus_forward",
        description="f1 + 0.2 * max(0, forward displacement).",
        func=_f1_plus_forward,
        needs_xy=True,
        needs_forward=True,
    ),
    "f1_plus_speed": FitnessSpec(
        name="f1_plus_speed",
        description="f1 + 0.05 * mean(|v . f_hat|).",
        func=_f1_plus_speed,
        needs_xy=True,
        needs_dt=True,
        needs_forward=True,
    ),
    "f2_plus_forward": FitnessSpec(
        name="f2_plus_forward",
        description="f2 (cov - 0.5*redundancy) + 0.2 * max(0, forward displacement).",
        func=_f2_plus_forward,
        needs_xy=True,
        needs_forward=True,
    ),
    "scalarized": FitnessSpec(
        name="scalarized",
        description="0.5*f1 + 0.3*path_efficiency + 0.2*max(0, forward displacement).",
        func=_scalarized,
        needs_xy=True,
        needs_forward=True,
    ),
    "f6_gated_speed": FitnessSpec(
        name="f6_gated_speed",
        description="Coverage integral f6 only when mean(|v . f_hat|) >= 0.01 m/s; else 0.",
        func=_f6_gated_speed,
        needs_N=False,
        needs_xy=True,
        needs_grid=True,
        needs_dt=True,
        needs_forward=True,
    ),
}


def evaluate_fitness(
    name: str,
    *,
    N: np.ndarray | None = None,
    xy: Sequence[tuple[float, float]] | np.ndarray | None = None,
    grid: GridSpec | None = None,
    dt: float | None = None,
    forward_xy: tuple[float, float] | None = None,
) -> float:
    """Look up `name` in the registry and call it with the right kwargs.

    Callers can always pass every kwarg; only the ones the chosen fitness
    actually needs are forwarded. Missing required inputs raise ValueError.
    """
    if name not in FITNESS_REGISTRY:
        msg = (
            f"Unknown fitness '{name}'. "
            f"Available: {sorted(FITNESS_REGISTRY)}"
        )
        raise ValueError(msg)
    spec = FITNESS_REGISTRY[name]

    kwargs: dict[str, Any] = {}
    if spec.needs_N:
        if N is None:
            raise ValueError(f"Fitness '{name}' needs `N` (cell visit counts).")
        kwargs["N"] = N
    if spec.needs_xy:
        if xy is None:
            raise ValueError(f"Fitness '{name}' needs `xy` (trajectory).")
        kwargs["xy"] = xy
    if spec.needs_grid:
        if grid is None:
            raise ValueError(f"Fitness '{name}' needs `grid` (GridSpec).")
        kwargs["grid"] = grid
    if spec.needs_dt:
        if dt is None:
            raise ValueError(f"Fitness '{name}' needs `dt` (sample spacing in s).")
        kwargs["dt"] = dt
    if spec.needs_forward:
        kwargs["forward_xy"] = forward_xy

    return float(spec.func(**kwargs))


def list_fitness_names() -> list[str]:
    """Return registered fitness names (stable order)."""
    return list(FITNESS_REGISTRY.keys())
