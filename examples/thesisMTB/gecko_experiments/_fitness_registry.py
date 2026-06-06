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

# ---------------------------------------------------------------------------
# Shared grid definition — single source of truth.
# Change nrow/ncol here to resize cells consistently across all scripts.
# Current: 10 m x 10 m arena with 10x10 cells => 1 m x 1 m per cell.
# Example: nrow=20, ncol=20 => 0.5 m x 0.5 m cells (finer resolution).
# ---------------------------------------------------------------------------
from ariel.simulation.tasks.exploration import GridSpec

ARENA_GRID = GridSpec(width_m=10.0, height_m=10.0, nrow=10, ncol=10, origin_xy=(0.0, 0.0))

from ariel.simulation.tasks.exploration import (
    GridSpec,
    fitness_f1_pure_coverage,
    fitness_f2_meaningful_coverage,
    fitness_f3_unique_cells_per_step,
    fitness_f4_unknown_area_reduction,
    fitness_f6_coverage_integral,
    fitness_f7_coverage_efficiency,
    fitness_f8_convex_hull_coverage,
    fitness_f9_waypoint_proximity,
    fitness_f10_coverage_hull,
    fitness_f11_hull_efficiency,
    fitness_f12_hull_integral,
    fitness_f13_hull_cov_efficiency,
    fitness_f14_hull_times_coverage,
    fitness_f15_coverage_waypoint,
    fitness_f16_weighted_coverage_waypoint,
    fitness_f17_efficiency_waypoint,
    fitness_f18_unknown_waypoint,
    fitness_f19_integral_waypoint,
    fitness_f20_cov_efficiency_waypoint,
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
    needs_targets: bool = False


def _f1(N: np.ndarray, **_: Any) -> float:
    return float(fitness_f1_pure_coverage(N))


def _f2(N: np.ndarray, **_: Any) -> float:
    return float(fitness_f2_meaningful_coverage(N, lambda_=0.6))


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


def _f8(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    **_: Any,
) -> float:
    return float(fitness_f8_convex_hull_coverage(xy, grid))


def _f9(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    targets: Sequence[tuple[float, float]],
    **_: Any,
) -> float:
    return float(fitness_f9_waypoint_proximity(xy, targets, visit_radius=2.5))


def _f10(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    **_: Any,
) -> float:
    return float(fitness_f10_coverage_hull(N, xy, grid))


def _f11(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    **_: Any,
) -> float:
    return float(fitness_f11_hull_efficiency(N, xy, grid))


def _f12(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    **_: Any,
) -> float:
    return float(fitness_f12_hull_integral(xy, grid))


def _f13(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    **_: Any,
) -> float:
    return float(fitness_f13_hull_cov_efficiency(N, xy, grid))


def _f14(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    **_: Any,
) -> float:
    return float(fitness_f14_hull_times_coverage(N, xy, grid))


def _f15(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    targets: Sequence[tuple[float, float]],
    **_: Any,
) -> float:
    return float(fitness_f15_coverage_waypoint(N, xy, targets, visit_radius=2.5))


def _f16(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    targets: Sequence[tuple[float, float]],
    **_: Any,
) -> float:
    return float(fitness_f16_weighted_coverage_waypoint(N, xy, targets, alpha=0.5, visit_radius=2.5))


def _f17(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    targets: Sequence[tuple[float, float]],
    **_: Any,
) -> float:
    return float(fitness_f17_efficiency_waypoint(N, xy, targets, visit_radius=2.5))


def _f18(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    targets: Sequence[tuple[float, float]],
    **_: Any,
) -> float:
    return float(fitness_f18_unknown_waypoint(N, xy, targets, visit_radius=2.5))


def _f19(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    targets: Sequence[tuple[float, float]],
    **_: Any,
) -> float:
    return float(fitness_f19_integral_waypoint(xy, grid, targets, visit_radius=2.5))


def _f20(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    targets: Sequence[tuple[float, float]],
    **_: Any,
) -> float:
    return float(fitness_f20_cov_efficiency_waypoint(N, xy, targets, visit_radius=2.5))



FITNESS_REGISTRY: dict[str, FitnessSpec] = {
    "f1": FitnessSpec(
        name="f1",
        description="Pure cell coverage: |V_T| / |C|.",
        func=_f1,
    ),
    "f2": FitnessSpec(
        name="f2",
        description="Meaningful coverage: cov(T) - 0.9 * R(T).",
        func=_f2,
    ), 
    "f3": FitnessSpec(
        name="f3",
        description="Path efficiency: unique cells / total cell entries. Score in (0,1].",
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
    "f8": FitnessSpec(
        name="f8",
        description="Convex hull coverage: hull_area(trajectory) / arena_area. Direct anti-circling signal.",
        func=_f8,
        needs_N=False,
        needs_xy=True,
        needs_grid=True,
    ),
    "f9": FitnessSpec(
        name="f9",
        description="Waypoint proximity: mean smooth proximity to 4 targets (R=2.0m).",
        func=_f9,
        needs_N=False,
        needs_xy=True,
        needs_targets=True,
    ),
    "f10": FitnessSpec(
        name="f10",
        description="0.5 * f1 + 0.5 * f8. Cell coverage + convex hull spread.",
        func=_f10,
        needs_xy=True,
        needs_grid=True,
    ),
    "f11": FitnessSpec(
        name="f11",
        description="0.5 * f8 + 0.5 * f3. Convex hull spread + path efficiency.",
        func=_f11,
        needs_xy=True,
        needs_grid=True,
    ),
    "f12": FitnessSpec(
        name="f12",
        description="0.5 * f8 + 0.5 * f6. Convex hull spread + coverage integral over time.",
        func=_f12,
        needs_N=False,
        needs_xy=True,
        needs_grid=True,
    ),
    "f13": FitnessSpec(
        name="f13",
        description="0.5 * f8 + 0.5 * f7. Convex hull spread + coverage+efficiency composite.",
        func=_f13,
        needs_xy=True,
        needs_grid=True,
    ),
    "f14": FitnessSpec(
        name="f14",
        description="f1 * f8. Multiplicative gate: coverage reward scaled by hull spread.",
        func=_f14,
        needs_xy=True,
        needs_grid=True,
    ),
    "f15": FitnessSpec(
        name="f15",
        description="f1 + f9: coverage + waypoint proximity (additive).",
        func=_f15,
        needs_xy=True,
        needs_targets=True,
    ),
    "f16": FitnessSpec(
        name="f16",
        description="0.5*f1 + 0.5*f9: weighted coverage + waypoint proximity.",
        func=_f16,
        needs_xy=True,
        needs_targets=True,
    ),
    "f17": FitnessSpec(
        name="f17",
        description="f3 + f9: path efficiency + waypoint proximity.",
        func=_f17,
        needs_xy=True,
        needs_targets=True,
    ),
    "f18": FitnessSpec(
        name="f18",
        description="f4 + f9: unknown-area reduction + waypoint proximity.",
        func=_f18,
        needs_xy=True,
        needs_targets=True,
    ),
    "f19": FitnessSpec(
        name="f19",
        description="f6 + f9: coverage integral + waypoint proximity.",
        func=_f19,
        needs_N=False,
        needs_xy=True,
        needs_grid=True,
        needs_targets=True,
    ),
    "f20": FitnessSpec(
        name="f20",
        description="f7 + f9: coverage+efficiency + waypoint proximity.",
        func=_f20,
        needs_xy=True,
        needs_targets=True,
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
    targets: Sequence[tuple[float, float]] | None = None,
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
    if spec.needs_targets:
        if targets is None:
            raise ValueError(f"Fitness '{name}' needs `targets` (waypoint positions).")
        kwargs["targets"] = targets

    return float(spec.func(**kwargs))


def list_fitness_names() -> list[str]:
    """Return registered fitness names (stable order)."""
    return list(FITNESS_REGISTRY.keys())
