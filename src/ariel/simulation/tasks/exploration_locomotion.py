"""Exploration + locomotion fitness utilities.

This module provides *scalar* fitness functions that combine the cell-based
exploration metrics in `ariel.simulation.tasks.exploration` with simple
locomotion cues derived from a planar trajectory `xy`.

All functions in this file are written for **maximisation** (higher is better),
consistent with `EASettings(is_maximisation=True)` in ARIEL's EA engine.

Notes
-----
- These are *not* part of the simulator; they are pure scoring functions.
- "Forward" is defined by a unit direction in the XY plane (default: +X).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ariel.simulation.tasks.exploration import (
    GridSpec,
    coverage_integral,
    fitness_f1_pure_coverage,
    fitness_f2_meaningful_coverage,
    path_efficiency,
)

# ---------------------------------------------------------------------------
# Displacement Metrics and Helpers
# ---------------------------------------------------------------------------

def _normalize_forward_dir(
    forward_xy: tuple[float, float] | None,
) -> np.ndarray:
    if forward_xy is None:
        return np.array([1.0, 0.0], dtype=float)
    v = np.asarray(forward_xy, dtype=float).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return (v / n).astype(float)


def net_forward_displacement_m(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    *,
    forward_xy: tuple[float, float] | None = None,
) -> float:
    """Signed forward displacement along a chosen axis: (p_end - p_start)·f̂."""
    arr = np.asarray(xy, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected xy as (T,2), got shape={arr.shape}")
    fhat = _normalize_forward_dir(forward_xy)
    p0 = arr[0, :2]
    p1 = arr[-1, :2]
    return float(np.dot(p1 - p0, fhat))


def mean_abs_forward_speed_m_s(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    *,
    dt: float,
    forward_xy: tuple[float, float] | None = None,
) -> float:
    """Average |v·f̂| using finite differences, consistent with `gait_learning` style."""
    if dt <= 0:
        return 0.0
    arr = np.asarray(xy, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected xy as (T,2), got shape={arr.shape}")
    if len(arr) < 2:
        return 0.0

    fhat = _normalize_forward_dir(forward_xy)
    diffs = np.diff(arr[:, :2], axis=0)
    v_par = diffs @ fhat
    return float(np.mean(np.abs(v_par) / float(dt)))


def planar_path_length_m(xy: Sequence[tuple[float, float]] | np.ndarray) -> float:
    """Total planar path length (sum of segment lengths)."""
    arr = np.asarray(xy, dtype=float)
    if len(arr) < 2:
        return 0.0
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected xy as (T,2), got shape={arr.shape}")
    d = np.diff(arr[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))


# ---------------------------------------------------------------------------
# Combined fitnesses (maximise)
# ---------------------------------------------------------------------------

def fitness_f1_plus_forward(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    *,
    forward_xy: tuple[float, float] | None = None,
    alpha: float = 0.2,
) -> float:
    """f1 (coverage) + α * max(0, forward displacement)."""
    f1 = float(fitness_f1_pure_coverage(N))
    d = float(max(0.0, net_forward_displacement_m(xy, forward_xy=forward_xy)))
    return float(f1 + float(alpha) * d)
####return float(float(1 - alpha) * f1 + float(alpha) * d)


def fitness_f2_plus_forward(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    *,
    forward_xy: tuple[float, float] | None = None,
    alpha: float = 0.2,
    lambda_redundancy: float = 0.2, #check for optimal values for lambda
) -> float:
    """f2 (cov - λR) + α * max(0, forward displacement)."""
    f2 = float(fitness_f2_meaningful_coverage(N, lambda_=float(lambda_redundancy)))
    d = float(max(0.0, net_forward_displacement_m(xy, forward_xy=forward_xy)))
    return float(f2 + float(alpha) * d)


def fitness_f1_plus_mean_forward_speed(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    *,
    dt: float,
    forward_xy: tuple[float, float] | None = None,
    beta: float = 0.05,
) -> float:
    """f1 + β * mean(|v·f̂|), where v is approximated by Δp/dt between samples."""
    f1 = float(fitness_f1_pure_coverage(N))
    vbar = float(mean_abs_forward_speed_m_s(xy, dt=float(dt), forward_xy=forward_xy))
    return float(f1 + float(beta) * vbar)


def fitness_scalarized_explore_locomote(
    N: np.ndarray,
    xy: Sequence[tuple[float, float]] | np.ndarray,
    *,
    forward_xy: tuple[float, float] | None = None,
    a: float = 0.5,
    b: float = 0.3,
    c: float = 0.2,
) -> float:
    """a*f1 + b*E + c*max(0, forward displacement), where E is path efficiency.

    E = |V| / sum N(c) (unique-cells per step) from `exploration.py`.
    """
    f1 = float(fitness_f1_pure_coverage(N))
    e = float(path_efficiency(N))
    d = float(max(0.0, net_forward_displacement_m(xy, forward_xy=forward_xy)))
    return float(float(a) * f1 + float(b) * e + float(c) * d)


def fitness_gated_f6_by_forward_speed(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    *,
    dt: float,
    v_min: float = 0.01,
    sample_every: int = 1,
    forward_xy: tuple[float, float] | None = None,
) -> float:
    """If mean forward speed is too low, return 0; else return coverage integral f6.

    This encodes: "only reward exploration over time if the robot is actually moving
    forward (not just vibrating in place)."
    """
    v = float(
        mean_abs_forward_speed_m_s(
            xy,
            dt=float(dt),
            forward_xy=forward_xy,
        )
    )
    if v < float(v_min):
        return 0.0
    return float(coverage_integral(xy, grid, sample_every=int(sample_every)))
