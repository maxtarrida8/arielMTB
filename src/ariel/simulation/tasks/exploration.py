"""Exploration metrics and fitness utilities (cell-based).

ARIEL terrains are continuous in MuJoCo world coordinates. For exploration-style
fitness functions we overlay a *grid* on the terrain and compute metrics based
on which grid cells were visited.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class GridSpec:
    """A rectangular grid overlay in world XY.

    Parameters
    ----------
    width_m, height_m
        Physical size of the area covered by the grid, in meters.
        For most ARIEL worlds (except `OlympicArena`), this is typically 10x10.
    nrow, ncol
        Grid resolution. Total cells = nrow*ncol.
    origin_xy
        World XY coordinate of the grid center. ARIEL floors are typically centered
        at (0, 0), so this default works for the standard worlds.
    """

    width_m: float
    height_m: float
    nrow: int
    ncol: int
    origin_xy: tuple[float, float] = (0.0, 0.0)

    @property
    def cell_size_x_m(self) -> float:
        return float(self.width_m) / max(int(self.ncol), 1)

    @property
    def cell_size_y_m(self) -> float:
        return float(self.height_m) / max(int(self.nrow), 1)

    @property
    def total_cells(self) -> int:
        return int(self.nrow) * int(self.ncol)

    def xy_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Map world XY to (row, col) with clamping.

        Notes
        -----
        - If (x,y) lies exactly on a cell boundary (including the world center),
          this mapping consistently assigns it to a single cell via clamping
          to [0, 1) and integer truncation.
        """
        ox, oy = self.origin_xy
        # Shift so that grid spans:
        # x in [ox - width/2, ox + width/2]
        # y in [oy - height/2, oy + height/2]
        u = (float(x) - ox) + float(self.width_m) / 2.0
        v = (float(y) - oy) + float(self.height_m) / 2.0

        # Normalize to [0, 1)
        fx = u / max(float(self.width_m), 1e-12)
        fy = v / max(float(self.height_m), 1e-12)

        # Clamp to [0, 1) to avoid out-of-bounds and resolve boundary points.
        eps = 1e-12
        fx = float(np.clip(fx, 0.0, 1.0 - eps))
        fy = float(np.clip(fy, 0.0, 1.0 - eps))

        col = int(fx * int(self.ncol))
        row = int(fy * int(self.nrow))
        # Extra safety clamps (in case ncol/nrow are 0 or strange)
        col = int(np.clip(col, 0, max(int(self.ncol) - 1, 0)))
        row = int(np.clip(row, 0, max(int(self.nrow) - 1, 0)))
        return row, col


# ---------------------------------------------------------------------------
# Trajectory -> cell visit statistics
# ---------------------------------------------------------------------------

def visit_counts_from_xy(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    """Return visit counts N for each cell from an XY trajectory.

    Parameters
    ----------
    xy
        Sequence of (x,y) samples in world coordinates.
    grid
        Grid specification for discretization.

    Returns
    -------
    np.ndarray
        Integer array with shape (nrow, ncol), where N[r,c] is the number of
        samples that landed in that cell.
    """
    N = np.zeros((int(grid.nrow), int(grid.ncol)), dtype=np.int64)
    if len(xy) == 0:
        return N

    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected xy as (T,2), got shape={arr.shape}")

    for x, y in arr:
        r, c = grid.xy_to_cell(float(x), float(y))
        N[r, c] += 1
    return N


def visited_set(N: np.ndarray) -> set[tuple[int, int]]:
    """Set of visited cells V from visit-count grid N."""
    if N.ndim != 2:
        raise ValueError("Expected N as 2D array")
    rs, cs = np.nonzero(N > 0)
    return {(int(r), int(c)) for r, c in zip(rs, cs, strict=True)}


# ---------------------------------------------------------------------------
# Metrics 
# ---------------------------------------------------------------------------

def coverage_fraction(N: np.ndarray) -> float:
    """cov(T) = |V_T| / |C|."""
    total_cells = int(N.size)
    if total_cells <= 0:
        return 0.0
    V = int(np.count_nonzero(N > 0))
    return float(V / total_cells)


def visited_cell_count(N: np.ndarray) -> int:
    """|V_T|."""
    return int(np.count_nonzero(N > 0))


def total_steps(N: np.ndarray) -> int:
    """Sum_c N(c). Interpretable as steps if you increment once per timestep sample."""
    return int(np.sum(N))


def redundancy_ratio(N: np.ndarray) -> float:
    """R(T) = sum_c max(0, N(c)-1) / sum_c N(c)."""
    denom = float(np.sum(N))
    if denom <= 0.0:
        return 0.0
    revisits = np.maximum(N - 1, 0)
    return float(np.sum(revisits) / denom)


def path_efficiency(N: np.ndarray) -> float:
    """E(T) = |V_T| / sum_c N(c)."""
    denom = float(np.sum(N))
    if denom <= 0.0:
        return 0.0
    return float(np.count_nonzero(N > 0) / denom)


def unknown_area(N: np.ndarray) -> int:
    """|U_T| where unknown := unvisited."""
    return int(np.count_nonzero(N == 0))


def exploration_gain(N: np.ndarray) -> float:
    """G(T) = (|U_0| - |U_T|) / |U_0|.

    With unknown := unvisited and U_0 = |C|, this reduces to coverage_fraction.
    """
    total_cells = int(N.size)
    if total_cells <= 0:
        return 0.0
    return float((total_cells - unknown_area(N)) / total_cells)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def frontier_cells(N: np.ndarray) -> set[tuple[int, int]]:
    """F_T: visited cells with at least one unvisited 4-neighbour."""
    if N.ndim != 2:
        raise ValueError("Expected N as 2D array")
    nrow, ncol = N.shape
    V = visited_set(N)
    F: set[tuple[int, int]] = set()
    for r, c in V:
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < nrow and 0 <= cc < ncol and N[rr, cc] == 0:
                F.add((r, c))
                break
    return F


def frontier_area(N: np.ndarray) -> int:
    """|F_T|."""
    return int(len(frontier_cells(N)))


def time_to_coverage(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    *,
    x_percent: float,
    dt: float = 1.0,
) -> float | None:
    """Smallest time t such that cov(t) >= x/100.

    Parameters
    ----------
    xy
        XY trajectory samples.
    grid
        Grid spec.
    x_percent
        Target coverage percentage in [0,100].
    dt
        Time step per XY sample (seconds). If you store tracker samples every N
        simulation steps, dt should match that effective interval.

    Returns
    -------
    float | None
        Time to reach the requested coverage, or None if never reached.
    """
    target = float(x_percent) / 100.0
    target = float(np.clip(target, 0.0, 1.0))
    if target <= 0.0:
        return 0.0
    if len(xy) == 0:
        return None

    N = np.zeros((int(grid.nrow), int(grid.ncol)), dtype=np.int64)
    V_count = 0
    total_cells = int(N.size)

    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected xy as (T,2), got shape={arr.shape}")

    visited_mask = np.zeros_like(N, dtype=bool)
    for k, (x, y) in enumerate(arr):
        r, c = grid.xy_to_cell(float(x), float(y))
        N[r, c] += 1
        if not visited_mask[r, c]:
            visited_mask[r, c] = True
            V_count += 1
            if (V_count / total_cells) >= target:
                return float((k + 1) * float(dt))
    return None


def coverage_integral(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    *,
    sample_every: int = 1,
) -> float:
    """Approximate (1/T) * int_0^T cov(t) dt by averaging cov over samples.

    This matches the Notion approximation:
        f6 ≈ (1/K) * sum_k cov(t_k)

    Parameters
    ----------
    sample_every
        Compute cov at every Nth sample to reduce compute cost.
    """
    if len(xy) == 0:
        return 0.0
    sample_every = max(int(sample_every), 1)

    N = np.zeros((int(grid.nrow), int(grid.ncol)), dtype=np.int64)
    visited_mask = np.zeros_like(N, dtype=bool)
    V_count = 0
    total_cells = int(N.size)
    cov_values: list[float] = []

    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected xy as (T,2), got shape={arr.shape}")

    for k, (x, y) in enumerate(arr):
        r, c = grid.xy_to_cell(float(x), float(y))
        N[r, c] += 1
        if not visited_mask[r, c]:
            visited_mask[r, c] = True
            V_count += 1
        if k % sample_every == 0:
            cov_values.append(float(V_count / total_cells))

    if not cov_values:
        return 0.0
    return float(np.mean(cov_values))


# ---------------------------------------------------------------------------
# Fitness functions 
# ---------------------------------------------------------------------------

def fitness_f1_pure_coverage(N: np.ndarray) -> float:
    """f1 = cov(T)."""
    return coverage_fraction(N)


def fitness_f2_meaningful_coverage(N: np.ndarray, *, lambda_: float) -> float:
    """f2 = cov(T) - lambda * R(T)."""
    return float(coverage_fraction(N) - float(lambda_) * redundancy_ratio(N))


def fitness_f3_unique_cells_per_step(N: np.ndarray) -> float:
    """f3 = |V_T| / sum_c N(c) = E(T)."""
    return path_efficiency(N)


def fitness_f4_unknown_area_reduction(N: np.ndarray) -> float:
    """f4 = (|U0| - |Ut|) / |U0|.

    With unknown := unvisited and U0 = |C|, this equals coverage_fraction.
    """
    return exploration_gain(N)


def fitness_f5_frontier_reduction(
    N0: np.ndarray,
    NT: np.ndarray,
) -> float:
    """f5 = (|F0| - |FT|) / |F0|.

    Notes
    -----
    This requires a reference frontier size at the start (F0). In practice you
    can set N0 from an initial short prefix of the trajectory (e.g. first few
    samples) to make the ratio meaningful.
    """
    f0 = float(frontier_area(N0))
    if f0 <= 0.0:
        return 0.0
    ft = float(frontier_area(NT))
    return float((f0 - ft) / f0)


def fitness_f6_coverage_integral(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
    *,
    sample_every: int = 1,
) -> float:
    """f6 ≈ (1/K) * sum_k cov(t_k)."""
    return coverage_integral(xy, grid, sample_every=sample_every)


def fitness_f8_convex_hull_coverage(
    xy: Sequence[tuple[float, float]] | np.ndarray,
    grid: GridSpec,
) -> float:
    """f8 = convex_hull_area(trajectory) / arena_area.

    Measures how broadly the robot spreads across the arena by computing the
    area of the convex hull of its XY trajectory, normalised by the total
    arena area so the score is in [0, 1].

    This is a direct anti-circling signal: a robot that circles stays inside
    a small hull (low score) regardless of how many steps it takes, while a
    robot that genuinely explores corners scores near 1.

    Parameters
    ----------
    xy : array-like of shape (T, 2)
        XY trajectory in world coordinates.
    grid : GridSpec
        Used only to derive the arena area (width_m * height_m).

    Returns
    -------
    float
        Hull area / arena area, clamped to [0, 1].
        Returns 0.0 if the trajectory has fewer than 3 non-collinear points.
    """
    arena_area = float(grid.width_m) * float(grid.height_m)
    if arena_area <= 0.0:
        return 0.0

    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return 0.0
    if len(arr) < 3:
        return 0.0

    # Remove duplicate points — QHull needs distinct positions
    arr = np.unique(arr, axis=0)
    if len(arr) < 3:
        return 0.0

    try:
        hull = ConvexHull(arr)
        hull_area = float(hull.volume)  # in 2D, .volume gives the area
    except QhullError:
        # All points collinear (robot moved in a straight line) — hull has 0 area
        return 0.0

    return float(np.clip(hull_area / arena_area, 0.0, 1.0))


def fitness_f7_coverage_efficiency(
    N: np.ndarray,
    *,
    alpha: float = 0.5,
) -> float:
    """f7 = alpha * cov(T) + (1 - alpha) * E(T).

    A parameter-free combination of coverage (f1) and path efficiency (f3)
    that rewards both visiting many cells *and* doing so without redundant
    revisits.  Unlike the ``scalarized`` fitness there is no forward-
    displacement term, so no preferred direction is imposed on the robot.

    Parameters
    ----------
    N : np.ndarray
        Cell visit-count grid (shape nrow × ncol).
    alpha : float
        Weight on coverage fraction (default 0.5).
        ``(1 - alpha)`` is applied to path efficiency.
        Both components are in [0, 1], so f7 is also in [0, 1].
    """
    cov = coverage_fraction(N)
    eff = path_efficiency(N)
    return float(alpha * cov + (1.0 - alpha) * eff)

