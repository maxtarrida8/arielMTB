"""Pipeline sanity check: force the gecko along a known path and verify f1, f2 & f3.

The robot is teleported along a predetermined square path by directly
overwriting the freejoint qpos each timestep.  Physics is still active
so the robot won't follow the exact intended path — but the fitness
pipeline should correctly compute f1, f2 and f3 for whatever trajectory
the tracker actually records.

f3 definition (entry-based):  |V_T| / sum(N)
  - N is incremented only on cell *transitions* (entry-based counting)
  - sum(N) = total cell entries = true measure of movement
  - f3 = 1.0 means every cell entry was a new cell (no re-entries)

f2 definition: cov(T) - lambda * R(T)
  - R(T) = redundancy ratio = re-entries / total entries
  - penalises revisiting cells

Usage:
    uv run examples/thesisMTB/gecko_experiments/pipeline_check.py
    uv run examples/thesisMTB/gecko_experiments/pipeline_check.py --viewer
"""

from __future__ import annotations

import argparse
from typing import cast

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Rectangle

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorldWalled
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _fitness_registry import ARENA_GRID  # noqa: E402

from ariel.simulation.tasks.exploration import (
    fitness_f1_pure_coverage,
    fitness_f2_meaningful_coverage,
    fitness_f3_unique_cells_per_step,
    redundancy_ratio,
    visit_counts_from_xy,
    planar_path_length_m,
)
from ariel.utils.tracker import Tracker

GRID = ARENA_GRID  # defined in _fitness_registry.py — change cell size there
DURATION = 30.0
SAVE_EVERY = 10

# Square path through cell centres (half-integer coords so the robot starts
# and turns in the middle of a cell, never on a boundary).
# With 10x10 grid over 10mx10m: cell centres are at ...-1.5, -0.5, 0.5, 1.5...
WAYPOINTS = [
    (0.5, 0.5),
    (3.5, 0.5),
    (3.5, 3.5),
    (0.5, 3.5),
    (0.5, 0.5),
]


def _build_path(waypoints: list[tuple[float, float]], total_time: float, dt: float) -> np.ndarray:
    n_steps = int(total_time / dt)
    segments = len(waypoints) - 1
    steps_per_seg = n_steps // segments

    points = []
    for i in range(segments):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        n = steps_per_seg if i < segments - 1 else n_steps - len(points)
        points.extend(
            [(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n) for t in range(n)]
        )
    return np.array(points, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline sanity check")
    parser.add_argument("--viewer", action="store_true", help="Launch MuJoCo viewer")
    args = parser.parse_args()

    # ---- Build world ----
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorldWalled(load_precompiled=False)
    world.spawn(gecko().spec, position=[0.5, 0.5, 0.1])
    model = cast(mujoco.MjModel, world.spec.compile())
    data = mujoco.MjData(model)

    # ---- Tracker (same setup as exploration_cpg.py) ----
    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )
    tracker.setup(world.spec, data)

    # ---- Precompute the path ----
    sim_dt = float(model.opt.timestep)
    path = _build_path(WAYPOINTS, DURATION, sim_dt)
    z_height = float(data.qpos[2])

    # ---- Controller: teleport robot along path + record via tracker ----
    step_idx = [0]

    def teleport_ctrl(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        idx = min(step_idx[0], len(path) - 1)
        data.qpos[0] = path[idx, 0]
        data.qpos[1] = path[idx, 1]
        data.qpos[2] = z_height

        t = int(np.ceil(data.time / sim_dt))
        if t % SAVE_EVERY == 0:
            tracker.update(data)

        step_idx[0] += 1

    mujoco.set_mjcb_control(teleport_ctrl)

    # ---- Run simulation ----
    if args.viewer:
        from mujoco import viewer
        viewer.launch(model=model, data=data)
    else:
        mujoco.mj_resetData(model, data)
        while data.time < DURATION:
            mujoco.mj_step(model, data, nstep=1)

    # ---- Extract trajectory from tracker ----
    xpos_history = tracker.history["xpos"]
    first_key = next(iter(xpos_history))
    traj = xpos_history[first_key]
    xy_tracked = [(float(p[0]), float(p[1])) for p in traj]

    # ---- Compute from tracker data ----
    N = visit_counts_from_xy(xy_tracked, GRID)
    xy_path = [(float(p[0]), float(p[1])) for p in path]
    N_intended = visit_counts_from_xy(xy_path, GRID)

    # f1: pipeline vs manual
    f1_pipeline = fitness_f1_pure_coverage(N)
    f1_manual = float(np.count_nonzero(N > 0)) / float(N.size)
    f1_intended = fitness_f1_pure_coverage(N_intended)

    # f2: coverage - lambda * redundancy  (lambda=0.5)
    f2_pipeline = fitness_f2_meaningful_coverage(N, lambda_=0.5)
    f2_intended = fitness_f2_meaningful_coverage(N_intended, lambda_=0.5)
    R_tracked = redundancy_ratio(N)
    R_intended = redundancy_ratio(N_intended)

    # f3: |V_T| / sum(N)  (path efficiency, entry-based)
    f3_pipeline = fitness_f3_unique_cells_per_step(N)
    unique_cells = int(np.count_nonzero(N > 0))
    total_entries = int(np.sum(N))
    f3_manual = float(unique_cells / total_entries) if total_entries > 0 else 0.0
    f3_intended = fitness_f3_unique_cells_per_step(N_intended)
    path_len_tracked = planar_path_length_m(xy_tracked)  # informational only

    # ---- Report ----
    print("=" * 60)
    print("PIPELINE SANITY CHECK")
    print("=" * 60)
    print(f"Predetermined path: {WAYPOINTS}")
    print(f"Duration: {DURATION}s, sim dt: {sim_dt}s")
    print(f"Path points: {len(path)}, Tracker samples: {len(xy_tracked)}")

    print()
    print("--- f1: pure coverage = |V_T| / |C| ---")
    print(f"  f1 (pipeline):             {f1_pipeline:.6f}")
    print(f"  f1 (manual = {unique_cells}/{N.size}):  {f1_manual:.6f}")
    print(f"  Pipeline == Manual:        {'YES' if abs(f1_pipeline - f1_manual) < 1e-9 else 'NO'}")
    print(f"  f1 (intended path):        {f1_intended:.6f}")

    print()
    print("--- f2: meaningful coverage = f1 - 0.5 * R(T) ---")
    print(f"  f2 (pipeline):             {f2_pipeline:.6f}")
    print(f"  Redundancy R(T):           {R_tracked:.6f}  ({R_tracked*100:.1f}% of entries were re-entries)")
    print(f"  f2 (intended path):        {f2_intended:.6f}  (R={R_intended:.4f})")

    print()
    print(f"--- f3: path efficiency = |V_T| / sum(N)  [entry-based] ---")
    print(f"  f3 (pipeline):             {f3_pipeline:.6f}")
    print(f"  f3 (manual = {unique_cells}/{total_entries}):  {f3_manual:.6f}")
    print(f"  Pipeline == Manual:        {'YES' if abs(f3_pipeline - f3_manual) < 1e-9 else 'NO'}")
    print(f"  f3 (intended path):        {f3_intended:.6f}")
    print(f"  [info] path length:        {path_len_tracked:.2f} m  (not used in f3 anymore)")

    print()
    print("Visit count grid (from tracker):")
    print(N)
    print()
    print("Visit count grid (intended path):")
    print(N_intended)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    hw = GRID.width_m / 2
    hh = GRID.height_m / 2
    cell_w = GRID.cell_size_x_m
    cell_h = GRID.cell_size_y_m

    for ax, xy_data, N_grid, title in [
        (axes[0], xy_tracked, N, f"Actual trajectory  f1={f1_pipeline:.3f}  f2={f2_pipeline:.3f}  f3={f3_pipeline:.3f}"),
        (axes[1], xy_path, N_intended, f"Intended path  f1={f1_intended:.3f}  f2={f2_intended:.3f}  f3={f3_intended:.3f}"),
    ]:
        arr = np.array(xy_data)

        # Shade visited cells
        for r in range(GRID.nrow):
            for c in range(GRID.ncol):
                if N_grid[r, c] > 0:
                    x0 = -hw + c * cell_w
                    y0 = -hh + r * cell_h
                    ax.add_patch(Rectangle((x0, y0), cell_w, cell_h,
                                           facecolor="cyan", alpha=0.3, edgecolor="none"))

        ax.plot(arr[:, 0], arr[:, 1], "b-", linewidth=1.0)
        ax.plot(arr[0, 0], arr[0, 1], "go", markersize=10, label="Start")
        ax.plot(arr[-1, 0], arr[-1, 1], "ro", markersize=10, label="End")

        # Draw grid lines
        for r in range(GRID.nrow + 1):
            y = -hh + r * cell_h
            ax.axhline(y, color="gray", linewidth=0.5)
        for c in range(GRID.ncol + 1):
            x = -hw + c * cell_w
            ax.axvline(x, color="gray", linewidth=0.5)

        # Label cell visit counts
        for r in range(GRID.nrow):
            for c in range(GRID.ncol):
                cx = -hw + (c + 0.5) * cell_w
                cy = -hh + (r + 0.5) * cell_h
                count = N_grid[r, c]
                if count > 0:
                    ax.text(cx, cy, str(count), ha="center", va="center",
                            fontsize=7, color="red", fontweight="bold")

        # Zoom to the region with data (plus 1 cell padding)
        visited_rows, visited_cols = np.nonzero(N_grid > 0)
        if len(visited_rows) > 0:
            r_min, r_max = visited_rows.min(), visited_rows.max()
            c_min, c_max = visited_cols.min(), visited_cols.max()
            x_lo = -hw + max(0, c_min - 1) * cell_w
            x_hi = -hw + min(GRID.ncol, c_max + 2) * cell_w
            y_lo = -hh + max(0, r_min - 1) * cell_h
            y_hi = -hh + min(GRID.nrow, r_max + 2) * cell_h
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

        ax.set_aspect("equal")
        ax.set_title(title)
        ax.legend()
        ax.grid(False)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
