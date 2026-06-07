"""Two diagnostic tests:

1. SPEED TEST — Load a real best_params.npz, run the gecko for several
   durations, measure actual path length and speed.  This tells us how
   far the robot walks per second so we can pick a good segment length.

2. FITNESS VERIFICATION — Synthetic trajectories with known properties
   to verify f21 and f24 score correctly and the deceptive landscape is fixed.

Usage:
    uv run examples/thesisMTB/gecko_experiments/test_speed_and_fitness.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import mujoco
import numpy as np

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorldWalledWithTargets
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker

sys.path.insert(0, str(Path(__file__).parent))
from _fitness_registry import ARENA_GRID  # noqa: E402

from ariel.simulation.tasks.exploration import (
    GridSpec,
    coverage_fraction,
    coverage_integral,
    fitness_f21_integral_meaningful,
    fitness_f24_meaningful_coverage_buffered,
    planar_path_length_m,
    redundancy_ratio,
    visit_counts_from_xy,
    visit_counts_from_xy_buffered,
)

GRID = ARENA_GRID
TARGETS = [(3.0, 3.0), (3.0, -3.0), (-3.0, -3.0), (-3.0, 3.0)]


def _xy_from_tracker(tracker: Tracker) -> list[tuple[float, float]]:
    xpos_history = tracker.history["xpos"]
    if not xpos_history:
        return []
    first_key = next(iter(xpos_history))
    traj = xpos_history[first_key]
    return [(float(p[0]), float(p[1])) for p in traj]


# =====================================================================
# TEST 1: Robot speed measurement
# =====================================================================
def test_speed() -> None:
    print("=" * 70)
    print("TEST 1: ROBOT SPEED MEASUREMENT")
    print("=" * 70)

    # Try to load a real best_params from a previous run
    # Pick the f10 run (best coverage result) or fall back to any available
    preferred = Path("__data__/exploration_cpg/runs/2026-06-05_16-46-51__f10_b200_d500/best_params.npz")
    if preferred.is_file():
        params_path = preferred
    else:
        candidates = sorted(Path("__data__/exploration_cpg/runs").glob("*/best_params.npz"))
        candidates = [c for c in candidates if "run_from_mac" not in str(c)]
        if not candidates:
            print("  No previous runs found. Skipping speed test.")
            return
        params_path = candidates[-1]
    print(f"  Loading params from: {params_path.parent.name}")

    loaded = np.load(params_path)
    params: dict[str, np.ndarray] = {k: loaded[k] for k in loaded.files}

    # Build world
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorldWalledWithTargets(load_precompiled=False, targets_xy=TARGETS)
    world.spawn(gecko().spec, position=[0.0, 0.0, 0.1])
    model = cast(mujoco.MjModel, world.spec.compile())
    data = mujoco.MjData(model)
    nu = int(model.nu)

    adj_dict = create_fully_connected_adjacency(nu)
    na_cpg = NaCPG(adj_dict, angle_tracking=False)

    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )
    ctrl = Controller(
        controller_callback_function=lambda _m, d, *a, **k: na_cpg.forward(float(d.time)),
        time_steps_per_ctrl_step=10,
        time_steps_per_save=10,
        alpha=1.0,
        tracker=tracker,
    )
    ctrl.tracker.setup(world.spec, data)

    # Run at multiple durations to measure speed
    durations = [30, 60, 120, 240, 500]
    print()
    print(f"  {'Duration':>10s}  {'Path(m)':>10s}  {'Speed(m/s)':>10s}  "
          f"{'Cells':>6s}  {'Cells/s':>8s}  {'Displace(m)':>12s}")
    print("  " + "-" * 66)

    for dur in durations:
        mujoco.mj_resetData(model, data)
        tracker.reset()
        na_cpg.set_param_with_dict(params)
        mujoco.set_mjcb_control(ctrl.set_control)
        simple_runner(model, data, duration=float(dur))

        xy = _xy_from_tracker(tracker)
        if len(xy) < 2:
            print(f"  {dur:>10d}  {'(no data)':>10s}")
            continue

        path_len = planar_path_length_m(xy)
        speed = path_len / dur

        N = visit_counts_from_xy(xy, GRID)
        n_cells = int(np.count_nonzero(N > 0))
        cells_per_s = n_cells / dur

        # Straight-line displacement from start to end
        start = np.array(xy[0])
        end = np.array(xy[-1])
        displacement = float(np.linalg.norm(end - start))

        print(f"  {dur:>10d}  {path_len:>10.2f}  {speed:>10.4f}  "
              f"{n_cells:>6d}  {cells_per_s:>8.4f}  {displacement:>12.2f}")

    # Compute cell size for reference
    print()
    print(f"  Cell size: {GRID.cell_size_x_m:.2f}m x {GRID.cell_size_y_m:.2f}m")
    print(f"  Arena: {GRID.width_m:.0f}m x {GRID.height_m:.0f}m, "
          f"{GRID.nrow}x{GRID.ncol} = {GRID.total_cells} cells")

    if len(durations) > 0 and len(xy) > 1:
        speed_30 = planar_path_length_m(_run_and_get_xy(
            model, data, na_cpg, ctrl, tracker, params, 30.0)) / 30.0
        cell_crossing_time = GRID.cell_size_x_m / max(speed_30, 1e-6)
        print()
        print(f"  At speed ~{speed_30:.4f} m/s:")
        print(f"    Time to cross 1 cell ({GRID.cell_size_x_m:.1f}m): "
              f"{cell_crossing_time:.1f}s")
        print(f"    Time to cross arena diagonal (~{GRID.width_m*1.41:.1f}m): "
              f"{GRID.width_m*1.41/max(speed_30,1e-6):.1f}s")
        print(f"    Suggested segment duration: "
              f"{max(cell_crossing_time * 3, 10):.0f}–{max(cell_crossing_time * 8, 30):.0f}s")


def _run_and_get_xy(model, data, na_cpg, ctrl, tracker, params, duration):
    mujoco.mj_resetData(model, data)
    tracker.reset()
    na_cpg.set_param_with_dict(params)
    mujoco.set_mjcb_control(ctrl.set_control)
    simple_runner(model, data, duration=duration)
    return _xy_from_tracker(tracker)


# =====================================================================
# TEST 2: f21 and f24 fitness verification
# =====================================================================
def test_fitness_functions() -> None:
    print()
    print("=" * 70)
    print("TEST 2: f21 AND f24 FITNESS VERIFICATION")
    print("=" * 70)

    grid = GridSpec(width_m=10.0, height_m=10.0, nrow=10, ncol=10, origin_xy=(0.0, 0.0))

    # --- Scenario A: Robot stays still ---
    xy_still = [(0.5, 0.5)] * 500
    f21_still = fitness_f21_integral_meaningful(xy_still, grid)
    f24_still = fitness_f24_meaningful_coverage_buffered(xy_still, grid)
    N_still = visit_counts_from_xy(xy_still, grid)

    print()
    print("  A) Robot stays still (500 samples at one cell)")
    print(f"     Coverage: {coverage_fraction(N_still):.4f}  ({int(np.count_nonzero(N_still))}/100 cells)")
    print(f"     f24 = {f24_still:.6f}")
    print(f"     f21 = {f21_still:.6f}")

    # --- Scenario B: Gait oscillation between 2 cells ---
    xy_osc2 = []
    for i in range(500):
        xy_osc2.append((0.5, 0.5) if i % 2 == 0 else (1.5, 0.5))
    f21_osc2 = fitness_f21_integral_meaningful(xy_osc2, grid)
    f24_osc2 = fitness_f24_meaningful_coverage_buffered(xy_osc2, grid)

    print()
    print("  B) Gait oscillation between 2 cells (500 samples)")
    print(f"     f24 = {f24_osc2:.6f}")
    print(f"     f21 = {f21_osc2:.6f}")
    print(f"     f24(osc) > f24(still)? {f24_osc2 > f24_still} "
          f"  ({f24_osc2:.6f} > {f24_still:.6f}) <- MUST BE TRUE")
    print(f"     f21(osc) > f21(still)? {f21_osc2 > f21_still} "
          f"  ({f21_osc2:.6f} > {f21_still:.6f}) <- MUST BE TRUE")

    # --- Scenario C: Straight walk covering 10 cells ---
    xy_straight = [(float(i) * 0.05, 0.5) for i in range(500)]  # 0 to 25m, but clamped
    f21_straight = fitness_f21_integral_meaningful(xy_straight, grid)
    f24_straight = fitness_f24_meaningful_coverage_buffered(xy_straight, grid)
    N_straight = visit_counts_from_xy(xy_straight, grid)
    N_buf_straight = visit_counts_from_xy_buffered(xy_straight, grid, buffer_size=4)

    print()
    print("  C) Straight walk (500 samples, crossing ~10 cells)")
    print(f"     Cells visited: {int(np.count_nonzero(N_straight))}")
    print(f"     R (regular):  {redundancy_ratio(N_straight):.4f}")
    print(f"     R (buffered): {redundancy_ratio(N_buf_straight):.4f}")
    print(f"     f24 = {f24_straight:.6f}")
    print(f"     f21 = {f21_straight:.6f}")
    print(f"     f24(straight) > f24(still)? {f24_straight > f24_still}  <- MUST BE TRUE")
    print(f"     f21(straight) > f21(still)? {f21_straight > f21_still}  <- MUST BE TRUE")

    # --- Scenario D: Walk out and back (large-scale backtracking) ---
    xy_outback = [(float(i) * 0.05, 0.5) for i in range(250)]
    xy_outback += [(float(250 - i) * 0.05, 0.5) for i in range(250)]
    f21_outback = fitness_f21_integral_meaningful(xy_outback, grid)
    f24_outback = fitness_f24_meaningful_coverage_buffered(xy_outback, grid)
    N_buf_outback = visit_counts_from_xy_buffered(xy_outback, grid, buffer_size=4)

    print()
    print("  D) Walk out 250 samples, walk back 250 (backtracking)")
    print(f"     R (buffered): {redundancy_ratio(N_buf_outback):.4f}")
    print(f"     f24 = {f24_outback:.6f}")
    print(f"     f21 = {f21_outback:.6f}")
    print(f"     f24(straight) > f24(outback)? {f24_straight > f24_outback}  "
          f"<- SHOULD BE TRUE (backtracking penalized)")
    print(f"     f21(straight) > f21(outback)? {f21_straight > f21_outback}  "
          f"<- SHOULD BE TRUE (backtracking penalized)")

    # --- Scenario E: Good exploration — visit 4 quadrants ---
    xy_explore = []
    # NE quadrant
    xy_explore += [(float(i) * 0.05 + 0.5, 0.5) for i in range(100)]
    # SE quadrant
    xy_explore += [(4.5, float(i) * 0.05 - 4.5) for i in range(100)]
    # SW quadrant
    xy_explore += [(4.5 - float(i) * 0.05, -4.5) for i in range(100)]
    # NW quadrant
    xy_explore += [(-4.5, -4.5 + float(i) * 0.05) for i in range(100)]
    # Back to center
    xy_explore += [(-4.5 + float(i) * 0.05, 0.5) for i in range(100)]

    f21_explore = fitness_f21_integral_meaningful(xy_explore, grid)
    f24_explore = fitness_f24_meaningful_coverage_buffered(xy_explore, grid)
    N_explore = visit_counts_from_xy(xy_explore, grid)

    print()
    print("  E) Good exploration — visit 4 quadrants (500 samples)")
    print(f"     Cells visited: {int(np.count_nonzero(N_explore))}")
    print(f"     f24 = {f24_explore:.6f}")
    print(f"     f21 = {f21_explore:.6f}")
    print(f"     f21(explore) > f21(straight)? {f21_explore > f21_straight}  "
          f"<- SHOULD BE TRUE (better exploration)")

    # --- Summary ---
    print()
    print("  SUMMARY — ranking (higher is better):")
    results = [
        ("A still", f21_still, f24_still),
        ("B osc-2", f21_osc2, f24_osc2),
        ("C straight", f21_straight, f24_straight),
        ("D out+back", f21_outback, f24_outback),
        ("E explore", f21_explore, f24_explore),
    ]
    print(f"  {'Scenario':>12s}  {'f21':>10s}  {'f24':>10s}")
    print("  " + "-" * 36)
    for name, f21, f24 in sorted(results, key=lambda r: r[1], reverse=True):
        print(f"  {name:>12s}  {f21:>10.6f}  {f24:>10.6f}")

    # Verify key invariants
    print()
    all_ok = True
    checks = [
        ("Movement > still (f24)", f24_osc2 > f24_still),
        ("Movement > still (f21)", f21_osc2 > f21_still),
        ("Straight > still (f24)", f24_straight > f24_still),
        ("Straight > backtrack (f24)", f24_straight > f24_outback),
        ("Explore > straight (f21)", f21_explore > f21_straight),
    ]
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {label}")
    print()
    print(f"  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")


def test_speed_comparison() -> None:
    """Compare speed of forward-displacement runs vs non-forward runs."""
    print()
    print("=" * 70)
    print("TEST 3: FORWARD vs NON-FORWARD SPEED COMPARISON")
    print("=" * 70)

    forward_runs = [
        ("fwd1: f1+fwd b500 30x30", Path("__data__/exploration_cpg/runs/run_from_mac/runswithforward/1/best_params.npz")),
        ("fwd2: f2+fwd b500", Path("__data__/exploration_cpg/runs/run_from_mac/runswithforward/2/best_params.npz")),
        ("fwd3: f1+fwd 10x10", Path("__data__/exploration_cpg/runs/run_from_mac/runswithforward/3/best_params.npz")),
        ("fwd4: f2+fwd seed1 10x10", Path("__data__/exploration_cpg/runs/run_from_mac/runswithforward/4/best_params.npz")),
        ("f1_fwd b1000 d240", Path("__data__/exploration_cpg/runs/2026-05-10_14-29-21__f1_forward_b1000_dur240/best_params.npz")),
    ]
    nonfwd_runs = [
        ("f10 b200 d500", Path("__data__/exploration_cpg/runs/2026-06-05_16-46-51__f10_b200_d500/best_params.npz")),
        ("f4 b600 walled", Path("__data__/exploration_cpg/runs/2026-05-28_18-48-44__f4_b600_walled/best_params.npz")),
        ("f3 b3000 walled", Path("__data__/exploration_cpg/runs/2026-05-30_10-56-25__f3_b3000_walled/best_params.npz")),
        ("f1 trial walled", Path("__data__/exploration_cpg/runs/2026-05-27_03-54-08__f1_trial_walled/best_params.npz")),
    ]

    # Build world once
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorldWalledWithTargets(load_precompiled=False, targets_xy=TARGETS)
    world.spawn(gecko().spec, position=[0.0, 0.0, 0.1])
    model = cast(mujoco.MjModel, world.spec.compile())
    data = mujoco.MjData(model)
    nu = int(model.nu)

    adj_dict = create_fully_connected_adjacency(nu)
    na_cpg = NaCPG(adj_dict, angle_tracking=False)

    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )
    ctrl = Controller(
        controller_callback_function=lambda _m, d, *a, **k: na_cpg.forward(float(d.time)),
        time_steps_per_ctrl_step=10,
        time_steps_per_save=10,
        alpha=1.0,
        tracker=tracker,
    )
    ctrl.tracker.setup(world.spec, data)

    dur = 120.0  # standard test duration

    print()
    print(f"  All runs simulated for {dur:.0f}s")
    print()
    print(f"  {'Run':>30s}  {'Path(m)':>8s}  {'m/s':>8s}  {'Cells':>6s}  "
          f"{'Displ(m)':>9s}  {'c/s':>6s}")
    print("  " + "-" * 75)

    def _measure(label: str, params_path: Path) -> None:
        if not params_path.is_file():
            print(f"  {label:>30s}  (file not found)")
            return
        try:
            loaded = np.load(params_path)
            params = {k: loaded[k] for k in loaded.files}
        except (PermissionError, Exception) as e:
            print(f"  {label:>30s}  (error: {e})")
            return

        mujoco.mj_resetData(model, data)
        tracker.reset()
        na_cpg.set_param_with_dict(params)
        mujoco.set_mjcb_control(ctrl.set_control)
        simple_runner(model, data, duration=dur)

        xy = _xy_from_tracker(tracker)
        if len(xy) < 2:
            print(f"  {label:>30s}  (no trajectory data)")
            return

        path_len = planar_path_length_m(xy)
        speed = path_len / dur
        N = visit_counts_from_xy(xy, GRID)
        n_cells = int(np.count_nonzero(N > 0))
        start = np.array(xy[0])
        end = np.array(xy[-1])
        displacement = float(np.linalg.norm(end - start))
        cells_per_s = n_cells / dur

        print(f"  {label:>30s}  {path_len:>8.2f}  {speed:>8.4f}  {n_cells:>6d}  "
              f"{displacement:>9.2f}  {cells_per_s:>6.3f}")

    print("  --- WITH forward displacement ---")
    for label, path in forward_runs:
        _measure(label, path)

    print()
    print("  --- WITHOUT forward displacement ---")
    for label, path in nonfwd_runs:
        _measure(label, path)

    print()
    print(f"  Cell size: {GRID.cell_size_x_m:.2f}m x {GRID.cell_size_y_m:.2f}m")


if __name__ == "__main__":
    test_speed()
    test_fitness_functions()
    test_speed_comparison()
