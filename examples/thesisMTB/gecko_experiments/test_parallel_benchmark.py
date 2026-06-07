"""Benchmark: sequential vs parallel MuJoCo eval + optimal worker sweep.

Compares the current sequential eval loop against batched ProcessPoolExecutor
parallelism for both steered and segmented architectures. Also sweeps worker
counts to find the throughput-optimal batch size for each.

Usage:
    uv run examples/thesisMTB/gecko_experiments/test_parallel_benchmark.py
    uv run ... --dur 15 --evals 12 --fitness f1
    uv run ... --arch steered
    uv run ... --arch segmented
    uv run ... --workers 1,2,4,8,12
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import torch

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorldWalledWithTargets
from ariel.simulation.tasks.exploration import visit_counts_from_xy
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker

sys.path.insert(0, str(Path(__file__).parent))
from _fitness_registry import ARENA_GRID, evaluate_fitness  # noqa: E402

TARGETS: list[tuple[float, float]] = [(3.0, 3.0), (3.0, -3.0), (-3.0, -3.0), (-3.0, 3.0)]
HALF_ARENA = ARENA_GRID.width_m / 2.0
NU = 8


# ---------------------------------------------------------------------------
# Shared factory - called inside each worker subprocess, never at import time
# ---------------------------------------------------------------------------
def _build_worker_objects() -> (
    tuple[mujoco.MjModel, mujoco.MjData, NaCPG, Tracker, Controller, float]
):
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
    sample_dt = float(model.opt.timestep) * float(ctrl.time_steps_per_save)
    return model, data, na_cpg, tracker, ctrl, sample_dt


def _xy_from_tracker(tracker: Tracker) -> list[tuple[float, float]]:
    xpos_history = tracker.history["xpos"]
    if not xpos_history:
        return []
    first_key = next(iter(xpos_history))
    return [(float(p[0]), float(p[1])) for p in xpos_history[first_key]]


# ---------------------------------------------------------------------------
# TOP-LEVEL picklable worker functions (must be module-level, no closures)
# ---------------------------------------------------------------------------

def _steered_worker(job: dict) -> float:
    """Evaluate one steered CPG candidate; builds its own MuJoCo context."""
    params = {k: np.asarray(v) for k, v in job["params"].items()}
    fitness_name: str = job["fitness_name"]
    duration: float = job["duration"]

    model, data, na_cpg, tracker, ctrl, sample_dt = _build_worker_objects()

    b_turn_np = params.pop("b_turn")
    wt_raw = params.pop("wall_threshold")
    wall_threshold = float(wt_raw.item() if hasattr(wt_raw, "item") else wt_raw)
    boundary = HALF_ARENA - wall_threshold

    na_cpg.set_param_with_dict(params)
    b_default = torch.from_numpy(np.array(params["b"])).float()
    b_turn_t = torch.from_numpy(np.array(b_turn_np)).float()

    # Closures created inside the worker (in subprocess) are fine - never pickled
    _nacpg, _bd, _bt, _bnd = na_cpg, b_default, b_turn_t, boundary

    def _callback(_model: mujoco.MjModel, d: mujoco.MjData, *_a, **_kw) -> None:
        x, y = float(d.qpos[0]), float(d.qpos[1])
        if abs(x) > _bnd or abs(y) > _bnd:
            _nacpg.set_params_by_group("b", _bt)
        else:
            _nacpg.set_params_by_group("b", _bd)
        return _nacpg.forward(float(d.time))

    ctrl.controller_callback_function = _callback
    mujoco.set_mjcb_control(ctrl.set_control)
    simple_runner(model, data, duration=duration)

    xy = _xy_from_tracker(tracker)
    N = visit_counts_from_xy(xy, ARENA_GRID)
    return evaluate_fitness(
        fitness_name, N=N, xy=xy, grid=ARENA_GRID,
        dt=sample_dt, forward_xy=(1.0, 0.0), targets=TARGETS,
    )


def _segmented_worker(job: dict) -> float:
    """Evaluate one segmented CPG candidate; builds its own MuJoCo context."""
    params = {k: np.asarray(v) for k, v in job["params"].items()}
    fitness_name: str = job["fitness_name"]
    duration: float = job["duration"]
    n_segments: int = job["n_segments"]

    model, data, na_cpg, tracker, ctrl, sample_dt = _build_worker_objects()

    base = {k: v for k, v in params.items() if not k.startswith("b_")}
    seg_dict = {int(k.split("_", 1)[1]): v for k, v in params.items() if k.startswith("b_")}
    segment_bs = [
        torch.from_numpy(np.asarray(seg_dict[i])).float() for i in range(n_segments)
    ]

    na_cpg.set_param_with_dict(base)
    mujoco.set_mjcb_control(ctrl.set_control)

    segment_duration = duration / n_segments
    for b_k in segment_bs:
        na_cpg.set_params_by_group("b", b_k)
        end_time = data.time + segment_duration
        while data.time < end_time:
            mujoco.mj_step(model, data, nstep=100)

    xy = _xy_from_tracker(tracker)
    N = visit_counts_from_xy(xy, ARENA_GRID)
    return evaluate_fitness(
        fitness_name, N=N, xy=xy, grid=ARENA_GRID,
        dt=sample_dt, forward_xy=(1.0, 0.0), targets=TARGETS,
    )


# ---------------------------------------------------------------------------
# Random param generators (timing-only - fitness quality irrelevant here)
# ---------------------------------------------------------------------------

def _make_steered_jobs(n: int, fitness_name: str, duration: float) -> list[dict]:
    rng = np.random.default_rng(42)
    return [
        {
            "fitness_name": fitness_name,
            "duration": duration,
            "params": {
                "phase":          rng.uniform(-2 * np.pi, 2 * np.pi, NU).tolist(),
                "w":              rng.uniform(-2 * np.pi, 2 * np.pi, NU).tolist(),
                "amplitudes":     rng.uniform(-2 * np.pi, 2 * np.pi, NU).tolist(),
                "ha":             rng.uniform(-10.0, 10.0, NU).tolist(),
                "b":              rng.uniform(-100.0, 100.0, NU).tolist(),
                "b_turn":         rng.uniform(-100.0, 100.0, NU).tolist(),
                "wall_threshold": float(rng.uniform(0.5, 4.0)),
            },
        }
        for _ in range(n)
    ]


def _make_segmented_jobs(
    n: int, n_segments: int, fitness_name: str, duration: float,
) -> list[dict]:
    rng = np.random.default_rng(42)
    jobs = []
    for _ in range(n):
        params: dict = {
            "phase":      rng.uniform(-2 * np.pi, 2 * np.pi, NU).tolist(),
            "w":          rng.uniform(-2 * np.pi, 2 * np.pi, NU).tolist(),
            "amplitudes": rng.uniform(-2 * np.pi, 2 * np.pi, NU).tolist(),
            "ha":         rng.uniform(-10.0, 10.0, NU).tolist(),
        }
        for k in range(n_segments):
            params[f"b_{k}"] = rng.uniform(-100.0, 100.0, NU).tolist()
        jobs.append({
            "fitness_name": fitness_name,
            "duration": duration,
            "n_segments": n_segments,
            "params": params,
        })
    return jobs


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _bench_sequential(worker_fn, jobs: list[dict]) -> float:
    """Run jobs one-by-one in the main process; return wall-clock seconds."""
    t0 = time.perf_counter()
    for job in jobs:
        worker_fn(job)
    return time.perf_counter() - t0


def _bench_parallel(worker_fn, jobs: list[dict], n_workers: int) -> float:
    """Submit all jobs to a ProcessPoolExecutor; return wall-clock seconds."""
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(worker_fn, jobs))
    return time.perf_counter() - t0


def _sweep(
    worker_fn, jobs: list[dict], counts: list[int],
) -> dict[int, float]:
    results: dict[int, float] = {}
    for n in counts:
        print(f"    workers={n:>3d} ...", end="", flush=True)
        wall = _bench_parallel(worker_fn, jobs, n_workers=n)
        results[n] = wall
        print(f"  {wall:.1f}s")
    return results


def _print_table(
    arch_name: str,
    n_evals: int,
    dur: float,
    seq_time: float,
    sweep: dict[int, float],
) -> None:
    print(f"\n  {'Workers':>7s}  {'Wall(s)':>8s}  {'Evals/min':>10s}  {'Speedup':>8s}")
    print("  " + "-" * 42)
    # Sequential row
    seq_tp = n_evals / seq_time * 60
    print(f"  {'seq':>7s}  {seq_time:>8.1f}  {seq_tp:>10.2f}  {'1.00x':>8s}")
    # Parallel rows
    for n_w in sorted(sweep):
        wall = sweep[n_w]
        tp = n_evals / wall * 60
        speedup = seq_time / wall
        print(f"  {n_w:>7d}  {wall:>8.1f}  {tp:>10.2f}  {speedup:>7.2f}x")
    best_n = min(sweep, key=sweep.__getitem__)
    best_wall = sweep[best_n]
    print(
        f"\n  Optimal: {best_n} workers -> {best_wall:.1f}s  "
        f"({seq_time / best_wall:.1f}x over sequential, "
        f"{n_evals / best_wall * 60:.2f} evals/min)"
    )
    print(
        f"  Note: sequential was {seq_time / n_evals:.1f}s/eval  "
        f"(parallel at {best_n}w: {best_wall / n_evals:.1f}s/eval effective)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential vs parallel eval benchmark")
    parser.add_argument("--dur", type=float, default=15.0,
                        help="Simulation seconds per eval (default: 15)")
    parser.add_argument("--evals", type=int, default=12,
                        help="Evaluations per benchmark run (default: 12)")
    parser.add_argument("--fitness", type=str, default="f1")
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--arch", type=str, default="both",
                        choices=["steered", "segmented", "both"])
    parser.add_argument("--workers", type=str, default=None,
                        help="Comma-separated worker counts (default: 1,2,4,6,8,10,<ncpu>)")
    args = parser.parse_args()

    n_cpu = os.cpu_count() or 4
    if args.workers:
        counts = sorted({int(x) for x in args.workers.split(",")})
    else:
        counts = sorted({1, 2, 4, 6, 8, 10, n_cpu} - {c for c in {6, 8, 10} if c > n_cpu})

    n_evals = args.evals
    dur = args.dur
    fitness_name = args.fitness

    print(f"Machine CPUs : {n_cpu}")
    print(f"Eval duration: {dur}s  |  N evals: {n_evals}  |  Fitness: {fitness_name}")
    print(f"Worker sweep : {counts}")
    print(f"Estimated sequential time per arch: ~{n_evals * dur:.0f}s (wall clock)")

    archs: list[tuple[str, object, list[dict]]] = []
    if args.arch in ("steered", "both"):
        archs.append(("STEERED", _steered_worker, _make_steered_jobs(n_evals, fitness_name, dur)))
    if args.arch in ("segmented", "both"):
        n_seg = args.segments
        archs.append((
            f"SEGMENTED (K={n_seg})",
            _segmented_worker,
            _make_segmented_jobs(n_evals, n_seg, fitness_name, dur),
        ))

    for arch_name, worker_fn, jobs in archs:
        print(f"\n{'='*60}")
        print(f"  {arch_name}")
        print(f"{'='*60}")

        print(f"\n  [1/2] Sequential ({n_evals} evals x {dur:.0f}s ~{n_evals*dur:.0f}s)...")
        seq_time = _bench_sequential(worker_fn, jobs)
        print(f"  Done: {seq_time:.1f}s  ({n_evals/seq_time*60:.2f} evals/min)")

        print(f"\n  [2/2] Parallel worker sweep (same {n_evals} jobs each)...")
        sweep = _sweep(worker_fn, jobs, counts)

        _print_table(arch_name, n_evals, dur, seq_time, sweep)

    print(f"\n{'='*60}")
    print("  Benchmark complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
