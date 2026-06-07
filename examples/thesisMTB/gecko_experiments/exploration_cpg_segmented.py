"""GECKO exploration with time-segmented CPG steering.

The simulation is split into K segments.  The optimiser finds one base gait
(phase, w, amplitudes, ha — 32 params) plus K separate ``b`` vectors
(8 each).  Between segments only ``b`` changes — CPG oscillator state
carries over smoothly so there is no discontinuity in the gait.

Nevergrad params (32 + K*8 total, e.g. 72 for K=5):
    phase (nu,), w (nu,), amplitudes (nu,), ha (nu,)   — 32 (shared)
    b_0 … b_{K-1} (nu,) each                           — K*8

Run
---
uv run examples/thesisMTB/gecko_experiments/exploration_cpg_segmented.py \\
    --budget 500 --workers 50 --dur 120 --fitness f1 --segments 5

Replay
------
uv run examples/thesisMTB/gecko_experiments/exploration_cpg_segmented.py \\
    --replay __data__/exploration_cpg_segmented/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import mujoco
import nevergrad as ng
import numpy as np
import torch
from mujoco import viewer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.traceback import install

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorldWalled
from ariel.simulation.environments import SimpleFlatWorldWalledWithTargets
from ariel.simulation.tasks.exploration import visit_counts_from_xy
from ariel.utils.tracker import Tracker

sys.path.insert(0, str(Path(__file__).parent))
from _fitness_registry import (  # noqa: E402
    ARENA_GRID,
    FITNESS_REGISTRY,
    evaluate_fitness,
    list_fitness_names,
)
from _run_utils import (  # noqa: E402
    make_run_dir,
    save_run_config,
    save_run_summary,
)

install()
console = Console()

DURATION = 30.0
GRID = ARENA_GRID
DEFAULT_SEGMENTS = 5

TARGETS: list[tuple[float, float]] = [
    (3.0, 3.0),
    (3.0, -3.0),
    (-3.0, -3.0),
    (-3.0, 3.0),
]


# ---------------------------------------------------------------------------
# World / model helpers
# ---------------------------------------------------------------------------
def _build_world() -> tuple[SimpleFlatWorldWalledWithTargets, mujoco.MjModel, mujoco.MjData]:
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorldWalledWithTargets(load_precompiled=False, targets_xy=TARGETS)
    world.spawn(gecko().spec, position=[0.0, 0.0, 0.1])
    model = cast(mujoco.MjModel, world.spec.compile())
    data = mujoco.MjData(model)
    return world, model, data


# ---------------------------------------------------------------------------
# Tracker helpers (identical to exploration_cpg.py)
# ---------------------------------------------------------------------------
def _xy_from_tracker(tracker: Tracker) -> list[tuple[float, float]]:
    xpos_history = tracker.history["xpos"]
    if not xpos_history:
        return []
    first_key = next(iter(xpos_history))
    traj = xpos_history[first_key]
    return [(float(p[0]), float(p[1])) for p in traj]


def _xpos_history_first(tracker: Tracker) -> list[list[float]]:
    xpos_history = tracker.history["xpos"]
    if not xpos_history:
        return []
    first_key = next(iter(xpos_history))
    traj = xpos_history[first_key]
    return [[float(p[0]), float(p[1]), float(p[2])] for p in traj]


def show_xpos_history(history: list[list[float]]) -> None:
    if not history:
        console.log("[yellow]No xpos history to plot.[/yellow]")
        return
    pos_data = np.array(history, dtype=float)
    if pos_data.ndim != 2 or pos_data.shape[1] < 2:
        console.log(f"[yellow]Unexpected xpos history shape: {pos_data.shape}[/yellow]")
        return

    half_w = GRID.width_m / 2.0
    half_h = GRID.height_m / 2.0
    ox, oy = GRID.origin_xy

    def _draw_path(ax: "plt.Axes") -> None:
        ax.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
        ax.plot(pos_data[0, 0], pos_data[0, 1], "go", markersize=8, label="Start")
        ax.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", markersize=8, label="End")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend()
        ax.grid(visible=True)
        ax.set_aspect("equal")

    fig1, ax1 = plt.subplots(figsize=(7, 7))
    _draw_path(ax1)
    ax1.set_xlim(ox - half_w, ox + half_w)
    ax1.set_ylim(oy - half_h, oy + half_h)
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (ox - half_w, oy - half_h), GRID.width_m, GRID.height_m,
        linewidth=1.5, edgecolor="gray", facecolor="none", linestyle="--",
        label="World boundary",
    )
    ax1.add_patch(rect)
    ax1.legend()
    ax1.set_title("Robot Path — full world view")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    _draw_path(ax2)
    ax2.set_title("Robot Path — zoomed to trajectory")
    fig2.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Segmented step loop (replaces simple_runner)
# ---------------------------------------------------------------------------
def _segmented_step_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    na_cpg: NaCPG,
    segment_bs: list[np.ndarray],
    segment_duration: float,
    steps_per_loop: int = 100,
) -> None:
    """Run K segments back-to-back, switching ``b`` between each.

    CPG oscillator state carries over — only ``b`` changes at boundaries.
    """
    for b_k in segment_bs:
        b_tensor = torch.from_numpy(np.asarray(b_k)).float()
        na_cpg.set_params_by_group("b", b_tensor)
        end_time = data.time + segment_duration
        while data.time < end_time:
            mujoco.mj_step(model, data, nstep=steps_per_loop)


# ---------------------------------------------------------------------------
# Param helpers — split Nevergrad dict into base gait + segment b's
# ---------------------------------------------------------------------------
def _split_params(
    params: dict[str, np.ndarray],
    n_segments: int,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Return (base_gait_dict, [b_0, b_1, ..., b_{K-1}])."""
    base = {}
    segment_bs = []
    for key, val in params.items():
        if key.startswith("b_"):
            segment_bs.append(np.array(val))
        else:
            base[key] = np.array(val)
    # Sort by segment index to guarantee order
    segment_bs_dict = {}
    for key, val in params.items():
        if key.startswith("b_"):
            idx = int(key.split("_", 1)[1])
            segment_bs_dict[idx] = np.array(val)
    segment_bs = [segment_bs_dict[i] for i in range(n_segments)]
    return base, segment_bs


# ---------------------------------------------------------------------------
# Top-level picklable worker — one subprocess per candidate evaluation.
# Must be module-level (not a closure) for ProcessPoolExecutor on Windows.
# ---------------------------------------------------------------------------
def _segmented_eval_worker(job: dict) -> tuple[float, list[tuple[float, float]]]:
    """Build a fresh MuJoCo context per subprocess and evaluate one candidate.

    Returns (fitness, xy_trajectory) so the main process can track best params.
    """
    params: dict[str, np.ndarray] = {k: np.asarray(v) for k, v in job["params"].items()}
    fitness_name: str = job["fitness_name"]
    duration: float = job["duration"]
    n_segments: int = job["n_segments"]
    forward_xy: tuple[float, float] = tuple(job["forward_xy"])  # type: ignore[assignment]

    world, model, data = _build_world()
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

    base, segment_bs = _split_params(params, n_segments)
    na_cpg.set_param_with_dict(base)
    mujoco.set_mjcb_control(ctrl.set_control)

    segment_duration = duration / n_segments
    _segmented_step_loop(model, data, na_cpg, segment_bs, segment_duration)

    xy = _xy_from_tracker(tracker)
    N = visit_counts_from_xy(xy, GRID)
    fitness = evaluate_fitness(
        fitness_name, N=N, xy=xy, grid=GRID,
        dt=sample_dt, forward_xy=forward_xy, targets=TARGETS,
    )
    return fitness, xy


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def _resolve_replay_dir(replay_arg: str) -> Path:
    p = Path(replay_arg).expanduser().resolve()
    if p.is_file():
        return p.parent
    if not p.exists():
        raise FileNotFoundError(f"Replay path does not exist: {p}")
    return p


def _make_segmented_callback(
    na_cpg: NaCPG,
    segment_bs: list[np.ndarray],
    segment_duration: float,
):
    """Return a callback that switches ``b`` based on simulation time."""
    n_segments = len(segment_bs)
    b_tensors = [torch.from_numpy(np.asarray(b)).float() for b in segment_bs]
    current_segment = [-1]

    def callback(_model: mujoco.MjModel, data: mujoco.MjData, *_a, **_kw):
        seg_idx = min(int(data.time / segment_duration), n_segments - 1)
        if seg_idx != current_segment[0]:
            na_cpg.set_params_by_group("b", b_tensors[seg_idx])
            current_segment[0] = seg_idx
        return na_cpg.forward(float(data.time))

    return callback


def run_replay(
    replay_dir: Path,
    *,
    duration: float,
    fitness_name: str,
    forward_xy: tuple[float, float],
) -> None:
    params_path = replay_dir / "best_params.npz"
    config_path = replay_dir / "config.json"
    if not params_path.is_file():
        raise FileNotFoundError(
            f"best_params.npz not found in {replay_dir}. "
            "Pass either the run folder or the .npz path itself via --replay."
        )

    # Read n_segments from saved config
    n_segments = DEFAULT_SEGMENTS
    if config_path.is_file():
        cfg = json.loads(config_path.read_text())
        extras = cfg.get("extras", {})
        n_segments = int(extras.get("n_segments", DEFAULT_SEGMENTS))

    console.log(f"Replaying best controller from: {params_path}")
    console.log(f"  segments = {n_segments}")

    world, model, data = _build_world()
    nu = int(model.nu)

    adj_dict = create_fully_connected_adjacency(nu)
    na_cpg = NaCPG(adj_dict, angle_tracking=False)

    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )

    loaded = np.load(params_path)
    params: dict[str, np.ndarray] = {k: loaded[k] for k in loaded.files}
    base, segment_bs = _split_params(params, n_segments)

    na_cpg.set_param_with_dict(base)

    segment_duration = duration / n_segments
    ctrl = Controller(
        controller_callback_function=_make_segmented_callback(
            na_cpg, segment_bs, segment_duration,
        ),
        time_steps_per_ctrl_step=10,
        time_steps_per_save=10,
        alpha=1.0,
        tracker=tracker,
    )
    ctrl.tracker.setup(world.spec, data)
    sample_dt = float(model.opt.timestep) * float(ctrl.time_steps_per_save)

    mujoco.mj_resetData(model, data)
    tracker.reset()
    mujoco.set_mjcb_control(ctrl.set_control)

    console.rule("[bold cyan]Viewer replay[/bold cyan]")
    viewer.launch(model=model, data=data)

    show_xpos_history(_xpos_history_first(tracker))

    xy = _xy_from_tracker(tracker)
    N = visit_counts_from_xy(xy, GRID)
    fit = evaluate_fitness(
        fitness_name, N=N, xy=xy, grid=GRID, dt=sample_dt,
        forward_xy=forward_xy, targets=TARGETS,
    )
    console.log(f"Replay {fitness_name} = {fit:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GECKO exploration — segmented NaCPG (per-segment b steering)"
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--dur", type=float, default=DURATION)
    parser.add_argument(
        "--fitness", type=str, default="f1", choices=list_fitness_names(),
        help=f"Fitness function. Available: {list_fitness_names()}",
    )
    parser.add_argument("--forward-x", type=float, default=1.0)
    parser.add_argument("--forward-y", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--segments", type=int, default=DEFAULT_SEGMENTS,
        help=f"Number of time segments K (default: {DEFAULT_SEGMENTS}).",
    )
    parser.add_argument("--replay", type=str, default=None)
    args = parser.parse_args()

    forward_xy: tuple[float, float] = (float(args.forward_x), float(args.forward_y))

    # ---- Replay-only mode ----
    if args.replay is not None:
        replay_dir = _resolve_replay_dir(args.replay)
        run_replay(
            replay_dir, duration=float(args.dur),
            fitness_name=str(args.fitness), forward_xy=forward_xy,
        )
        return

    budget = int(args.budget)
    num_workers = int(args.workers)
    duration = float(args.dur)
    seed = int(args.seed)
    n_segments = int(args.segments)
    np.random.seed(seed)

    fitness_name = str(args.fitness)
    if fitness_name not in FITNESS_REGISTRY:
        raise SystemExit(f"Unknown fitness '{fitness_name}'.")

    # ---- Run folder ----
    run_dir = make_run_dir(Path(__file__), args.run_name)
    save_run_config(
        run_dir, args,
        extras={
            "resolved": {"budget": budget, "num_workers": num_workers},
            "fitness": fitness_name,
            "fitness_description": FITNESS_REGISTRY[fitness_name].description,
            "forward_xy": list(forward_xy),
            "controller": "segmented_nacpg",
            "n_segments": n_segments,
        },
    )
    console.log(f"Run folder: {run_dir}")

    # ---- World + model ----
    console.log("Building world and GECKO model...")
    world, model, data = _build_world()
    nu = int(model.nu)
    console.log(f"GECKO actuators (nu): {nu}")

    # ---- Controller ----
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

    # ---- Nevergrad search space (32 + K*8 params) ----
    param_kwargs: dict[str, ng.p.Array | ng.p.Scalar] = {
        "phase": ng.p.Array(shape=(nu,)).set_bounds(-2 * np.pi, 2 * np.pi),
        "w": ng.p.Array(shape=(nu,)).set_bounds(-2 * np.pi, 2 * np.pi),
        "amplitudes": ng.p.Array(shape=(nu,)).set_bounds(-2 * np.pi, 2 * np.pi),
        "ha": ng.p.Array(shape=(nu,)).set_bounds(-10.0, 10.0),
    }
    for k in range(n_segments):
        param_kwargs[f"b_{k}"] = ng.p.Array(shape=(nu,)).set_bounds(-100.0, 100.0)

    params_spec = ng.p.Instrumentation(**param_kwargs)
    optimizer = ng.optimizers.PSO(
        parametrization=params_spec,
        budget=budget,
        num_workers=num_workers,
    )
    segment_duration = duration / n_segments
    console.log(
        f"Nevergrad params: {optimizer.parametrization.dimension}  "
        f"(32 base + {n_segments}x8 segments, "
        f"segment_duration={segment_duration:.1f}s)"
    )

    # ---- Optimisation loop ----
    console.rule(
        f"[bold blue]Segmented NaCPG — fitness={fitness_name}, "
        f"budget={budget}, workers={num_workers}, dur={duration}s, "
        f"K={n_segments}, nu={nu}[/bold blue]"
    )

    best_fitness = -float("inf")
    best_params: dict[str, np.ndarray] | None = None
    best_xy: list[tuple[float, float]] = []
    all_fitness: list[float] = []
    best_so_far: list[float] = []

    start_time = time.perf_counter()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Optimising...", total=budget, info="")

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            evals_done = 0
            while evals_done < budget:
                batch_size = min(num_workers, budget - evals_done)
                candidates = [optimizer.ask() for _ in range(batch_size)]
                jobs = [
                    {
                        "fitness_name": fitness_name,
                        "duration": duration,
                        "n_segments": n_segments,
                        "forward_xy": list(forward_xy),
                        "params": {
                            k: np.asarray(v).tolist() for k, v in c.kwargs.items()
                        },
                    }
                    for c in candidates
                ]
                futures = [pool.submit(_segmented_eval_worker, j) for j in jobs]
                batch_results = [f.result() for f in futures]

                for c, (f_value, xy) in zip(candidates, batch_results):
                    optimizer.tell(c, -f_value)
                    all_fitness.append(f_value)
                    if f_value > best_fitness:
                        best_fitness = f_value
                        best_params = {k: np.array(v) for k, v in c.kwargs.items()}
                        best_xy = xy
                    best_so_far.append(best_fitness)

                evals_done += batch_size
                last_f = batch_results[-1][0]
                progress.update(
                    task, advance=batch_size,
                    info=f"| {fitness_name}={last_f:.4f} | best={best_fitness:.4f}",
                )

    runtime_s = time.perf_counter() - start_time

    # ---- Results ----
    console.rule("[bold green]Optimisation complete[/bold green]")
    console.log(
        f"Best {fitness_name} = [bold]{best_fitness:.4f}[/bold] "
        f"(runtime {runtime_s:.1f}s)"
    )

    params_path = run_dir / "best_params.npz"
    if best_params is not None:
        np.savez(params_path, **best_params)
        console.log(f"Best params saved to: {params_path}")

    traj_path = run_dir / "best_trajectory.npy"
    if best_xy:
        np.save(traj_path, np.array(best_xy, dtype=float))
        console.log(f"Best trajectory saved to: {traj_path}")

    save_run_summary(
        run_dir,
        best_fitness=f"{best_fitness:.6f}",
        fitness=fitness_name,
        controller="segmented_nacpg",
        n_segments=n_segments,
        segment_duration_s=f"{segment_duration:.1f}",
        budget=budget,
        num_workers=num_workers,
        duration_s=duration,
        runtime_s=f"{runtime_s:.2f}",
        seed=seed,
        nu=nu,
        n_params=optimizer.parametrization.dimension,
        best_params_path=str(params_path) if best_params is not None else "",
        forward_xy=f"({forward_xy[0]}, {forward_xy[1]})",
    )

    # ---- Fitness plot ----
    window = max(num_workers, 10)
    avg_fitness = np.convolve(all_fitness, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(all_fitness)), all_fitness, ".", color="lightblue",
            markersize=2, alpha=0.4, label="Per-evaluation")
    ax.plot(range(window - 1, window - 1 + len(avg_fitness)), avg_fitness,
            "-", color="blue", linewidth=1.2, label=f"Rolling avg (w={window})")
    ax.plot(range(len(best_so_far)), best_so_far, "-", color="red",
            linewidth=1.5, label="Best so far")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel(f"Fitness ({fitness_name})")
    ax.set_title(
        f"Segmented NaCPG (K={n_segments}) — {fitness_name}, budget={budget}, "
        f"workers={num_workers}, dur={duration}s"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = run_dir / "fitness_over_evaluations.png"
    fig.savefig(plot_path, dpi=150)
    console.log(f"Fitness plot saved to: {plot_path}")
    plt.show()

    # ---- Viewer replay ----
    if best_params is not None and not args.no_viewer:
        console.rule("[bold cyan]Viewer replay (best controller)[/bold cyan]")

        mujoco.mj_resetData(model, data)
        tracker.reset()

        bp = {k: np.array(v) for k, v in best_params.items()}
        base, segment_bs = _split_params(bp, n_segments)
        na_cpg.set_param_with_dict(base)

        ctrl.controller_callback_function = _make_segmented_callback(
            na_cpg, segment_bs, segment_duration,
        )
        mujoco.set_mjcb_control(ctrl.set_control)

        viewer.launch(model=model, data=data)

        show_xpos_history(_xpos_history_first(tracker))

        replay_xy = _xy_from_tracker(tracker)
        replay_N = visit_counts_from_xy(replay_xy, GRID)
        replay_fit = evaluate_fitness(
            fitness_name, N=replay_N, xy=replay_xy, grid=GRID,
            dt=sample_dt, forward_xy=forward_xy, targets=TARGETS,
        )
        console.log(f"Replay {fitness_name} = {replay_fit:.4f}")


if __name__ == "__main__":
    main()
