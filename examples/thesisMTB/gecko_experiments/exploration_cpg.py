"""GECKO exploration using Nevergrad (PSO) to optimise a NaCPG controller.

Run
---
uv run examples/thesisMTB/gecko_experiments/exploration_cpg.py \\
    --budget 500 --workers 50 --dur 120 --fitness f1

# Replay best controller from a previous run folder:
uv run examples/thesisMTB/gecko_experiments/exploration_cpg.py \\
    --replay __data__/exploration_cpg/runs/2026-04-29_18-00-00__default
"""

from __future__ import annotations


import argparse
import sys
import time
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import mujoco
import nevergrad as ng
import numpy as np
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
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.exploration import (
    visit_counts_from_xy,
)
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker

# Local helpers (same folder).
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

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DURATION = 30.0  # evaluation duration in seconds

GRID = ARENA_GRID  # defined in _fitness_registry.py — change cell size there

TARGETS: list[tuple[float, float]] = [
    (3.0, 3.0),
    (3.0, -3.0),
    (-3.0, -3.0),
    (-3.0, 3.0),
]


# ---------------------------------------------------------------------------
# World / model helpers
# ---------------------------------------------------------------------------
def _build_world() -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld(load_precompiled=False, targets_xy=TARGETS)
    world.spawn(gecko().spec, position=[0.0, 0.0, 0.1])
    model = cast(mujoco.MjModel, world.spec.compile())
    data = mujoco.MjData(model)
    return world, model, data


# ---------------------------------------------------------------------------
# Tracker helper
# ---------------------------------------------------------------------------
def _xy_from_tracker(tracker: Tracker) -> list[tuple[float, float]]:
    xpos_history = tracker.history["xpos"]
    if not xpos_history:
        return []
    first_key = next(iter(xpos_history))
    traj = xpos_history[first_key]
    return [(float(p[0]), float(p[1])) for p in traj]


def _xpos_history_first(tracker: Tracker) -> list[list[float]]:
    """Return the first tracked `xpos` history as list of [x,y,z]."""
    xpos_history = tracker.history["xpos"]
    if not xpos_history:
        return []
    first_key = next(iter(xpos_history))
    traj = xpos_history[first_key]
    return [[float(p[0]), float(p[1]), float(p[2])] for p in traj]


def show_xpos_history(history: list[list[float]]) -> None:
    """Plot XY trajectory as two side-by-side figures.

    - Left / Figure 1: full 10 m × 10 m world (GRID bounds), so you can see
      how much of the arena the robot actually used.
    - Right / Figure 2: zoomed to the path's bounding box (original view),
      so fine detail of the motion is still visible.
    """
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

    # ---- Figure 1: full world view ----
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    _draw_path(ax1)
    ax1.set_xlim(ox - half_w, ox + half_w)
    ax1.set_ylim(oy - half_h, oy + half_h)
    # Draw the grid boundary as a faint rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (ox - half_w, oy - half_h),
        GRID.width_m,
        GRID.height_m,
        linewidth=1.5,
        edgecolor="gray",
        facecolor="none",
        linestyle="--",
        label="World boundary",
    )
    ax1.add_patch(rect)
    ax1.legend()
    ax1.set_title("Robot Path — full world view (10 m × 10 m)")
    fig1.tight_layout()

    # ---- Figure 2: zoomed to path bounding box ----
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    _draw_path(ax2)
    ax2.set_title("Robot Path — zoomed to trajectory")
    fig2.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Single-rollout evaluation
# ---------------------------------------------------------------------------
def _evaluate(
    params: dict[str, np.ndarray],
    *,
    fitness_name: str,
    forward_xy: tuple[float, float],
    sample_dt: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    na_cpg: NaCPG,
    ctrl: Controller,
    tracker: Tracker,
    duration: float,
    grid: GridSpec,
) -> float:
    """Run one rollout, score with the chosen fitness (higher is better)."""
    mujoco.mj_resetData(model, data)
    tracker.reset()

    na_cpg.set_param_with_dict(params)
    mujoco.set_mjcb_control(ctrl.set_control)
    simple_runner(model, data, duration=duration)

    xy = _xy_from_tracker(tracker)
    N = visit_counts_from_xy(xy, grid)
    return evaluate_fitness(
        fitness_name,
        N=N,
        xy=xy,
        grid=grid,
        dt=sample_dt,
        forward_xy=forward_xy,
        targets=TARGETS,
    )


# ---------------------------------------------------------------------------
# Replay mode (uses a previously saved best_params.npz)
# ---------------------------------------------------------------------------
def _resolve_replay_dir(replay_arg: str) -> Path:
    """Accept either a run dir or the path to best_params.npz directly."""
    p = Path(replay_arg).expanduser().resolve()
    if p.is_file():
        return p.parent
    if not p.exists():
        raise FileNotFoundError(f"Replay path does not exist: {p}")
    return p


def run_replay(
    replay_dir: Path,
    *,
    duration: float,
    fitness_name: str,
    forward_xy: tuple[float, float],
) -> None:
    """Load best_params.npz from a run folder and launch the MuJoCo viewer."""
    params_path = replay_dir / "best_params.npz"
    if not params_path.is_file():
        raise FileNotFoundError(
            f"best_params.npz not found in {replay_dir}. "
            "Pass either the run folder or the .npz path itself via --replay."
        )

    console.log(f"Replaying best controller from: {params_path}")

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

    loaded = np.load(params_path)
    params: dict[str, np.ndarray] = {k: loaded[k] for k in loaded.files}

    mujoco.mj_resetData(model, data)
    tracker.reset()
    na_cpg.set_param_with_dict(params)
    mujoco.set_mjcb_control(ctrl.set_control)

    console.rule("[bold cyan]Viewer replay[/bold cyan]")
    viewer.launch(model=model, data=data)

    show_xpos_history(_xpos_history_first(tracker))

    xy = _xy_from_tracker(tracker)
    N = visit_counts_from_xy(xy, GRID)
    fit = evaluate_fitness(
        fitness_name,
        N=N,
        xy=xy,
        grid=GRID,
        dt=sample_dt,
        forward_xy=forward_xy,
        targets=TARGETS,
    )
    console.log(
        f"Replay {fitness_name} = {fit:.4f} "
        f"(rollout duration = {duration}s, but viewer is interactive)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GECKO exploration via Nevergrad PSO + NaCPG"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=500,
        help="Total number of NaCPG rollout evaluations (default: 500).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Nevergrad num_workers / PSO swarm size (default: 50).",
    )
    parser.add_argument(
        "--dur",
        type=float,
        default=DURATION,
        help="Simulation duration per evaluation in seconds (default: 30).",
    )
    parser.add_argument(
        "--fitness",
        type=str,
        default="f1",
        choices=list_fitness_names(),
        help=(
            "Fitness function name (see _fitness_registry.py). "
            f"Available: {list_fitness_names()}"
        ),
    )
    parser.add_argument(
        "--forward-x",
        type=float,
        default=1.0,
        help="X component of forward direction (only used by speed/forward fitnesses).",
    )
    parser.add_argument(
        "--forward-y",
        type=float,
        default=0.0,
        help="Y component of forward direction (only used by speed/forward fitnesses).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional human-readable label appended to the run folder name.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the MuJoCo viewer replay after optimisation.",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help=(
            "Run folder (or path to best_params.npz) to replay in the MuJoCo "
            "viewer. When set, optimisation is skipped."
        ),
    )
    args = parser.parse_args()

    forward_xy: tuple[float, float] = (float(args.forward_x), float(args.forward_y))

    # ------------------------------------------------------------------ #
    # Replay-only mode
    # ------------------------------------------------------------------ #
    if args.replay is not None:
        replay_dir = _resolve_replay_dir(args.replay)
        run_replay(
            replay_dir,
            duration=float(args.dur),
            fitness_name=str(args.fitness),
            forward_xy=forward_xy,
        )
        return

    budget = int(args.budget)
    num_workers = int(args.workers)
    duration = float(args.dur)
    seed = int(args.seed)
    np.random.seed(seed)

    fitness_name = str(args.fitness)
    if fitness_name not in FITNESS_REGISTRY:
        raise SystemExit(f"Unknown fitness '{fitness_name}'.")

    # ------------------------------------------------------------------ #
    # Run folder
    # ------------------------------------------------------------------ #
    run_dir = make_run_dir(Path(__file__), args.run_name)
    save_run_config(
        run_dir,
        args,
        extras={
            "resolved": {"budget": budget, "num_workers": num_workers},
            "fitness": fitness_name,
            "fitness_description": FITNESS_REGISTRY[fitness_name].description,
            "forward_xy": list(forward_xy),
        },
    )
    console.log(f"Run folder: {run_dir}")

    # ------------------------------------------------------------------ #
    # World + model (built once; shared across all evaluations)
    # ------------------------------------------------------------------ #
    console.log("Building world and GECKO model...")
    world, model, data = _build_world()
    nu = int(model.nu)
    console.log(f"GECKO actuators (nu): {nu}")

    # ------------------------------------------------------------------ #
    # Controller (NaCPG, same settings as A2_template_cpg.py)
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Nevergrad search space (5 parameter vectors, same bounds as A2)
    # ------------------------------------------------------------------ #
    params_spec = ng.p.Instrumentation(
        phase=ng.p.Array(shape=(nu,)).set_bounds(-2 * np.pi, 2 * np.pi),
        w=ng.p.Array(shape=(nu,)).set_bounds(-2 * np.pi, 2 * np.pi),
        amplitudes=ng.p.Array(shape=(nu,)).set_bounds(-2 * np.pi, 2 * np.pi),
        ha=ng.p.Array(shape=(nu,)).set_bounds(-10.0, 10.0),
        b=ng.p.Array(shape=(nu,)).set_bounds(-100.0, 100.0),
    )
    optimizer = ng.optimizers.PSO(
        parametrization=params_spec,
        budget=budget,
        num_workers=num_workers,
    )

    # ------------------------------------------------------------------ #
    # Optimisation loop (sequential ask / tell)
    # ------------------------------------------------------------------ #
    console.rule(
        f"[bold blue]Nevergrad PSO -- fitness={fitness_name}, "
        f"budget={budget}, workers={num_workers}, dur={duration}s, nu={nu}[/bold blue]"
    )

    best_fitness = -float("inf")
    best_params: dict[str, np.ndarray] | None = None
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
        task = progress.add_task(
            "[cyan]Optimising...", total=budget, info=""
        )

        for _ in range(budget):
            x = optimizer.ask()
            candidate_params: dict[str, np.ndarray] = x.kwargs  # type: ignore[assignment]

            f_value = _evaluate(
                candidate_params,
                fitness_name=fitness_name,
                forward_xy=forward_xy,
                sample_dt=sample_dt,
                model=model,
                data=data,
                na_cpg=na_cpg,
                ctrl=ctrl,
                tracker=tracker,
                duration=duration,
                grid=GRID,
            )
            optimizer.tell(x, -f_value)  # Nevergrad minimises -> negate

            all_fitness.append(f_value)
            if f_value > best_fitness:
                best_fitness = f_value
                best_params = {k: np.array(v) for k, v in candidate_params.items()}
                best_xy = _xy_from_tracker(tracker)
            best_so_far.append(best_fitness)

            progress.update(
                task,
                advance=1,
                info=f"| {fitness_name}={f_value:.4f} | best={best_fitness:.4f}",
            )

    runtime_s = time.perf_counter() - start_time

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
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
        budget=budget,
        num_workers=num_workers,
        duration_s=duration,
        runtime_s=f"{runtime_s:.2f}",
        seed=seed,
        nu=nu,
        best_params_path=str(params_path) if best_params is not None else "",
        forward_xy=f"({forward_xy[0]}, {forward_xy[1]})",
    )

    # ------------------------------------------------------------------ #
    # Fitness over evaluations plot
    # ------------------------------------------------------------------ #
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
    ax.set_title(f"{fitness_name} — budget={budget}, workers={num_workers}, dur={duration}s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = run_dir / "fitness_over_evaluations.png"
    fig.savefig(plot_path, dpi=150)
    console.log(f"Fitness plot saved to: {plot_path}")
    plt.show()

    # ------------------------------------------------------------------ #
    # Viewer replay
    # ------------------------------------------------------------------ #
    if best_params is not None and not args.no_viewer:
        console.rule("[bold cyan]Viewer replay (best controller)[/bold cyan]")

        mujoco.mj_resetData(model, data)
        tracker.reset()
        na_cpg.set_param_with_dict(best_params)
        mujoco.set_mjcb_control(ctrl.set_control)

        viewer.launch(model=model, data=data)

        show_xpos_history(_xpos_history_first(tracker))

        replay_xy = _xy_from_tracker(tracker)
        replay_N = visit_counts_from_xy(replay_xy, GRID)
        replay_fit = evaluate_fitness(
            fitness_name,
            N=replay_N,
            xy=replay_xy,
            grid=GRID,
            dt=sample_dt,
            forward_xy=forward_xy,
            targets=TARGETS,
        )
        console.log(f"Replay {fitness_name} = {replay_fit:.4f}")


if __name__ == "__main__":
    main()
