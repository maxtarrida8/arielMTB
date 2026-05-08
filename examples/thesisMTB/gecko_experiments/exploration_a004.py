"""GECKO exploration demo using ARIEL's step-based EA (`ariel.ec.a004`).

This experiment evolves a continuous controller parameter vector for a GECKO
robot in `SimpleFlatWorld`. The fitness function is selectable at runtime via
the `--fitness` CLI flag (see `_fitness_registry.py`). Default is
``f1_plus_speed`` (cell coverage + mean forward speed).

Each invocation creates its own folder under
``__data__/exploration_a004/runs/<timestamp>[__<run-name>]/`` containing
``config.json``, ``summary.txt`` and ``database.db``.

Run
---
uv run examples/thesisMTB/gecko_experiments/exploration_a004.py \\
    --generations 10 --pop 20 --dur 30 --fitness f1_plus_speed --run-name baseline
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
from mujoco import viewer
from rich.console import Console
from rich.traceback import install

import ariel.ec.a004 as ea_mod
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EAStep
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.exploration import (
    GridSpec,
    visit_counts_from_xy,
)
from ariel.utils.tracker import Tracker
from ariel.utils.runners import simple_runner

# Local helpers (same folder).
sys.path.insert(0, str(Path(__file__).parent))
from _fitness_registry import (  # noqa: E402
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


def _build_model_and_data() -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
    mujoco.set_mjcb_control(None)  # ensure clean callback

    world = SimpleFlatWorld(load_precompiled=False)
    robot = gecko()
    world.spawn(robot.spec, position=[0.0, 0.0, 0.1])
    model = cast(mujoco.MjModel, world.spec.compile())
    data = mujoco.MjData(model)
    return world, model, data


def _clip_params(x: np.ndarray, nu: int) -> np.ndarray:
    """Clip per-parameter-group to keep NaCPG stable."""
    x = np.asarray(x, dtype=float).flatten()
    if x.size != nu * 5:
        x = np.resize(x, nu * 5)

    phase = np.clip(x[0 * nu : 1 * nu], -2 * np.pi, 2 * np.pi)
    w = np.clip(x[1 * nu : 2 * nu], -2 * np.pi, 2 * np.pi)
    amplitudes = np.clip(x[2 * nu : 3 * nu], -2 * np.pi, 2 * np.pi)
    ha = np.clip(x[3 * nu : 4 * nu], -10.0, 10.0)
    b = np.clip(x[4 * nu : 5 * nu], -100.0, 100.0)
    return np.concatenate([phase, w, amplitudes, ha, b]).astype(float)


def _make_initial_genotype(nu: int, rng: np.random.Generator) -> list[float]:
    # Start near zeros with small noise.
    base = np.zeros(nu * 5, dtype=float)
    noise = rng.normal(loc=0.0, scale=0.25, size=base.shape)
    return _clip_params(base + noise, nu=nu).tolist()


def _xy_from_tracker(tracker: Tracker) -> list[tuple[float, float]]:
    xpos_history = tracker.history["xpos"]
    if not xpos_history:
        return []
    first_key = next(iter(xpos_history))
    traj = xpos_history[first_key]
    return [(float(p[0]), float(p[1])) for p in traj]


def evaluate_population(
    population: list[Individual],
    *,
    world: SimpleFlatWorld,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grid: GridSpec,
    duration_s: float,
    fitness_name: str,
    forward_xy: tuple[float, float] | None,
) -> list[Individual]:
    """Evaluate individuals that require evaluation."""
    # Tracker on the GECKO core geom positions.
    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )
    tracker.setup(world.spec, data)

    # NaCPG controller instance (parameters will be set per individual).
    nu = int(model.nu)
    adj = create_fully_connected_adjacency(nu)
    nacpg = NaCPG(adj, angle_tracking=False)
    ctrl = Controller(
        controller_callback_function=lambda _m, d: nacpg.forward(float(d.time)),
        # Save trajectory frequently for cell coverage.
        time_steps_per_ctrl_step=10,
        time_steps_per_save=10,
        alpha=1.0,
        tracker=tracker,
    )
    ctrl.tracker.setup(world.spec, data)
    # Effective time between stored trajectory samples.
    sample_dt = float(model.opt.timestep) * float(ctrl.time_steps_per_save)

    for ind in population:
        if (not ind.alive) or (not ind.requires_eval):
            continue

        # Build parameter vector
        geno = np.asarray(ind.genotype, dtype=float).flatten()
        geno = _clip_params(geno, nu=nu)

        # Reset sim + tracking
        mujoco.mj_resetData(model, data)
        tracker.reset()
        mujoco.set_mjcb_control(ctrl.set_control)

        # Set controller params and run
        phase = geno[0 * nu : 1 * nu]
        w = geno[1 * nu : 2 * nu]
        amplitudes = geno[2 * nu : 3 * nu]
        ha = geno[3 * nu : 4 * nu]
        b = geno[4 * nu : 5 * nu]
        nacpg.set_param_with_dict(
            {
                "phase": phase,
                "w": w,
                "amplitudes": amplitudes,
                "ha": ha,
                "b": b,
            }
        )
        simple_runner(model, data, duration=float(duration_s))

        # Extract XY trajectory from tracker
        xy = _xy_from_tracker(tracker)

        N = visit_counts_from_xy(xy, grid)
        fit = evaluate_fitness(
            fitness_name,
            N=N,
            xy=xy,
            grid=grid,
            dt=sample_dt,
            forward_xy=forward_xy,
        )

        ind.fitness = fit  # sets requires_eval=False

    return population


def replay_best(
    ind: Individual,
    *,
    world: SimpleFlatWorld,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grid: GridSpec,
    duration_s: float,
    fitness_name: str,
    forward_xy: tuple[float, float] | None,
) -> float:
    tracker = Tracker(
        mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )
    tracker.setup(world.spec, data)

    nu = int(model.nu)
    adj = create_fully_connected_adjacency(nu)
    nacpg = NaCPG(adj, angle_tracking=False)
    ctrl = Controller(
        controller_callback_function=lambda _m, d: nacpg.forward(float(d.time)),
        time_steps_per_ctrl_step=10,
        time_steps_per_save=10,
        alpha=1.0,
        tracker=tracker,
    )
    ctrl.tracker.setup(world.spec, data)
    sample_dt = float(model.opt.timestep) * float(ctrl.time_steps_per_save)

    geno = _clip_params(np.asarray(ind.genotype, dtype=float).flatten(), nu=nu)
    phase = geno[0 * nu : 1 * nu]
    w = geno[1 * nu : 2 * nu]
    amplitudes = geno[2 * nu : 3 * nu]
    ha = geno[3 * nu : 4 * nu]
    b = geno[4 * nu : 5 * nu]
    nacpg.set_param_with_dict(
        {
            "phase": phase,
            "w": w,
            "amplitudes": amplitudes,
            "ha": ha,
            "b": b,
        }
    )

    mujoco.mj_resetData(model, data)
    tracker.reset()
    mujoco.set_mjcb_control(ctrl.set_control)
    viewer.launch(model=model, data=data)

    xy = _xy_from_tracker(tracker)
    N = visit_counts_from_xy(xy, grid)
    return evaluate_fitness(
        fitness_name,
        N=N,
        xy=xy,
        grid=grid,
        dt=sample_dt,
        forward_xy=forward_xy,
    )


def parent_selection(population: list[Individual], *, fraction: float = 0.4) -> list[Individual]:
    """Select a top fraction as parents by tagging them with 'ps'."""
    # Higher fitness is better (maximisation).
    alive = [ind for ind in population if ind.alive and (ind.fitness_ is not None)]
    alive.sort(key=lambda i: i.fitness, reverse=True)

    k = max(2, int(np.ceil(len(alive) * float(fraction))))
    parent_ids = {id(ind) for ind in alive[:k]}
    for ind in population:
        ind.tags = {"ps": id(ind) in parent_ids}
    return population


def reproduction(
    population: list[Individual],
    *,
    target_pool_multiplier: int,
    rng: np.random.Generator,
    mutation_prob: float = 0.25,
    mutation_sigma: float = 0.25,
) -> list[Individual]:
    """Create offspring using uniform crossover + Gaussian mutation."""
    parents = [ind for ind in population if ind.alive and ind.tags.get("ps", False)]
    if len(parents) < 2:
        parents = [ind for ind in population if ind.alive]

    target_pool = ea_mod.config.target_population_size * int(target_pool_multiplier)
    base_len = len(np.asarray(parents[0].genotype, dtype=float).flatten()) if parents else 0

    offspring: list[Individual] = []
    while len(population) + len(offspring) < target_pool:
        p1, p2 = random.sample(parents, 2)
        g1 = np.asarray(p1.genotype, dtype=float).flatten()
        g2 = np.asarray(p2.genotype, dtype=float).flatten()
        if g1.size != base_len:
            g1 = np.resize(g1, base_len)
        if g2.size != base_len:
            g2 = np.resize(g2, base_len)

        mask = rng.random(base_len) < 0.5
        child = np.where(mask, g1, g2).astype(float)

        mut_mask = rng.random(base_len) < float(mutation_prob)
        child[mut_mask] += rng.normal(0.0, float(mutation_sigma), size=int(np.sum(mut_mask)))
        # We don't have `nu` directly here; keep broad safety clip.
        child = np.clip(child, -100.0, 100.0)

        ind = Individual()
        ind.genotype = child.tolist()
        ind.requires_eval = True
        ind.tags = {"ps": False}
        offspring.append(ind)

    population.extend(offspring)
    return population


def survivor_selection(population: list[Individual]) -> list[Individual]:
    """Keep best N individuals alive and kill the rest."""
    alive = [ind for ind in population if ind.alive and (ind.fitness_ is not None)]
    alive.sort(key=lambda i: i.fitness, reverse=True)
    survivors = alive[: ea_mod.config.target_population_size]
    survivor_obj_ids = {id(ind) for ind in survivors}
    for ind in population:
        if id(ind) not in survivor_obj_ids:
            ind.alive = False
    return population


def main() -> None:
    parser = argparse.ArgumentParser(description="GECKO exploration with ARIEL EA")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--dur", type=float, default=30.0, help="Evaluation duration (seconds)")
    parser.add_argument(
        "--fitness",
        type=str,
        default="f1_plus_speed",
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
        "--run-name",
        type=str,
        default=None,
        help="Optional human-readable label appended to the run folder name.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip replaying the best individual in the MuJoCo viewer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    fitness_name = str(args.fitness)
    if fitness_name not in FITNESS_REGISTRY:
        raise SystemExit(f"Unknown fitness '{fitness_name}'.")

    # Per-run output folder
    run_dir = make_run_dir(Path(__file__), args.run_name)
    save_run_config(
        run_dir,
        args,
        extras={
            "fitness": fitness_name,
            "fitness_description": FITNESS_REGISTRY[fitness_name].description,
            "forward_xy": [float(args.forward_x), float(args.forward_y)],
        },
    )
    console.log(f"Run folder: {run_dir}")

    # Configure ARIEL EA module (global config is what EA reads)
    ea_mod.config = ea_mod.EASettings(
        is_maximisation=True,
        num_of_generations=int(args.generations),
        target_population_size=int(args.pop),
        db_file_path=run_dir / "database.db",
        db_handling="delete",
    )

    # Build environment once (shared evaluation model)
    world, model, data = _build_model_and_data()

    nu = int(model.nu)

    # Exploration grid overlay (10m x 10m, 100x100 cells)
    grid = GridSpec(width_m=10.0, height_m=10.0, nrow=100, ncol=100, origin_xy=(0.0, 0.0))

    def create_individual() -> Individual:
        ind = Individual()
        ind.genotype = _make_initial_genotype(nu, rng)
        ind.tags = {"ps": False}
        return ind

    # Initial population
    population = [create_individual() for _ in range(int(args.pop))]
    forward_xy = (float(args.forward_x), float(args.forward_y))

    start_time = time.perf_counter()
    population = evaluate_population(
        population,
        world=world,
        model=model,
        data=data,
        grid=grid,
        duration_s=float(args.dur),
        fitness_name=fitness_name,
        forward_xy=forward_xy,
    )

    # EA steps
    ops = [
        EAStep("parent_selection", lambda pop: parent_selection(pop, fraction=0.4)),
        EAStep(
            "reproduction",
            lambda pop: reproduction(
                pop,
                target_pool_multiplier=2,
                rng=rng,
                mutation_prob=0.25,
                mutation_sigma=0.25,
            ),
        ),
        EAStep(
            "evaluation",
            lambda pop: evaluate_population(
                pop,
                world=world,
                model=model,
                data=data,
                grid=grid,
                duration_s=float(args.dur),
                fitness_name=fitness_name,
                forward_xy=forward_xy,
            ),
        ),
        EAStep("survivor_selection", survivor_selection),
    ]

    ea = EA(population, operations=ops, num_of_generations=int(args.generations))
    ea.run()
    runtime_s = time.perf_counter() - start_time

    best = ea.get_solution("best", only_alive=False)
    best_fitness = float(best.fitness) if best is not None else float("-inf")
    db_path = run_dir / "database.db"

    console.rule("[bold green]Best individual[/bold green]")
    console.log(
        f"Best {fitness_name} = {best_fitness:.4f} "
        f"(forward=({args.forward_x:g},{args.forward_y:g}), runtime={runtime_s:.1f}s)"
    )
    console.log(f"Saved DB to: {db_path}")

    save_run_summary(
        run_dir,
        best_fitness=f"{best_fitness:.6f}",
        fitness=fitness_name,
        generations=int(args.generations),
        pop=int(args.pop),
        duration_s=float(args.dur),
        runtime_s=f"{runtime_s:.2f}",
        seed=int(args.seed),
        nu=nu,
        db_path=str(db_path),
        forward_xy=f"({args.forward_x:g}, {args.forward_y:g})",
    )

    if best and not args.no_viewer:
        console.rule("[bold cyan]Viewer replay[/bold cyan]")
        replay_fit = replay_best(
            best,
            world=world,
            model=model,
            data=data,
            grid=grid,
            duration_s=float(args.dur),
            fitness_name=fitness_name,
            forward_xy=forward_xy,
        )
        console.log(f"Replay {fitness_name} = {replay_fit:.4f}")


if __name__ == "__main__":
    main()

