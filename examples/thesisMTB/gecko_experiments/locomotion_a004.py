"""GECKO locomotion baseline using the same EA pipeline as
`examples/c_genotypes/2_body_brain_evolution_tree_fast.py`.

Morphology is fixed to the prebuilt `gecko()` body; only the `SimpleCPG`
controller is evolved, using the same `init_ctrl_prior` / crossover / mutation
/ novelty / stagnation heuristics and the same progress-shaped target fitness
as the fast tree co-evolution example.

The goal is a sanity check that separates "EA and locomotion signal" from
"exploration fitness too sparse for early search" when compared to
`gecko_exploration.py`.

Run
---
uv run examples/thesisMTB/gecko_locomotion.py --budget 20 --pop 30 --dur 20
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Literal

import ariel.ec.a004 as ea_mod
import mujoco
import numpy as np
import torch
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EASettings, EAStep
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.simple_cpg import (
    SimpleCPG,
    create_fully_connected_adjacency,
)
from ariel.simulation.environments._simple_flat_with_target import (
    SimpleFlatWorldWithTarget,
)
from ariel.utils.renderers import video_renderer
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from mujoco import viewer
from rich.console import Console
from rich.progress import track
from rich.traceback import install

# Local helpers (same folder).
sys.path.insert(0, str(Path(__file__).parent))
from _run_utils import (  # noqa: E402
    make_run_dir,
    save_run_config,
    save_run_summary,
)

install()
console = Console()

SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.5])

Population = list[Individual]
ViewerTypes = Literal["launcher", "video", "simple"]


class Evolution:
    """Same controller evolution loop as 2_body_brain_evolution_tree_fast (no morph)."""

    def __init__(
        self,
        *,
        nu: int,
        rng: np.random.Generator,
        budget: int,
        pop_size: int,
        duration_max: int,
        data_dir: Path,
    ) -> None:
        self._nu = int(nu)
        self._rng = rng
        self._budget = int(budget)
        self._pop_size = int(pop_size)
        self._duration_max = int(duration_max)
        self._data_dir = data_dir
        self.config = EASettings(
            is_maximisation=False,
            num_of_generations=self._budget,
            target_population_size=self._pop_size,
            db_file_path=self._data_dir / "database.db",
            db_handling="delete",
        )
        ea_mod.config = self.config
        self.generation = 0
        self.best_avg_fitness_seen = float("inf")
        self.stagnation_generations = 0

    def get_eval_duration(self) -> int:
        if self.generation < 10:
            return min(5, self._duration_max)
        if self.generation < 25:
            return min(10, self._duration_max)
        return self._duration_max

    def map_genotype_to_brain(self, cpg: SimpleCPG, full_genome: list[float]) -> None:
        n = cpg.phase.shape[0]
        params = np.array(full_genome)
        required_size = n * 5
        if required_size > len(params):
            params = np.resize(params, required_size)
        else:
            params = params[:required_size]

        p_phase = params[0 * n : 1 * n]
        p_w = params[1 * n : 2 * n]
        p_amp = params[2 * n : 3 * n]
        p_ha = params[3 * n : 4 * n]
        p_b = params[4 * n : 5 * n]

        with torch.no_grad():
            cpg.phase.data.copy_(torch.from_numpy(p_phase * np.pi).float())
            cpg.w.data.copy_(torch.from_numpy(0.2 + (3.8 * (p_w + 1.0) / 2.0)).float())
            cpg.amplitudes.data.copy_(
                torch.from_numpy(0.5 + (3.5 * (p_amp + 1.0) / 2.0)).float()
            )
            cpg.ha.data.copy_(torch.from_numpy(p_ha * 2.0).float())
            cpg.b.data.copy_(torch.from_numpy(p_b * 0.5).float())

    def init_ctrl_prior(self) -> list[float]:
        n = self._nu
        phase = np.linspace(-0.5, 0.5, n)
        w = np.full(n, 0.0)
        amp = np.full(n, 0.2)
        ha = np.zeros(n)
        b = np.zeros(n)
        base = np.concatenate([phase, w, amp, ha, b])
        noise = self._rng.normal(0.0, 0.15, size=base.shape)
        return np.clip(base + noise, -1.0, 1.0).tolist()

    def mutate_ctrl_vector(self, genome: list[float]) -> list[float]:
        arr = np.array(genome)
        mask = self._rng.random(arr.shape) < 0.35
        noise = self._rng.normal(0, 0.4, arr.shape)
        arr[mask] += noise[mask]
        return np.clip(arr, -1.0, 1.0).tolist()

    def crossover_ctrl_vectors(self, ctrl1: list[float], ctrl2: list[float]) -> list[float]:
        arr1 = np.array(ctrl1)
        arr2 = np.array(ctrl2)
        size = max(len(arr1), len(arr2))
        if len(arr1) < size:
            arr1 = np.resize(arr1, size)
        if len(arr2) < size:
            arr2 = np.resize(arr2, size)
        mask = self._rng.random(size) < 0.5
        return np.where(mask, arr1, arr2).tolist()

    def create_individual(self) -> Individual:
        ind = Individual()
        ind.genotype = self.init_ctrl_prior()
        ind.tags = {
            "ps": False,
            "valid": True,
            "debug_joints": -1,
            "seed": int(self._rng.integers(0, 2**31, endpoint=False)),
        }
        return ind

    def reproduction(self, population: Population) -> Population:
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            parents = list(population)

        new_offspring: list[Individual] = []
        target_pool = self.config.target_population_size * 2
        offspring_counter = 0
        max_iterations = target_pool * 3
        iterations = 0

        ctrl_mut_prob = 0.6 if self.generation < 10 else (0.45 if self.generation < 20 else 0.3)
        if self.stagnation_generations >= 2:
            ctrl_mut_prob = min(0.75, ctrl_mut_prob + 0.15)

        novelty_period = 5 if self.stagnation_generations < 2 else 2

        while len(population) + len(new_offspring) < target_pool and iterations < max_iterations:
            iterations += 1
            if offspring_counter > 0 and offspring_counter % novelty_period == 0:
                new_offspring.append(self.create_individual())
                offspring_counter += 1
                continue

            use_sexual = len(parents) >= 2 and self._rng.random() < 0.7
            if use_sexual:
                p1, p2 = random.sample(parents, 2)
                c_ctrl = self.crossover_ctrl_vectors(
                    list(p1.genotype), list(p2.genotype)
                )
            else:
                parent = random.choice(parents)
                c_ctrl = list(parent.genotype).copy()

            if self._rng.random() < ctrl_mut_prob:
                c_ctrl = self.mutate_ctrl_vector(c_ctrl)

            ind = Individual()
            ind.genotype = c_ctrl
            ind.tags = {
                "ps": False,
                "valid": True,
                "debug_joints": -1,
                "seed": int(self._rng.integers(0, 2**31, endpoint=False)),
            }
            ind.requires_eval = True
            new_offspring.append(ind)
            offspring_counter += 1

        if iterations >= max_iterations:
            console.log(
                f"[yellow]Warning: Reproduction hit iteration limit. "
                f"Generated {len(new_offspring)} offspring "
                f"(target: {target_pool - len(population)})[/yellow]"
            )

        population.extend(new_offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        to_eval = [
            ind
            for ind in population
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]
        if not to_eval:
            return population

        duration = float(self.get_eval_duration())
        for ind in track(to_eval, description=f"Evaluating (dur={duration})..."):
            fitness = self.run_simulation("simple", ind, duration=duration)
            ind.fitness = fitness
            ind.requires_eval = False
        return population

    def parent_selection(self, population: Population) -> Population:
        population.sort(
            key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf")
        )
        cutoff = max(2, int(len(population) * 0.35))
        for i, ind in enumerate(population):
            ind.tags = {"ps": i < cutoff}
        ps_count = sum(1 for ind in population if ind.tags.get("ps", False))
        console.log(f"[cyan]Parent Selection: {ps_count}/{len(population)}[/cyan]")
        return population

    def survivor_selection(self, population: Population) -> Population:
        population.sort(
            key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf")
        )
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False

        good = [
            ind.fitness_
            for ind in survivors
            if ind.fitness_ is not None and ind.fitness_ != float("inf")
        ]
        avg_fitness = float(np.mean(good)) if good else float("inf")

        if avg_fitness < (self.best_avg_fitness_seen - 1e-2):
            self.best_avg_fitness_seen = float(avg_fitness)
            self.stagnation_generations = 0
        else:
            self.stagnation_generations += 1

        console.log(
            f"[green]Gen {self.generation:02d} | Avg fitness = {avg_fitness:.4f} | "
            f"Eval dur = {self.get_eval_duration()} | Stag = {self.stagnation_generations}[/green]"
        )
        self.generation += 1
        return population

    def run_simulation(self, mode: ViewerTypes, ind: Individual, duration: float) -> float:
        """Fixed GECKO body + `SimpleCPG`; same reward as tree_fast (progress-shaped)."""
        mujoco.set_mjcb_control(None)

        world = SimpleFlatWorldWithTarget(load_precompiled=False)
        world.spec.body("target_marker").pos = TARGET_POSITION.tolist()
        world.spawn(gecko().spec, position=SPAWN_POSITION)
        model = world.spec.compile()
        data = mujoco.MjData(model)
        if model.nu == 0:
            return float("inf")

        adj_dict = create_fully_connected_adjacency(model.nu)
        cpg = SimpleCPG(adj_dict)
        self.map_genotype_to_brain(cpg, list(ind.genotype))

        tracker = Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"])
        ctrl = Controller(
            controller_callback_function=lambda m, d, *a, **k: cpg.forward(d.time),
            time_steps_per_save=20,
            tracker=tracker,
        )
        ctrl.tracker.setup(world.spec, data)
        mujoco.set_mjcb_control(
            lambda m, d: ctrl.set_control(m, d, duration=duration)
        )

        sim_seed = ind.tags.get("seed", 0)
        np.random.seed(int(sim_seed) % (2**31))
        mujoco.mj_resetData(model, data)

        if mode == "simple":
            steps_required = int(duration / model.opt.timestep)
            for _ in range(steps_required):
                mujoco.mj_step(model, data)
        elif mode == "video":
            rec_dir = self._data_dir / "videos"
            rec_dir.mkdir(exist_ok=True, parents=True)
            recorder = VideoRecorder(
                output_folder=str(rec_dir), file_name=f"gecko_loco_{ind.id}"
            )
            video_renderer(model, data, duration=duration, video_recorder=recorder)
        elif mode == "launcher":
            viewer.launch(model=model, data=data)

        if not tracker.history["xpos"]:
            return float("inf")

        first_key = list(tracker.history["xpos"].keys())[0]
        traj = tracker.history["xpos"][first_key]
        if not traj:
            return float("inf")

        delay_time = min(1.0, float(duration))
        sample_dt = model.opt.timestep * ctrl.time_steps_per_save
        if sample_dt <= 0:
            start_idx = 0
        else:
            start_idx = int(np.ceil(delay_time / sample_dt))
        start_idx = min(max(0, start_idx), max(0, len(traj) - 1))

        mid_idx = start_idx + (len(traj) - start_idx) // 2
        mid_idx = min(max(start_idx + 1, mid_idx), len(traj) - 1)

        pos_start = np.array(traj[start_idx])
        pos_mid = np.array(traj[mid_idx])
        pos_final = np.array(traj[-1])

        to_target_xy = TARGET_POSITION[:2] - pos_start[:2]
        target_norm = float(np.linalg.norm(to_target_xy))
        if target_norm < 1e-8:
            target_dir = np.array([1.0, 0.0])
        else:
            target_dir = to_target_xy / target_norm

        move1_xy = pos_mid[:2] - pos_start[:2]
        move2_xy = pos_final[:2] - pos_mid[:2]
        progress1 = float(np.dot(move1_xy, target_dir))
        progress2 = float(np.dot(move2_xy, target_dir))
        sustained_progress = min(progress1, progress2)
        total_progress = progress1 + progress2
        total_move_xy = pos_final[:2] - pos_start[:2]
        total_forward = float(np.dot(total_move_xy, target_dir))
        lateral_drift = float(
            np.linalg.norm(total_move_xy - total_forward * target_dir)
        )
        dist_final = float(
            np.linalg.norm(pos_final[:2] - TARGET_POSITION[:2])
        )
        return float(
            dist_final
            - 1.0 * total_progress
            - 1.5 * max(0.0, sustained_progress)
            + 0.3 * lateral_drift
        )

    def evolve(self) -> Individual | None:
        console.log("Initializing population (GECKO body, no morphology evolution)...")
        population: Population = [self.create_individual() for _ in range(self._pop_size)]
        population = self.evaluate(population)

        ops = [
            EAStep("parent_selection", self.parent_selection),
            EAStep("reproduction", self.reproduction),
            EAStep("evaluation", self.evaluate),
            EAStep("survivor_selection", self.survivor_selection),
        ]
        ea = EA(population, operations=ops, num_of_generations=self._budget)
        ea.run()
        return ea.get_solution("best", only_alive=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GECKO locomotion: same EA as 2_body_brain_evolution_tree_fast, fixed body"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Number of generations (same as tree_fast EA)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="If set, overrides --budget (alias for older scripts).",
    )
    parser.add_argument(
        "--pop", type=int, default=30, help="Population size (tree_fast default: 30)"
    )
    parser.add_argument(
        "--dur", type=int, default=20, help="Max simulation duration (seconds)"
    )
    parser.add_argument(
        "--final-viewer",
        type=str,
        default="none",
        choices=["none", "simple", "video", "launcher"],
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Do not open a viewer after the run (same as --final-viewer none).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional human-readable label appended to the run folder name.",
    )
    args = parser.parse_args()

    budget = int(args.generations) if args.generations is not None else int(args.budget)
    pop = int(args.pop)
    duration = int(args.dur)
    seed = int(args.seed)
    final_v = "none" if args.no_viewer else args.final_viewer

    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    run_dir = make_run_dir(Path(__file__), args.run_name)
    save_run_config(
        run_dir,
        args,
        extras={
            "resolved": {
                "budget": budget,
                "pop": pop,
                "duration": duration,
                "final_viewer": final_v,
            },
            "spawn_position": list(SPAWN_POSITION),
            "target_position": TARGET_POSITION.tolist(),
        },
    )
    console.log(f"Run folder: {run_dir}")

    mujoco.set_mjcb_control(None)
    w = SimpleFlatWorldWithTarget(load_precompiled=False)
    w.spec.body("target_marker").pos = TARGET_POSITION.tolist()
    w.spawn(gecko().spec, position=SPAWN_POSITION)
    nu = int(w.spec.compile().nu)

    evo = Evolution(
        nu=nu,
        rng=rng,
        budget=budget,
        pop_size=pop,
        duration_max=duration,
        data_dir=run_dir,
    )
    start_time = time.perf_counter()
    best = evo.evolve()
    runtime_s = time.perf_counter() - start_time

    db_path = run_dir / "database.db"
    best_fitness = float(best.fitness) if best is not None else float("inf")
    if best:
        console.rule("[bold green]Final Best (GECKO, tree_fast EA)[/bold green]")
        console.log(
            f"Best fitness: {best_fitness:.4f} (runtime {runtime_s:.1f}s)"
        )
    console.log(f"Saved DB to: {db_path}")

    save_run_summary(
        run_dir,
        best_fitness=f"{best_fitness:.6f}",
        budget=budget,
        pop=pop,
        duration_max_s=duration,
        runtime_s=f"{runtime_s:.2f}",
        seed=seed,
        nu=nu,
        db_path=str(db_path),
        spawn_position=str(SPAWN_POSITION),
        target_position=str(TARGET_POSITION.tolist()),
    )

    if best and final_v != "none":
        console.rule(f"[bold cyan]Replay: {final_v}[/bold cyan]")
        evo.run_simulation(final_v, best, duration=float(duration))


if __name__ == "__main__":
    main()
