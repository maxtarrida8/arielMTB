"""
Run
---
uv run examples/thesisMTB/gecko_experiments/NNexp.py \\
    --budget 200 --workers 8 --dur 1200

Replay
------
uv run examples/thesisMTB/gecko_experiments/NNexp.py \\
    --replay __data__/NNexp/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import time as time_mod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import mujoco
import nevergrad as ng
import numpy as np
import torch
from matplotlib.patches import Rectangle
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
from torch import nn

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld

install()
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARENA_WIDTH = 10.0   # metres
ARENA_HEIGHT = 10.0
GRID_ROWS = 10
GRID_COLS = 10
RING_VISIT_COUNT = 999
CELL_W = ARENA_WIDTH / GRID_COLS   # 1.0 m
CELL_H = ARENA_HEIGHT / GRID_ROWS  # 1.0 m

CONTROL_STEP_FREQ = 20  # NN fires every 20 physics steps (0.04 s)

DURATION = 1200.0  # default seconds

NUM_START_POSITIONS = 4
SPAWN_MARGIN = 0.5  # stay this far from arena edge

def random_start_positions(n: int = NUM_START_POSITIONS) -> list[tuple[float, float]]:
    """Sample n random positions within the arena (with margin)."""
    half_w = ARENA_WIDTH / 2.0 - SPAWN_MARGIN
    half_h = ARENA_HEIGHT / 2.0 - SPAWN_MARGIN
    return [
        (float(np.random.uniform(-half_w, half_w)),
         float(np.random.uniform(-half_h, half_h)))
        for _ in range(n)
    ]

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA_ROOT = CWD / "__data__" / SCRIPT_NAME / "runs"
DATA_ROOT.mkdir(exist_ok=True, parents=True)


# =========================================================================== #
#                              Exploration Grid                               #
# =========================================================================== #
class ExplorationGrid:
    """12×12 visit grid: 10×10 inner scoring region + 1-cell high-count ring.

    Visit counts are incremented only on cell boundary crossings.
    """

    def __init__(self) -> None:
        # 12×12 grid: rows and cols indexed 0-11
        # Ring = row 0, row 11, col 0, col 11
        # Inner = rows 1-10, cols 1-10
        self.grid = np.zeros((GRID_ROWS + 2, GRID_COLS + 2), dtype=np.int64)
        # Fill ring with high visit count
        self.grid[0, :] = RING_VISIT_COUNT
        self.grid[-1, :] = RING_VISIT_COUNT
        self.grid[:, 0] = RING_VISIT_COUNT
        self.grid[:, -1] = RING_VISIT_COUNT

        self.prev_cell: tuple[int, int] | None = None

    def reset(self) -> None:
        self.grid[:] = 0
        self.grid[0, :] = RING_VISIT_COUNT
        self.grid[-1, :] = RING_VISIT_COUNT
        self.grid[:, 0] = RING_VISIT_COUNT
        self.grid[:, -1] = RING_VISIT_COUNT
        self.prev_cell = None

    @staticmethod
    def is_inside_arena(x: float, y: float) -> bool:
        half_w = ARENA_WIDTH / 2.0
        half_h = ARENA_HEIGHT / 2.0
        return -half_w <= x <= half_w and -half_h <= y <= half_h

    def xy_to_inner_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """Map world XY to inner grid (row, col) in [0, GRID_ROWS-1] × [0, GRID_COLS-1].

        Returns None when the position is outside the arena.
        """
        if not self.is_inside_arena(x, y):
            return None
        half_w = ARENA_WIDTH / 2.0
        half_h = ARENA_HEIGHT / 2.0
        fx = (x + half_w) / ARENA_WIDTH
        fy = (y + half_h) / ARENA_HEIGHT
        col = min(int(fx * GRID_COLS), GRID_COLS - 1)
        row = min(int(fy * GRID_ROWS), GRID_ROWS - 1)
        return row, col

    def _to_grid_index(self, inner_row: int, inner_col: int) -> tuple[int, int]:
        """Inner cell (0-9, 0-9) → full grid index (1-10, 1-10)."""
        return inner_row + 1, inner_col + 1

    def update(self, x: float, y: float) -> None:
        """Increment visit count if the robot crossed into a new cell.

        Does nothing when outside the arena and resets prev_cell so that
        re-entering counts as a fresh cell transition.
        """
        inner = self.xy_to_inner_cell(x, y)
        if inner is None:
            self.prev_cell = None
            return
        if inner != self.prev_cell:
            gr, gc = self._to_grid_index(*inner)
            self.grid[gr, gc] += 1
            self.prev_cell = inner

    _LOG1P_RING = float(np.log1p(RING_VISIT_COUNT))

    def neighbor_visits(self, x: float, y: float) -> tuple[float, float, float, float]:
        """Return normalised visit counts of the 4 neighbours (N, S, E, W).

        Uses log1p(count) / log1p(RING_VISIT_COUNT) for a smoother, less
        saturating signal than the previous tanh normalization.  Output is
        bounded to [0, 1].

        When outside the arena, returns all 1.0.
        """
        inner = self.xy_to_inner_cell(x, y)
        if inner is None:
            return 1.0, 1.0, 1.0, 1.0
        gr, gc = self._to_grid_index(*inner)
        raw = np.array([
            self.grid[gr + 1, gc],  # north  (+row = +y)
            self.grid[gr - 1, gc],  # south
            self.grid[gr, gc + 1],  # east   (+col = +x)
            self.grid[gr, gc - 1],  # west
        ], dtype=np.float64)
        normed = np.log1p(raw) / self._LOG1P_RING
        return float(normed[0]), float(normed[1]), float(normed[2]), float(normed[3])

    def neighbor_edge_flags(self, x: float, y: float) -> tuple[float, float, float, float]:
        """Return binary arena-edge flags for the 4 neighbours (N, S, E, W).

        1.0 if that neighbour lies on the arena boundary (ring), 0.0 otherwise.
        When outside the arena, returns all 1.0.
        """
        inner = self.xy_to_inner_cell(x, y)
        if inner is None:
            return 1.0, 1.0, 1.0, 1.0
        inner_r, inner_c = inner
        n_edge = 1.0 if inner_r == GRID_ROWS - 1 else 0.0
        s_edge = 1.0 if inner_r == 0 else 0.0
        e_edge = 1.0 if inner_c == GRID_COLS - 1 else 0.0
        w_edge = 1.0 if inner_c == 0 else 0.0
        return n_edge, s_edge, e_edge, w_edge

    def coverage_fraction(self) -> float:
        """Fraction of inner 10×10 cells visited at least once."""
        inner = self.grid[1:GRID_ROWS + 1, 1:GRID_COLS + 1]
        return float(np.count_nonzero(inner > 0)) / float(GRID_ROWS * GRID_COLS)

    def inner_grid(self) -> np.ndarray:
        """Return a copy of the 10×10 inner visit counts."""
        return self.grid[1:GRID_ROWS + 1, 1:GRID_COLS + 1].copy()


# =========================================================================== #
#                                NN Controller                                #
# =========================================================================== #
class ExplorationNetwork(nn.Module):
    """16 inputs → 16 hidden (ELU) → 8 joint angles (Tanh·π/2).

    Inputs: x, y, vx, vy, 4 neighbor visit counts (log1p),
    4 arena-edge flags, 4 gait-clock phases (sin/cos × 2 freqs).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 8)
        self.hidden_act = nn.ELU()
        self.output_act = nn.Tanh()

        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, state: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(state).float()
        x = self.hidden_act(self.fc1(x))
        x = self.output_act(self.fc2(x)) * (torch.pi / 2)
        return x.numpy()


def fill_parameters(net: nn.Module, vector: np.ndarray) -> None:
    """Load a flat parameter vector into the network weights."""
    t = torch.from_numpy(vector).float()
    offset = 0
    for p in net.parameters():
        n = p.numel()
        p.data.copy_(t[offset:offset + n].view_as(p))
        offset += n


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


# =========================================================================== #
#                          Build simulation context                           #
# =========================================================================== #
def _build_world(
    spawn_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """Return (model, data, core_geom_id)."""
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld(load_precompiled=False)
    world.spawn(gecko().spec, position=[spawn_xy[0], spawn_xy[1], 0.1])
    model = cast(mujoco.MjModel, world.spec.compile())
    data = mujoco.MjData(model)

    core_id = -1
    for i in range(model.ngeom):
        if "core" in model.geom(i).name:
            core_id = i
            break
    if core_id < 0:
        raise RuntimeError("Could not find 'core' geom in model")

    return model, data, core_id


# =========================================================================== #
#                              Simulation runner                              #
# =========================================================================== #
def run_exploration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    core_geom_id: int,
    net: ExplorationNetwork,
    f_slow: float,
    f_fast: float,
    duration: float,
) -> tuple[float, float, list[tuple[float, float]], list[tuple[float, float]], bool]:
    """Simulate one evaluation.

    Returns (coverage_integral, final_coverage, xy_trajectory, coverage_curve,
    diverged).
    coverage_integral is the coverage fraction time-averaged over the full
    requested duration (in [0, 1]): reaching the same final coverage earlier
    scores higher.
    final_coverage is the plain fraction of cells visited by the end.
    diverged is True when MuJoCo's auto-reset fired (numerical divergence);
    the episode stops there and the remaining time contributes zero to the
    integral, so divergence is penalised proportionally to how early it hit.
    """
    grid = ExplorationGrid()
    trajectory: list[tuple[float, float]] = []
    coverage_curve: list[tuple[float, float]] = []
    coverage_sum = 0.0

    current_ctrl = np.zeros(model.nu)
    half_w = ARENA_WIDTH / 2.0
    half_h = ARENA_HEIGHT / 2.0

    # Fixed step count: an auto-reset rewinds data.time, so a time-based
    # loop could run far longer than the requested duration.
    n_steps = int(round(duration / model.opt.timestep))
    n_samples = -(-n_steps // CONTROL_STEP_FREQ)  # control fires at 0, 20, ...
    diverged = False

    for step in range(n_steps):
        if step % CONTROL_STEP_FREQ == 0:
            # --- Read robot state ---
            pos = data.geom_xpos[core_geom_id]
            x, y = float(pos[0]), float(pos[1])
            vel = data.qvel[:2]
            vx, vy = float(vel[0]), float(vel[1])

            # --- Update visit grid ---
            grid.update(x, y)
            trajectory.append((x, y))
            cov_now = grid.coverage_fraction()
            coverage_sum += cov_now
            coverage_curve.append((float(data.time), cov_now))

            # --- Build NN inputs ---
            x_norm = x / half_w   # [-1, 1] within arena
            y_norm = y / half_h
            vel_scale = 5.0       # gecko maxes ~0.05 m/s → normalised ~0.25
            vx_norm = np.tanh(vx * vel_scale)
            vy_norm = np.tanh(vy * vel_scale)

            vn, vs, ve, vw = grid.neighbor_visits(x, y)
            en, es, ee, ew = grid.neighbor_edge_flags(x, y)

            t = float(data.time)
            phase_slow_sin = np.sin(2.0 * np.pi * f_slow * t)
            phase_slow_cos = np.cos(2.0 * np.pi * f_slow * t)
            phase_fast_sin = np.sin(2.0 * np.pi * f_fast * t)
            phase_fast_cos = np.cos(2.0 * np.pi * f_fast * t)

            state = np.array([
                x_norm, y_norm,
                vx_norm, vy_norm,
                vn, vs, ve, vw,
                en, es, ee, ew,
                phase_slow_sin, phase_slow_cos,
                phase_fast_sin, phase_fast_cos,
            ], dtype=np.float32)

            current_ctrl = net.forward(state)

        data.ctrl[:] = current_ctrl
        t_before = data.time
        mujoco.mj_step(model, data)
        if data.time <= t_before:
            # MuJoCo auto-reset on divergence rewinds the clock and teleports
            # the robot back to the compiled spawn — stop scoring here.
            diverged = True
            break

    coverage_integral = coverage_sum / max(n_samples, 1)
    return coverage_integral, grid.coverage_fraction(), trajectory, coverage_curve, diverged


# =========================================================================== #
#                     Parallel worker (one per subprocess)                    #
# =========================================================================== #
_worker_ctx: dict | None = None


def _init_worker() -> None:
    """Called once per subprocess — build the MuJoCo world and reuse it."""
    global _worker_ctx
    torch.set_num_threads(1)
    model, data, core_id = _build_world(spawn_xy=(0.0, 0.0))
    net = ExplorationNetwork()
    _worker_ctx = {
        "model": model,
        "data": data,
        "core_id": core_id,
        "net": net,
    }


def _reset_to_spawn(data: mujoco.MjData, model: mujoco.MjModel, sx: float, sy: float) -> None:
    """Reset simulation and place the robot at (sx, sy)."""
    mujoco.mj_resetData(model, data)
    data.qpos[0] = sx
    data.qpos[1] = sy
    mujoco.mj_forward(model, data)


def _eval_worker(job: dict) -> tuple[float, float, int]:
    """Evaluate one candidate across multiple starting positions.

    Returns (fitness, avg_final_coverage, n_diverged): fitness is the negated
    average coverage integral (nevergrad minimises); the average final
    coverage and the count of diverged episodes are kept for logging only.
    """
    assert _worker_ctx is not None
    model = _worker_ctx["model"]
    data = _worker_ctx["data"]
    core_id = _worker_ctx["core_id"]
    net = _worker_ctx["net"]

    weights = np.asarray(job["weights"], dtype=np.float64)
    duration = float(job["duration"])
    start_positions: list[tuple[float, float]] = job["start_positions"]

    n_nn_params = job["n_nn_params"]
    nn_weights = weights[:n_nn_params]
    f_slow = float(np.clip(np.abs(weights[n_nn_params]), 0.1, 5.0))
    f_fast = float(np.clip(np.abs(weights[n_nn_params + 1]), 0.1, 10.0))

    fill_parameters(net, nn_weights)

    total_integral = 0.0
    total_final = 0.0
    n_diverged = 0
    for sx, sy in start_positions:
        _reset_to_spawn(data, model, sx, sy)
        cov_integral, final_cov, _, _, diverged = run_exploration(
            model, data, core_id, net, f_slow, f_fast, duration)
        total_integral += cov_integral
        total_final += final_cov
        n_diverged += int(diverged)

    avg_integral = total_integral / len(start_positions)
    avg_final = total_final / len(start_positions)
    # Nevergrad minimises → negate the coverage integral
    return -avg_integral, avg_final, n_diverged


# =========================================================================== #
#                             Evolution loop                                  #
# =========================================================================== #
def evolve(args: argparse.Namespace) -> tuple[np.ndarray, Path]:
    n_nn_params = count_parameters(ExplorationNetwork())
    n_total = n_nn_params + 2  # + f_slow, f_fast
    console.log(f"NN params: {n_nn_params}, total CMA-ES dimensions: {n_total}")

    # CMA-ES population sizing: at least 4 + floor(3·ln(n))
    min_pop = 4 + int(3 * np.log(max(n_total, 2)))
    pop_size = max(args.workers, min_pop)
    if pop_size % 2 != 0:
        pop_size += 1

    budget = args.budget * pop_size
    console.log(
        f"Budget: {args.budget} generations × {pop_size} pop = {budget} evals | "
        f"Workers: {args.workers} | Duration: {args.dur}s"
    )

    # Initialisation: NN weights ~ U(-0.5, 0.5), frequencies ~ [1.0, 2.0]
    init = np.zeros(n_total)
    init[:n_nn_params] = np.random.uniform(-0.5, 0.5, n_nn_params)
    init[n_nn_params] = 1.0      # f_slow initial
    init[n_nn_params + 1] = 2.0  # f_fast initial

    param = ng.p.Array(init=init)
    param.set_mutation(sigma=0.3)

    cma_config = ng.optimizers.ParametrizedCMA(popsize=pop_size)
    optimizer = cma_config(parametrization=param, budget=budget, num_workers=pop_size)

    # Run directory
    timestamp = time_mod.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or f"nn_b{args.budget}_d{int(args.dur)}"
    run_dir = DATA_ROOT / f"{timestamp}___{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "script": str(Path(__file__).relative_to(CWD)),
        "args": vars(args),
        "n_nn_params": n_nn_params,
        "n_total_params": n_total,
        "pop_size": pop_size,
        "budget_evals": budget,
        "start_positions": "randomized_per_generation",
        "num_start_positions": NUM_START_POSITIONS,
        "spawn_margin": SPAWN_MARGIN,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    best_fitness = float("inf")
    best_weights: np.ndarray | None = None
    best_avg_final_coverage: float | None = None
    fitness_history: list[float] = []
    mean_history: list[float] = []

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Evolving", total=args.budget)

            for gen in range(args.budget):
                starts = random_start_positions()
                candidates = [optimizer.ask() for _ in range(pop_size)]
                jobs = [
                    {
                        "weights": c.value,
                        "duration": args.dur,
                        "start_positions": starts,
                        "n_nn_params": n_nn_params,
                    }
                    for c in candidates
                ]

                results = list(pool.map(_eval_worker, jobs))
                fitnesses = [r[0] for r in results]
                avg_final_covs = [r[1] for r in results]
                n_diverged = sum(r[2] for r in results)

                for c, f in zip(candidates, fitnesses):
                    optimizer.tell(c, f)

                gen_best = min(fitnesses)
                gen_best_idx = fitnesses.index(gen_best)
                if gen_best < best_fitness:
                    best_fitness = gen_best
                    best_weights = np.array(candidates[gen_best_idx].value)
                    best_avg_final_coverage = avg_final_covs[gen_best_idx]

                fitness_history.append(-gen_best)  # store as positive coverage integral
                mean_history.append(-float(np.mean(fitnesses)))
                progress.update(task, advance=1)
                console.log(
                    f"Gen {gen + 1}/{args.budget} | "
                    f"best_int={-best_fitness:.4f} | "
                    f"gen_best_int={-gen_best:.4f} | "
                    f"gen_mean_int={-np.mean(fitnesses):.4f} | "
                    f"gen_best_avg_endcov={avg_final_covs[gen_best_idx]:.4f}"
                )
                if n_diverged:
                    console.log(
                        f"[yellow]Warning: {n_diverged} diverged episode(s) "
                        f"in gen {gen + 1} (penalised in fitness)[/yellow]"
                    )

    # Final recommendation
    rec = optimizer.provide_recommendation().value
    if best_weights is None:
        best_weights = np.array(rec)

    # Save results
    np.save(run_dir / "best_weights.npy", best_weights)
    np.save(run_dir / "fitness_history.npy", np.array(fitness_history))
    np.save(run_dir / "mean_history.npy", np.array(mean_history))

    # Post-evolution evaluation: save coverage-over-time curve from (0, 0)
    nn_w = best_weights[:n_nn_params]
    f_s = float(np.clip(np.abs(best_weights[n_nn_params]), 0.1, 5.0))
    f_f = float(np.clip(np.abs(best_weights[n_nn_params + 1]), 0.1, 10.0))
    eval_model, eval_data, eval_core = _build_world(spawn_xy=(0.0, 0.0))
    eval_net = ExplorationNetwork()
    fill_parameters(eval_net, nn_w)
    _, _, _, curve, _ = run_exploration(
        eval_model, eval_data, eval_core, eval_net, f_s, f_f, args.dur)
    np.save(run_dir / "coverage_curve.npy", np.array(curve))

    # Summary
    summary = {
        "fitness_metric": "time_averaged_coverage_integral",
        "best_coverage_integral": -best_fitness,
        "best_avg_final_coverage": best_avg_final_coverage,
        "generations": args.budget,
        "pop_size": pop_size,
        "duration_s": args.dur,
        "n_params": n_total,
        "seed": args.seed,
        "f_slow": float(np.clip(np.abs(best_weights[n_nn_params]), 0.1, 5.0)),
        "f_fast": float(np.clip(np.abs(best_weights[n_nn_params + 1]), 0.1, 10.0)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Fitness plot
    generations = list(range(1, len(fitness_history) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(generations, fitness_history, label="Best (gen)")
    ax.plot(generations, mean_history, color="red", alpha=0.7, label="Mean (gen)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Coverage Integral (time-averaged)")
    ax.set_title("Coverage Integral over Generations")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(run_dir / "fitness_history.png", dpi=150)
    plt.close(fig)

    console.log(f"Best coverage integral: {-best_fitness:.4f}")
    if best_avg_final_coverage is not None:
        console.log(
            f"Avg final coverage of best candidate (over starts): "
            f"{best_avg_final_coverage:.4f}"
        )
    console.log(f"Run saved to: {run_dir}")

    return best_weights, run_dir


# =========================================================================== #
#                                  Replay                                     #
# =========================================================================== #
def run_replay(replay_dir: Path, duration: float) -> None:
    weights_path = replay_dir / "best_weights.npy"
    config_path = replay_dir / "config.json"
    if not weights_path.is_file():
        raise FileNotFoundError(f"best_weights.npy not found in {replay_dir}")

    weights = np.load(weights_path)
    n_nn_params = count_parameters(ExplorationNetwork())

    if config_path.is_file():
        cfg = json.loads(config_path.read_text())
        n_nn_params = cfg.get("n_nn_params", n_nn_params)

    nn_weights = weights[:n_nn_params]
    f_slow = float(np.clip(np.abs(weights[n_nn_params]), 0.1, 5.0))
    f_fast = float(np.clip(np.abs(weights[n_nn_params + 1]), 0.1, 10.0))

    console.log(f"Replaying from: {replay_dir}")
    console.log(f"  f_slow={f_slow:.3f} Hz, f_fast={f_fast:.3f} Hz")

    net = ExplorationNetwork()
    fill_parameters(net, nn_weights)

    # Build world at centre spawn for replay
    model, data, core_id = _build_world(spawn_xy=(0.0, 0.0))
    grid = ExplorationGrid()
    trajectory: list[tuple[float, float]] = []

    half_w = ARENA_WIDTH / 2.0
    half_h = ARENA_HEIGHT / 2.0
    current_ctrl = np.zeros(model.nu)
    step = 0
    coverage_sum = 0.0
    n_cov_samples = 0
    coverage_curve: list[tuple[float, float]] = []

    def control_callback(_m: mujoco.MjModel, d: mujoco.MjData) -> None:
        nonlocal current_ctrl, step, coverage_sum, n_cov_samples
        if step % CONTROL_STEP_FREQ == 0:
            pos = d.geom_xpos[core_id]
            x, y = float(pos[0]), float(pos[1])
            vel = d.qvel[:2]
            vx, vy = float(vel[0]), float(vel[1])

            grid.update(x, y)
            trajectory.append((x, y))
            cov_now = grid.coverage_fraction()
            coverage_sum += cov_now
            n_cov_samples += 1
            coverage_curve.append((float(d.time), cov_now))

            x_norm = x / half_w
            y_norm = y / half_h
            vel_scale = 5.0
            vx_norm = np.tanh(vx * vel_scale)
            vy_norm = np.tanh(vy * vel_scale)
            vn, vs, ve, vw = grid.neighbor_visits(x, y)
            en, es, ee, ew = grid.neighbor_edge_flags(x, y)
            t = float(d.time)

            state = np.array([
                x_norm, y_norm,
                vx_norm, vy_norm,
                vn, vs, ve, vw,
                en, es, ee, ew,
                np.sin(2.0 * np.pi * f_slow * t),
                np.cos(2.0 * np.pi * f_slow * t),
                np.sin(2.0 * np.pi * f_fast * t),
                np.cos(2.0 * np.pi * f_fast * t),
            ], dtype=np.float32)

            current_ctrl = net.forward(state)

        d.ctrl[:] = current_ctrl
        step += 1

    mujoco.set_mjcb_control(control_callback)

    console.rule("[bold cyan]Viewer replay — close window when done[/bold cyan]")
    viewer.launch(model=model, data=data)

    # Post-replay analysis
    cov = grid.coverage_fraction()
    cov_integral = coverage_sum / max(n_cov_samples, 1)
    console.log(f"Final coverage at end of replay: {cov:.4f} ({cov * 100:.1f}%)")
    console.log(f"Coverage integral (time-averaged over replay): {cov_integral:.4f}")

    # Save coverage curve to the run directory
    if coverage_curve:
        np.save(replay_dir / "coverage_curve_replay.npy", np.array(coverage_curve))
        console.log(f"Coverage curve saved to: {replay_dir / 'coverage_curve_replay.npy'}")

    if trajectory:
        pos_data = np.array(trajectory)
        curve_data = np.array(coverage_curve) if coverage_curve else None
        n_plots = 3 if curve_data is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))

        # Trajectory plot
        ax1 = axes[0]
        ax1.plot(pos_data[:, 0], pos_data[:, 1], "b-", linewidth=0.5)
        ax1.plot(pos_data[0, 0], pos_data[0, 1], "go", markersize=8, label="Start")
        ax1.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", markersize=8, label="End")
        ax1.set_xlim(-half_w, half_w)
        ax1.set_ylim(-half_h, half_h)
        rect = Rectangle(
            (-half_w, -half_h), ARENA_WIDTH, ARENA_HEIGHT,
            linewidth=1.5, edgecolor="gray", facecolor="none", linestyle="--",
        )
        ax1.add_patch(rect)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title("Robot Trajectory")
        ax1.set_aspect("equal")
        ax1.legend()
        ax1.grid(True)

        # Heatmap of visit counts
        ax2 = axes[1]
        inner = grid.inner_grid()
        im = ax2.imshow(
            inner, origin="lower", cmap="YlOrRd", interpolation="nearest",
            extent=[-half_w, half_w, -half_h, half_h],
        )
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title(f"Visit Heatmap — {cov * 100:.1f}% coverage")
        ax2.set_aspect("equal")
        fig.colorbar(im, ax=ax2, label="Visits")

        # Coverage over time
        if curve_data is not None:
            ax3 = axes[2]
            ax3.plot(curve_data[:, 0], curve_data[:, 1] * 100, "b-", linewidth=1)
            for thresh in [50, 80]:
                ax3.axhline(thresh, color="gray", linestyle="--", linewidth=0.8)
                hits = curve_data[curve_data[:, 1] * 100 >= thresh]
                if len(hits) > 0:
                    t_hit = hits[0, 0]
                    ax3.axvline(t_hit, color="red", linestyle=":", linewidth=0.8)
                    ax3.annotate(
                        f"{thresh}% @ {t_hit:.0f}s",
                        xy=(t_hit, thresh), xytext=(10, 5),
                        textcoords="offset points", fontsize=8,
                    )
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Coverage (%)")
            ax3.set_title("Coverage over Time")
            ax3.set_ylim(0, 105)
            ax3.grid(True)

        fig.tight_layout()
        plt.show()


# =========================================================================== #
#                                   Main                                      #
# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description="NN exploration controller")
    parser.add_argument("--budget", type=int, default=200,
                        help="Number of CMA-ES generations")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                        help="Parallel worker processes")
    parser.add_argument("--dur", type=float, default=DURATION,
                        help="Simulation duration per evaluation (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--replay", type=str, default=None,
                        help="Path to a run directory to replay")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.replay is not None:
        p = Path(args.replay).expanduser().resolve()
        if p.is_file():
            p = p.parent
        run_replay(p, duration=args.dur)
        return

    _, run_dir = evolve(args)
    console.rule("[bold green]Evolution complete — launching replay[/bold green]")
    run_replay(run_dir, duration=args.dur)


if __name__ == "__main__":
    main()
