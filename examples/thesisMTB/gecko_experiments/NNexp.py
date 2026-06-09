"""NN-based exploration controller for the gecko robot.

A small neural network receives the robot's position, velocity, a local
visit-count gradient (4 neighboring cells), and two evolvable-frequency
gait clocks.  It outputs 8 joint angles directly — no CPG.

CMA-ES evolves the NN weights + 2 gait-clock frequencies to maximise
coverage (fraction of 10×10 grid cells visited) over a fixed duration.

The visit grid is 12×12 internally: a 10×10 scoring region surrounded by
a 1-cell ring with hardcoded high visit counts that acts as a repulsive
boundary (the NN learns to avoid high-count cells, including the ring).

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

    def xy_to_inner_cell(self, x: float, y: float) -> tuple[int, int]:
        """Map world XY to inner grid (row, col) in [0, GRID_ROWS-1] × [0, GRID_COLS-1].

        Clamps positions outside the arena to the nearest edge cell.
        """
        half_w = ARENA_WIDTH / 2.0
        half_h = ARENA_HEIGHT / 2.0
        # Normalise to [0, 1)
        fx = (x + half_w) / ARENA_WIDTH
        fy = (y + half_h) / ARENA_HEIGHT
        eps = 1e-12
        fx = float(np.clip(fx, 0.0, 1.0 - eps))
        fy = float(np.clip(fy, 0.0, 1.0 - eps))
        col = int(fx * GRID_COLS)
        row = int(fy * GRID_ROWS)
        col = int(np.clip(col, 0, GRID_COLS - 1))
        row = int(np.clip(row, 0, GRID_ROWS - 1))
        return row, col

    def _to_grid_index(self, inner_row: int, inner_col: int) -> tuple[int, int]:
        """Inner cell (0-9, 0-9) → full grid index (1-10, 1-10)."""
        return inner_row + 1, inner_col + 1

    def update(self, x: float, y: float) -> None:
        """Increment visit count if the robot crossed into a new cell."""
        inner = self.xy_to_inner_cell(x, y)
        if inner != self.prev_cell:
            gr, gc = self._to_grid_index(*inner)
            self.grid[gr, gc] += 1
            self.prev_cell = inner

    def neighbor_visits(self, x: float, y: float) -> tuple[float, float, float, float]:
        """Return normalised visit counts of the 4 neighbours (N, S, E, W).

        Uses tanh(count / 5) so the signal saturates: the difference between
        0 and 3 visits matters more than between 30 and 33.
        """
        inner_r, inner_c = self.xy_to_inner_cell(x, y)
        gr, gc = self._to_grid_index(inner_r, inner_c)
        raw = np.array([
            self.grid[gr + 1, gc],  # north  (+row = +y)
            self.grid[gr - 1, gc],  # south
            self.grid[gr, gc + 1],  # east   (+col = +x)
            self.grid[gr, gc - 1],  # west
        ], dtype=np.float64)
        normed = np.tanh(raw / 5.0)
        return float(normed[0]), float(normed[1]), float(normed[2]), float(normed[3])

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
    """12 inputs → 16 hidden (ELU) → 8 joint angles (Tanh·π/2)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(12, 16)
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
) -> tuple[float, list[tuple[float, float]]]:
    """Simulate one evaluation.  Returns (coverage_fraction, xy_trajectory)."""
    grid = ExplorationGrid()
    trajectory: list[tuple[float, float]] = []

    current_ctrl = np.zeros(model.nu)
    step = 0
    half_w = ARENA_WIDTH / 2.0
    half_h = ARENA_HEIGHT / 2.0

    while data.time < duration:
        if step % CONTROL_STEP_FREQ == 0:
            # --- Read robot state ---
            pos = data.geom_xpos[core_geom_id]
            x, y = float(pos[0]), float(pos[1])
            vel = data.qvel[:2]
            vx, vy = float(vel[0]), float(vel[1])

            # --- Update visit grid ---
            grid.update(x, y)
            trajectory.append((x, y))

            # --- Build NN inputs ---
            x_norm = x / half_w   # [-1, 1] within arena
            y_norm = y / half_h
            vel_scale = 5.0       # gecko maxes ~0.05 m/s → normalised ~0.25
            vx_norm = np.tanh(vx * vel_scale)
            vy_norm = np.tanh(vy * vel_scale)

            vn, vs, ve, vw = grid.neighbor_visits(x, y)

            t = float(data.time)
            phase_slow_sin = np.sin(2.0 * np.pi * f_slow * t)
            phase_slow_cos = np.cos(2.0 * np.pi * f_slow * t)
            phase_fast_sin = np.sin(2.0 * np.pi * f_fast * t)
            phase_fast_cos = np.cos(2.0 * np.pi * f_fast * t)

            state = np.array([
                x_norm, y_norm,
                vx_norm, vy_norm,
                vn, vs, ve, vw,
                phase_slow_sin, phase_slow_cos,
                phase_fast_sin, phase_fast_cos,
            ], dtype=np.float32)

            current_ctrl = net.forward(state)

        data.ctrl[:] = current_ctrl
        mujoco.mj_step(model, data)
        step += 1

    return grid.coverage_fraction(), trajectory


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


def _eval_worker(job: dict) -> float:
    """Evaluate one candidate across multiple starting positions."""
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

    total_coverage = 0.0
    for sx, sy in start_positions:
        _reset_to_spawn(data, model, sx, sy)
        coverage, _ = run_exploration(model, data, core_id, net, f_slow, f_fast, duration)
        total_coverage += coverage

    avg_coverage = total_coverage / len(start_positions)
    # Nevergrad minimises → negate coverage
    return -avg_coverage


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
        "start_positions": START_POSITIONS,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    best_fitness = float("inf")
    best_weights: np.ndarray | None = None
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

                fitnesses = list(pool.map(_eval_worker, jobs))

                for c, f in zip(candidates, fitnesses):
                    optimizer.tell(c, f)

                gen_best = min(fitnesses)
                if gen_best < best_fitness:
                    best_fitness = gen_best
                    best_idx = fitnesses.index(gen_best)
                    best_weights = np.array(candidates[best_idx].value)

                fitness_history.append(-gen_best)  # store as positive coverage
                mean_history.append(-float(np.mean(fitnesses)))
                progress.update(task, advance=1)
                console.log(
                    f"Gen {gen + 1}/{args.budget} | "
                    f"best_cov={-gen_best:.4f} | "
                    f"gen_best={-min(fitnesses):.4f} | "
                    f"gen_mean={-np.mean(fitnesses):.4f}"
                )

    # Final recommendation
    rec = optimizer.provide_recommendation().value
    if best_weights is None:
        best_weights = np.array(rec)

    # Save results
    np.save(run_dir / "best_weights.npy", best_weights)
    np.save(run_dir / "fitness_history.npy", np.array(fitness_history))
    np.save(run_dir / "mean_history.npy", np.array(mean_history))

    # Summary
    summary = {
        "best_coverage": -best_fitness,
        "generations": args.budget,
        "pop_size": pop_size,
        "duration_s": args.dur,
        "n_params": n_total,
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
    ax.set_ylabel("Coverage Fraction")
    ax.set_title("Coverage over Generations")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(run_dir / "fitness_history.png", dpi=150)
    plt.close(fig)

    console.log(f"Best coverage: {-best_fitness:.4f}")
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

    def control_callback(_m: mujoco.MjModel, d: mujoco.MjData) -> None:
        nonlocal current_ctrl, step
        if step % CONTROL_STEP_FREQ == 0:
            pos = d.geom_xpos[core_id]
            x, y = float(pos[0]), float(pos[1])
            vel = d.qvel[:2]
            vx, vy = float(vel[0]), float(vel[1])

            grid.update(x, y)
            trajectory.append((x, y))

            x_norm = x / half_w
            y_norm = y / half_h
            vel_scale = 5.0
            vx_norm = np.tanh(vx * vel_scale)
            vy_norm = np.tanh(vy * vel_scale)
            vn, vs, ve, vw = grid.neighbor_visits(x, y)
            t = float(d.time)

            state = np.array([
                x_norm, y_norm,
                vx_norm, vy_norm,
                vn, vs, ve, vw,
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
    console.log(f"Coverage at end of replay: {cov:.4f} ({cov * 100:.1f}%)")

    if trajectory:
        pos_data = np.array(trajectory)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Trajectory plot
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
