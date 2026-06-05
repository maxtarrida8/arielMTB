"""Compare all exploration_cpg runs at a glance.

Scans every subdirectory under ``__data__/exploration_cpg/runs/``, parses
``summary.txt`` (or the first ``.txt`` file found as a fallback for older
runs), and displays:

  1. A Rich table sorted by ``best_fitness`` (descending).
  2. A matplotlib bar chart with one bar per run, coloured by fitness function.

Run
---
    uv run examples/thesisMTB/compare_runs.py

Optional flags
--------------
  --runs-dir PATH   Override the default runs directory.
  --no-plot         Print the table only, skip the plot.
  --min-budget N    Only show runs with budget >= N.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Shared grid — must match the definition in gecko_experiments/_fitness_registry.py.
# Import ARENA_GRID from there so both files always stay in sync.
# ---------------------------------------------------------------------------
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent / "gecko_experiments"))
from _fitness_registry import ARENA_GRID  # noqa: E402
from ariel.simulation.tasks.exploration import GridSpec


def time_to_coverage(
    xy: np.ndarray,
    grid: GridSpec,
    *,
    x_percent: float,
    dt: float = 1.0,
) -> float | None:
    """Smallest time t such that the robot has covered >= x_percent% of the grid.

    Parameters
    ----------
    xy : np.ndarray, shape (T, 2)
        XY trajectory samples in world coordinates.
    grid : GridSpec
        Grid specification (must match the one used during the run).
    x_percent : float
        Target coverage in [0, 100].
    dt : float
        Time between consecutive xy samples in seconds.

    Returns
    -------
    float | None
        Time (seconds) to reach the target coverage, or None if never reached.
    """
    target = float(np.clip(float(x_percent) / 100.0, 0.0, 1.0))
    if target <= 0.0:
        return 0.0
    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        return None

    N = np.zeros((int(grid.nrow), int(grid.ncol)), dtype=np.int64)
    visited_mask = np.zeros_like(N, dtype=bool)
    V_count = 0
    total_cells = int(N.size)

    for k, (x, y) in enumerate(arr):
        r, c = grid.xy_to_cell(float(x), float(y))
        N[r, c] += 1
        if not visited_mask[r, c]:
            visited_mask[r, c] = True
            V_count += 1
            if (V_count / total_cells) >= target:
                return float((k + 1) * float(dt))
    return None

console = Console()

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNS_DIR = _REPO_ROOT / "__data__" / "exploration_cpg" / "runs"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    folder: Path
    run_name: str

    # From summary / txt
    best_fitness: float = float("nan")
    fitness: str = "?"
    budget: int = 0
    num_workers: int = 0
    duration_s: float = 0.0
    runtime_s: float = 0.0
    seed: int = 0

    # From config.json
    start_time: str = ""
    walled: bool = False

    # Computed from best_trajectory.npy (if saved)
    t25_s: float | None = None   # seconds to reach 25% coverage
    t50_s: float | None = None   # seconds to reach 50% coverage

    # Raw lines (for debugging)
    _raw: dict[str, str] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def _parse_txt(path: Path) -> dict[str, str]:
    """Parse a ``key: value`` text file (summary.txt style)."""
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _find_summary(folder: Path) -> dict[str, str]:
    """Return parsed summary dict, preferring summary.txt."""
    canonical = folder / "summary.txt"
    if canonical.exists():
        return _parse_txt(canonical)
    # Fallback: first .txt file in the folder
    for txt in sorted(folder.glob("*.txt")):
        return _parse_txt(txt)
    return {}


def _parse_config(folder: Path) -> dict:
    cfg_path = folder / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _is_walled(folder: Path, cfg: dict) -> bool:
    """Heuristic: check folder name or config for 'walled' keyword."""
    if "walled" in folder.name.lower():
        return True
    cmd = cfg.get("command", "")
    return "walled" in cmd.lower()


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------
def load_run(folder: Path) -> RunRecord | None:
    summary = _find_summary(folder)
    cfg = _parse_config(folder)

    if not summary and not cfg:
        return None  # empty / incomplete run folder

    rec = RunRecord(
        folder=folder,
        run_name=folder.name,
        _raw=summary,
    )

    # --- summary fields ---
    try:
        rec.best_fitness = float(summary.get("best_fitness", "nan"))
    except ValueError:
        pass
    rec.fitness = summary.get("fitness") or cfg.get("extras", {}).get("fitness", "?")
    try:
        rec.budget = int(summary.get("budget", cfg.get("args", {}).get("budget", 0)))
    except (ValueError, TypeError):
        pass
    try:
        rec.num_workers = int(summary.get("num_workers", 0))
    except ValueError:
        pass
    try:
        rec.duration_s = float(summary.get("duration_s", 0))
    except ValueError:
        pass
    try:
        rec.runtime_s = float(summary.get("runtime_s", 0))
    except ValueError:
        pass
    try:
        rec.seed = int(summary.get("seed", 0))
    except ValueError:
        pass

    # --- config fields ---
    rec.start_time = cfg.get("start_time", "")
    rec.walled = _is_walled(folder, cfg)

    # --- trajectory-derived fields ---
    traj_path = folder / "best_trajectory.npy"
    if traj_path.exists():
        try:
            xy = np.load(traj_path)
            # Derive sample_dt from duration and trajectory length
            duration_s = rec.duration_s if rec.duration_s > 0 else 30.0
            sample_dt = duration_s / max(len(xy) - 1, 1)
            grid = ARENA_GRID
            rec.t25_s = time_to_coverage(xy, grid, x_percent=25.0, dt=sample_dt)
            rec.t50_s = time_to_coverage(xy, grid, x_percent=50.0, dt=sample_dt)
        except Exception:
            pass

    return rec


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def _runtime_str(seconds: float) -> str:
    if seconds <= 0:
        return "-"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_table(records: list[RunRecord]) -> None:
    table = Table(
        title="Exploration CPG Runs",
        show_lines=True,
        highlight=True,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Run name", style="cyan", no_wrap=False, max_width=40)
    table.add_column("Fitness", style="magenta")
    table.add_column("Best", justify="right", style="bold green")
    table.add_column("Budget", justify="right")
    table.add_column("Workers", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Runtime", justify="right")
    table.add_column("Seed", justify="right")
    table.add_column("Walled", justify="center")
    table.add_column("t25 (s)", justify="right", style="yellow")
    table.add_column("t50 (s)", justify="right", style="yellow")
    table.add_column("Date", style="dim")

    for i, r in enumerate(records, 1):
        fit_str = f"{r.best_fitness:.6f}" if not np.isnan(r.best_fitness) else "N/A"
        table.add_row(
            str(i),
            r.run_name,
            r.fitness,
            fit_str,
            str(r.budget) if r.budget else "-",
            str(r.num_workers) if r.num_workers else "-",
            f"{r.duration_s:.0f}s" if r.duration_s else "-",
            _runtime_str(r.runtime_s),
            str(r.seed) if r.seed else "-",
            "Y" if r.walled else "-",
            f"{r.t25_s:.1f}" if r.t25_s is not None else "-",
            f"{r.t50_s:.1f}" if r.t50_s is not None else "-",
            r.start_time[:10] if r.start_time else "-",
        )

    console.print(table)
    console.print(f"[dim]Total runs shown: {len(records)}[/dim]")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_runs(records: list[RunRecord]) -> None:
    if not records:
        return

    # Assign a colour per unique fitness function
    fitness_names = sorted({r.fitness for r in records})
    cmap = cm.get_cmap("tab10", max(len(fitness_names), 1))
    colour_map = {name: cmap(i) for i, name in enumerate(fitness_names)}

    fig, ax = plt.subplots(figsize=(max(8, len(records) * 0.9), 5))

    x_pos = np.arange(len(records))
    bar_colours = [colour_map[r.fitness] for r in records]
    bar_heights = [
        r.best_fitness if not np.isnan(r.best_fitness) else 0.0
        for r in records
    ]

    bars = ax.bar(x_pos, bar_heights, color=bar_colours, edgecolor="white", linewidth=0.5)

    # Hatch walled runs
    for bar, rec in zip(bars, records):
        if rec.walled:
            bar.set_hatch("//")
            bar.set_edgecolor("black")

    # X-axis labels: short run name
    short_names = []
    for r in records:
        # Strip timestamp prefix (YYYY-MM-DD_HH-MM-SS__)
        label = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}__?", "", r.run_name)
        label = label or r.run_name[:20]
        short_names.append(label)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Best fitness")
    ax.set_title("Exploration CPG — best fitness per run\n(hatched = walled arena)")

    # Legend for fitness functions
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=colour_map[name], label=name) for name in fitness_names
    ]
    ax.legend(handles=legend_handles, title="Fitness", loc="upper left", fontsize=8)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exploration_cpg runs.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=_DEFAULT_RUNS_DIR,
        help=f"Path to runs directory (default: {_DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print table only, skip matplotlib plot.",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=0,
        help="Only include runs with budget >= N.",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    if not runs_dir.exists():
        console.print(f"[red]Runs directory not found:[/red] {runs_dir}")
        return

    # Load all runs
    records: list[RunRecord] = []
    for folder in sorted(runs_dir.iterdir()):
        if not folder.is_dir():
            continue
        rec = load_run(folder)
        if rec is None:
            continue
        if args.min_budget and rec.budget < args.min_budget:
            continue
        records.append(rec)

    if not records:
        console.print("[yellow]No completed runs found.[/yellow]")
        return

    # Sort by best_fitness descending (NaN last)
    records.sort(key=lambda r: r.best_fitness if not np.isnan(r.best_fitness) else -1, reverse=True)

    print_table(records)

    if not args.no_plot:
        plot_runs(records)


if __name__ == "__main__":
    main()
