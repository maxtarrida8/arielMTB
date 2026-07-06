"""
Generate thesis figures 4–7 from the Thesis Runs data.

Figure 4 (§5.2.1): Convergence plots — 2×3 grid (fitness × duration),
    best fitness per generation, mean ± std shaded band across seeds.
Figure 5 (§5.2.2): Box plots of final coverage fraction per condition.
Figure 6 (§5.2.2): Box plots of coverage integral per condition.
Figure 7 (§5.2.3): Coverage fraction vs simulation time for the best
    controller from each of the 6 conditions.

Usage
-----
    uv run examples/thesisMTB/plot_results.py
    uv run examples/thesisMTB/plot_results.py --save   # save PNGs instead of showing
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

_BASE = Path(__file__).resolve().parent.parent.parent / "__data__"
# The Google Drive folder has a trailing space in its name.
DATA_ROOT = _BASE / "Thesis Runs " if (_BASE / "Thesis Runs ").is_dir() else _BASE / "Thesis Runs"

CONDITIONS = {
    "F1": "Coverage fraction ($f_1$)",
    "F6": "Coverage integral ($f_{\\mathrm{int}}$)",
}
DURATIONS = [300, 600, 1200]
DURATION_LABELS = {300: "300 s", 600: "600 s", 1200: "1200 s"}

COLORS = {
    "F1": "#d62728",
    "F6": "#1f77b4",
}


def _load_runs(condition_dir: Path) -> list[dict]:
    """Load all zip runs from a condition directory.

    Returns a list of dicts with keys:
        seed, fitness_metric, best_coverage_integral, best_final_coverage,
        duration_s, fitness_history, mean_history, f_slow, f_fast
    """
    runs = []
    if not condition_dir.is_dir():
        return runs

    for zpath in sorted(condition_dir.glob("*.zip")):
        zf = zipfile.ZipFile(zpath)
        summary = config = None
        fitness_hist = mean_hist = coverage_curve = None

        for name in zf.namelist():
            if "__MACOSX" in name:
                continue
            if name.endswith("summary.json"):
                summary = json.loads(zf.read(name).decode("utf-8", errors="replace"))
            elif name.endswith("config.json"):
                config = json.loads(zf.read(name).decode("utf-8", errors="replace"))
            elif name.endswith("fitness_history.npy"):
                fitness_hist = np.load(io.BytesIO(zf.read(name)))
            elif name.endswith("mean_history.npy"):
                mean_hist = np.load(io.BytesIO(zf.read(name)))
            elif name.endswith("coverage_curve.npy"):
                coverage_curve = np.load(io.BytesIO(zf.read(name)))

        if summary is None or config is None or fitness_hist is None:
            continue

        seed = summary.get("seed", config.get("args", {}).get("seed", -1))

        best_integral = summary.get("best_coverage_integral")
        best_final = summary.get(
            "best_final_coverage",
            summary.get("best_avg_final_coverage",
                        summary.get("best_coverage")),
        )

        runs.append({
            "seed": seed,
            "fitness_metric": summary.get("fitness_metric", "unknown"),
            "best_coverage_integral": best_integral,
            "best_final_coverage": best_final,
            "duration_s": summary.get("duration_s"),
            "fitness_history": fitness_hist,
            "mean_history": mean_hist,
            "f_slow": summary.get("f_slow"),
            "f_fast": summary.get("f_fast"),
            "coverage_curve": coverage_curve,
            "zip_name": zpath.name,
        })

    return runs


def load_all_data() -> dict[str, dict[int, list[dict]]]:
    """Load all runs organised as {fitness_label: {duration: [runs]}}."""
    data: dict[str, dict[int, list[dict]]] = {}
    for fitness_key in CONDITIONS:
        data[fitness_key] = {}
        for dur in DURATIONS:
            cond_dir = DATA_ROOT / f"{fitness_key}, {dur}"
            data[fitness_key][dur] = _load_runs(cond_dir)
    return data


def plot_convergence(data: dict, save: bool, out_dir: Path) -> None:
    """Figure 4: 2×3 convergence grid."""
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 7), sharex=True, sharey="row",
    )

    for row, fitness_key in enumerate(["F1", "F6"]):
        for col, dur in enumerate(DURATIONS):
            ax = axes[row, col]
            runs = data[fitness_key][dur]

            if not runs:
                ax.text(
                    0.5, 0.5, "No data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="gray",
                )
                ax.set_xlim(0, 1000)
                if row == 1:
                    ax.set_xlabel("Generation")
                if col == 0:
                    ax.set_ylabel(CONDITIONS[fitness_key])
                if row == 0:
                    ax.set_title(DURATION_LABELS[dur])
                continue

            histories = np.array([r["fitness_history"] for r in runs])
            n_gen = histories.shape[1]
            gens = np.arange(1, n_gen + 1)

            if histories.shape[0] == 1:
                ax.plot(gens, histories[0], color=COLORS[fitness_key], linewidth=1)
            else:
                mean = histories.mean(axis=0)
                std = histories.std(axis=0)
                ax.plot(gens, mean, color=COLORS[fitness_key], linewidth=1.2)
                ax.fill_between(
                    gens, mean - std, mean + std,
                    color=COLORS[fitness_key], alpha=0.2,
                )

            if mean_hists := [r["mean_history"] for r in runs if r["mean_history"] is not None]:
                mean_arr = np.array(mean_hists)
                pop_mean = mean_arr.mean(axis=0)
                ax.plot(
                    gens, pop_mean,
                    color=COLORS[fitness_key], linewidth=0.7,
                    alpha=0.5, linestyle="--", label="Pop. mean",
                )

            ax.set_xlim(0, n_gen)
            ax.grid(True, alpha=0.3)

            n_seeds = len(runs)
            seeds_str = ", ".join(str(r["seed"]) for r in runs)
            ax.annotate(
                f"n={n_seeds} (seeds: {seeds_str})",
                xy=(0.98, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color="gray",
            )

            if row == 0:
                ax.set_title(DURATION_LABELS[dur], fontsize=12)
            if row == 1:
                ax.set_xlabel("Generation")
            if col == 0:
                ax.set_ylabel(f"Best fitness\n({CONDITIONS[fitness_key]})")

    fig.suptitle(
        "Convergence: best fitness per generation (mean ± std across seeds)",
        fontsize=13, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        path = out_dir / "fig4_convergence.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def plot_box(
    data: dict,
    metric_key: str,
    ylabel: str,
    title: str,
    fig_name: str,
    save: bool,
    out_dir: Path,
) -> None:
    """Grouped box plot: F1 vs F6 at each duration."""
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = []
    box_data = []
    tick_positions = []
    tick_labels = []
    colors = []

    group_width = 2.5
    bar_width = 0.8

    for i, dur in enumerate(DURATIONS):
        center = i * group_width
        for j, fitness_key in enumerate(["F1", "F6"]):
            pos = center + (j - 0.5) * bar_width
            runs = data[fitness_key][dur]
            values = [r[metric_key] for r in runs if r.get(metric_key) is not None]
            if not values:
                values = [float("nan")]
            positions.append(pos)
            box_data.append(values)
            colors.append(COLORS[fitness_key])

        tick_positions.append(center)
        tick_labels.append(DURATION_LABELS[dur])

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, (pos, vals) in enumerate(zip(positions, box_data)):
        if not any(np.isnan(v) for v in vals):
            for v in vals:
                ax.plot(pos, v, "o", color=colors[i], markersize=5, alpha=0.8, zorder=5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["F1"], alpha=0.6, label=CONDITIONS["F1"]),
        Patch(facecolor=COLORS["F6"], alpha=0.6, label=CONDITIONS["F6"]),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.tight_layout()

    if save:
        path = out_dir / f"{fig_name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def _interpolate_curves(
    runs: list[dict], dur: float, n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate all available coverage curves onto a common time grid.

    Returns (common_times, stacked_coverages) where stacked_coverages has
    shape (n_runs_with_curves, n_points) and values are in [0, 100] (percent).
    Returns None if no runs have a coverage curve.
    """
    curves = [r["coverage_curve"] for r in runs if r.get("coverage_curve") is not None]
    if not curves:
        return None
    common_t = np.linspace(0, dur, n_points)
    interpolated = []
    for curve in curves:
        times = curve[:, 0]
        coverages = curve[:, 1] * 100
        interp = np.interp(common_t, times, coverages)
        interpolated.append(interp)
    return common_t, np.array(interpolated)


def plot_coverage_over_time(data: dict, save: bool, out_dir: Path) -> None:
    """Figure 7: mean coverage-vs-time with std band across all seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    linestyles = {"F1": "-", "F6": "--"}

    for col, dur in enumerate(DURATIONS):
        ax = axes[col]

        for fitness_key in ["F1", "F6"]:
            runs = data[fitness_key][dur]
            result = _interpolate_curves(runs, dur)
            if result is None:
                continue

            common_t, stacked = result
            mean_cov = stacked.mean(axis=0)
            n_curves = stacked.shape[0]

            ax.plot(
                common_t, mean_cov,
                color=COLORS[fitness_key],
                linestyle=linestyles[fitness_key],
                linewidth=1.5,
                label=f"{CONDITIONS[fitness_key]} (n={n_curves})",
            )

            if n_curves > 1:
                std_cov = stacked.std(axis=0)
                ax.fill_between(
                    common_t, mean_cov - std_cov, mean_cov + std_cov,
                    color=COLORS[fitness_key], alpha=0.15,
                )

            for thresh in [50, 80]:
                hits = np.where(mean_cov >= thresh)[0]
                if len(hits) > 0:
                    t_hit = common_t[hits[0]]
                    ax.plot(t_hit, thresh, "o", color=COLORS[fitness_key],
                            markersize=6, zorder=5)

        for thresh in [50, 80]:
            ax.axhline(thresh, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

        ax.set_xlim(0, dur)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Time (s)")
        ax.set_title(DURATION_LABELS[dur], fontsize=12)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Coverage (%)")
        if col == 2:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Coverage over time (mean ± std across seeds)",
        fontsize=13, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        path = out_dir / "fig7_coverage_over_time.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def print_summary_table(data: dict) -> None:
    """Print a text summary of all runs for quick inspection."""
    print(f"\n{'Condition':<20} {'Seeds':>6} {'Cov. Integral':>16} {'Final Cov.':>14}")
    print("-" * 60)
    for fitness_key in ["F1", "F6"]:
        for dur in DURATIONS:
            runs = data[fitness_key][dur]
            label = f"{fitness_key}, {dur}s"
            if not runs:
                print(f"{label:<20} {'0':>6} {'—':>16} {'—':>14}")
                continue
            integrals = [r["best_coverage_integral"] for r in runs
                         if r.get("best_coverage_integral") is not None]
            finals = [r["best_final_coverage"] for r in runs
                      if r.get("best_final_coverage") is not None]
            n = len(runs)
            int_str = (f"{np.mean(integrals):.3f} ± {np.std(integrals, ddof=1):.3f}"
                       if integrals else "—")
            fin_str = (f"{np.mean(finals):.3f} ± {np.std(finals, ddof=1):.3f}"
                       if finals else "—")
            print(f"{label:<20} {n:>6} {int_str:>16} {fin_str:>14}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot thesis results (§5.2)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures as PNGs instead of showing")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for saved figures")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else DATA_ROOT / "figures"
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_data()
    print_summary_table(data)

    plot_convergence(data, save=args.save, out_dir=out_dir)

    plot_box(
        data,
        metric_key="best_final_coverage",
        ylabel="Final coverage fraction",
        title="Final Coverage Fraction by Fitness Function and Duration",
        fig_name="fig5_final_coverage",
        save=args.save,
        out_dir=out_dir,
    )

    plot_box(
        data,
        metric_key="best_coverage_integral",
        ylabel="Coverage integral (time-averaged)",
        title="Coverage Integral by Fitness Function and Duration",
        fig_name="fig6_coverage_integral",
        save=args.save,
        out_dir=out_dir,
    )

    plot_coverage_over_time(data, save=args.save, out_dir=out_dir)


if __name__ == "__main__":
    main()
