# Evolving Neural Network Controllers for Autonomous Robot Exploration

MSc thesis project — Vrije Universiteit Amsterdam, 2026.

This repository extends the [ARIEL](https://github.com/ci-group/ariel) framework to evolve a feedforward neural network controller for autonomous spatial exploration of a simulated gecko robot in MuJoCo.

## Method

A 408-parameter feedforward neural network (16→16→8, ELU/Tanh) is evolved using CMA-ES (population 22, 1000 generations) to control 8 hinge joints of a modular gecko robot. Two additional parameters (gait clock frequencies) are co-evolved, totalling 410 optimised dimensions.

**Inputs (16):** normalised position (2), normalised velocity (2), log1p-normalised neighbour visit counts (4), boundary edge flags (4), sinusoidal gait clocks (4).

**Outputs (8):** joint angles in [-π/2, π/2].

The arena is a flat 10×10 m surface discretised into a 10×10 grid. An outer ring of cells with pre-set visit counts acts as a soft boundary. Each candidate is evaluated across 4 randomised starting positions per generation.

Two fitness functions are compared at three simulation durations (300, 600, 1200 s):
- **f₁** (coverage fraction): |visited cells| / |total cells|
- **f_int** (coverage integral): (1/K) Σ c(tₖ), rewarding early cell discovery

## Repository Structure

```
src/ariel/                          ARIEL framework
  body_phenotypes/                  Modular robot construction
  simulation/environments/          MuJoCo world definitions
  simulation/controllers/           NaCPG, SimpleCPG controllers
  simulation/tasks/                 Fitness functions (f1–f24)
  utils/                            Simulation runners, state tracking

examples/thesisMTB/
  gecko_experiments/NNexp.py        Neural network experiment (main)
  gecko_experiments/exploration_cpg.py  CPG baseline experiment
  plot_results.py                   Thesis figure generation
  backfill_curves.py                Coverage curve recovery utility
```

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
git clone https://github.com/ci-group/ariel.git
cd ariel
git checkout thesis-mtb
uv sync
```

## Reproducing Experiments

### Evolution

```bash
uv run examples/thesisMTB/gecko_experiments/NNexp.py \
    --budget 1000 --dur 1200 --fitness integral --seed 42
```

Options: `--fitness {integral,fraction}`, `--dur {300,600,1200}`, `--seed <int>`, `--workers <int>`.

Results are saved to `__data__/NNexp/runs/<timestamp>/`.

### Replay

```bash
uv run examples/thesisMTB/gecko_experiments/NNexp.py \
    --replay __data__/NNexp/runs/<run_directory>
```

### Figure Generation

```bash
uv run examples/thesisMTB/plot_results.py --save
```

Expects data in `__data__/Thesis Runs/` organised by condition (e.g. `F1, 300/`, `F6, 1200/`).

## Data

Experiment data is not included in the repository. Each run produces best weights, fitness history, coverage curves, and configuration files. Approximate wall-clock times per run: 4 h (300 s), 36 h (600 s), 16 h (1200 s), depending on hardware and worker count.

## License

GPL-3.0. This project extends [ARIEL](https://github.com/ci-group/ariel), developed at the CI Group, Vrije Universiteit Amsterdam.
