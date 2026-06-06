## 2026-05-10 ##

**What I did:**
- Ran exploration_cpg with f3, budget=2500, workers=100, seed=3
- Added 10x10 coarser grid to reduce circle-favouring artefacts

**Result:**
- best_fitness: 0.000833
- Behavior observed: robot walks straigth down until leaving the terrain grid (see picture below and in runs folder). Working on workarounds for this, 
![alt text](image-c0a08362-4ecf-48de-8ac6-2c1307eada1f.png)
- Run dir: __data__/exploration_cpg/runs/2026-05-08_HHMMSS__f3_b300_dur240_seed3

**plan for tomorrow:**
- ...

## 2026-05-27 ##

**What I did:**
- Ran exploration_cpg with f3, budget=300, workers=100, seed=3
- Ran exploration_cpg with f2, budget=2000, workers=100, seed=39, lambda=0.9
- added walls to simple_flat_terrain, to prevent robot from escaping.
- built compare_runs.py 

**Result:**
- best_fitness: 
- Behavior observed: 
- Run dir: 

**plan for tomorrow:**
- combining f1 and f3 into a single function to promote coverage while keeping efficiency. Try different optimizers?

**plan for tomorrow:**
- ...

## 2026-05-28 ##

**What I did:**
- created f7 = f1 + f3, an ran it at duration 600 budget 300. 

**Result:**
- best_fitness: 
- Behavior observed: 
- Run dir: 

**plan for tomorrow:**
- 

## 2026-06-03 ##

**What I did:**
-  - created f8, which rewards spreading by maximizing the convex hull of the robot's path. This was done in order to stop the circling behavior the previous fitness functions have resulted by making the spread out more, as this function is maximized if the robot visits the four corners of the arena. 
- created SimpleFlatWorldWithTargets world. This was done to add 4 evenly spread out targets to the arena that the robot should visit in order to get rid of the looping in a circle behavior. 
- added combinations between the new fitness functions (f8 and f9) and the existing ones (f1, f3, f4, f6,f7). 
- ran f6 at 300 budget and 240 duration.

**Result:**
- best_fitness: 0.0452 (f6)
- Behavior observed: robot walked straight to the wall and to a corner
- Run dir: 

**plan for tomorrow:**
- testing all the new fitness function created today. If behavior doesn't change consider changing controller. Apply feedback from supervisor too.

---

## 2026-06-05 — Major session: entry-based counting, fitness fixes, architecture decisions ##

### Core bug fixes

**1. Entry-based cell visit counting (`visit_counts_from_xy`)**
- **Problem:** N[r,c] was being incremented every trajectory sample, so dwell time inflated visit counts. A robot sitting still in one cell for 30s looked like it had visited it 1500 times. All redundancy-based metrics were meaningless.
- **Fix:** N[r,c] is now only incremented when the robot *transitions into* a cell from a different cell. Staying still = 1 entry. Leaving and coming back = 2 entries.
- **Impact:** `sum(N)` now means "total cell transitions" — a true measure of movement. `redundancy_ratio(N)` is now semantically correct.
- **File:** `src/ariel/simulation/tasks/exploration.py` → `visit_counts_from_xy()`

**2. f3 restored to original definition**
- **Problem:** f3 had been changed to `unique_cells / path_length_m` which is unbounded and exploitable (robot scored f3=54 by making tiny movements, keeping path_length near 0).
- **Fix:** f3 = `|V_T| / sum(N)` = `path_efficiency(N)` — unique cells divided by total cell entries.
- **Semantics with entry-based N:** perfect robot (never re-enters) scores 1.0. Circling robot scores <1.0. Staying still scores 1/1=1.0 (degenerate exploit, but f2 fixes this).
- **File:** `src/ariel/simulation/tasks/exploration.py` → `fitness_f3_unique_cells_per_step(N)`

**3. f2 deceptive landscape problem**
- **Problem:** f2 = `cov - λ*R` with λ=0.6–0.9 creates a deceptive local optimum. Staying still (1 cell, R=0) scores f2=0.01. Any random CPG movement causes re-entries and scores f2<0.01. Optimizer always converges to "don't move."
- **Root cause:** Random CPG parameters produce oscillatory gaits that cross cell boundaries — penalised even though they're not "bad" exploration, just gait mechanics.
- **Proposed fix (NOT yet implemented):** Buffered redundancy — only count a re-entry if the re-entered cell is NOT one of the last K recently visited cells (e.g. K=4). Gait oscillation between 2–4 adjacent cells is free; large-scale loops (revisiting cells from 30+ seconds ago) are still penalised.

**4. Spawn position fix**
- **Problem:** Robot spawned at (0,0) = exact intersection of 4 cells. Normal gait oscillation caused immediate spurious re-entries, inflating R(T) and making f2 unfairly penalise good gaits.
- **Fix:** All experiment scripts now spawn at (0.5, 0.5) = center of cell (5,5) on a 10×10 grid. Half-metre clearance to any boundary in every direction.
- **Caveat:** For f10 (no redundancy penalty), (0.0, 0.0) is actually better — boundary-crossing signal helps optimizer find moving gaits faster. Cell-center spawn mainly matters for f2.
- **Files changed:** `exploration_cpg.py`, `exploration_simple_cpg.py`, `pipeline_check.py`

### Grid resolution decision

**Single source of truth: `ARENA_GRID` in `_fitness_registry.py`**
- All scripts import `ARENA_GRID` from `_fitness_registry.py`. Change it in one place to resize cells everywhere.
- **Current setting:** 10×10 (1m × 1m cells). Correct because:
  - Signal strength: each new cell adds 1/100 = 0.01 to coverage (optimizer can feel it)
  - Noise immunity: 1m cell is 4× gecko body size, gait oscillation stays within a cell
  - 30×30 caused stagnation at f=0.0011 — signal too weak for the optimizer
- **Formula:** cell size = `width_m / ncol` × `height_m / nrow`

### Fitness function analysis

| Fitness | Verdict | Notes |
|---|---|---|
| f1 | Baseline, no anti-circling | Use for warm-start or comparison only |
| f2 (λ=0.5–0.6) | Strong candidate BUT deceptive local optima | Need buffered redundancy fix first |
| f3 | Weak alone | Exploitable (stay still = 1.0), use combined |
| f8 (hull) | Good anti-circling signal | Too weak alone |
| **f10 = 0.5*f1 + 0.5*f8** | **Best current option** | No deceptive local optima, rewards coverage + spatial spread |
| f14 = f1 * f8 | Interesting | Multiplicative gate, requires both coverage AND spread |

**f10 stagnation issue:** f10 stagnated at 0.005 with budget=200, duration=500, spawn=(0.5,0.5). Cause: spawn at cell center means robot needs 0.5m movement before any fitness change — flat landscape in early gens. Fix: use (0.0, 0.0) for f10 specifically.

### Controller architecture — critical insight

**All CPGs are open-loop (fixed heading):**
NaCPG, SimpleCPG, NaCPGBeta are all open-loop. Parameters set once, gait is fixed for the entire run. The robot walks at a constant heading angle (or constant turning rate = circular arc). The ONLY way it changes direction is hitting a wall.

- Best achievable trajectory = straight line + wall bounces (billiard ball)
- A spiral is NOT achievable with a fixed CPG (spiral needs decreasing turning rate over time)
- The optimizer finds the gait with the right speed and turning radius for good wall bounces

**Neural network controller exists in this repo:**
`examples/re_book/1_brain_evolution.py` — `Network` class (2-layer FC, ELU activation). Reads robot state + vision + phase signals every control step. Outputs motor commands directly. This is closed-loop and CAN change direction mid-run.

To adapt for gecko exploration:
- Remove vision/camera input
- Feed (x, y, heading) so the network knows where it is
- Keep sin/cos phase signals for rhythmic output
- Evolve with CMA-ES or PSO
- Estimated effort: 1–2 days
- Tradeoff: much larger parameter space (~hundreds vs 40 for CPG)

### Open TODO for next session

1. **Implement buffered redundancy for f2** — only penalise re-entry if cell NOT in last K visited (K≈4)
2. **Try f10 with spawn (0.0, 0.0)** — should fix the flat-landscape stagnation
3. **Decide: adapt NN controller or stick with CPG + walls**
4. **Run proper f10 experiment** — budget ≥500, duration ≥300, spawn (0.0,0.0)
5. **Compare f10 vs f2-buffered** — main experiment comparison for thesis

### Key files changed this session
- `src/ariel/simulation/tasks/exploration.py` — entry-based counting, f3 restored
- `examples/thesisMTB/gecko_experiments/_fitness_registry.py` — ARENA_GRID single source of truth, locomotion imports removed
- `examples/thesisMTB/gecko_experiments/exploration_cpg.py` — spawn (0.5,0.5), GRID=ARENA_GRID
- `examples/thesisMTB/gecko_experiments/exploration_simple_cpg.py` — spawn (0.5,0.5), GRID=ARENA_GRID
- `examples/thesisMTB/gecko_experiments/pipeline_check.py` — spawn (0.5,0.5), f2 added, cell-center waypoints
- `examples/thesisMTB/compare_runs.py` — imports ARENA_GRID