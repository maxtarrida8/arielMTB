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
- created f8, which rewards spreading by maximizing the convex hull of the robot's path. This was done in order to stop the circling behavior the previous fitness functions have resulted by making the spread out more, as this function is maximized if the robot visits the four corners of the arena. 
- added 
**Result:**
- best_fitness: 
- Behavior observed: 
- Run dir: 

**plan for tomorrow:**
- 

## 2026-06-03 ##

**What I did:**
-  
**Result:**
- best_fitness: 
- Behavior observed: 
- Run dir: 

**plan for tomorrow:**