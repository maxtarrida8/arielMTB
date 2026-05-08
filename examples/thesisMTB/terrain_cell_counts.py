"""Report discrete terrain cell counts for ARIEL environments.

This is intended for exploration-style fitness functions where you track
"visited cells" on a discretized terrain grid. For heightfield terrains we can
infer a natural grid resolution from the underlying heightmap.

Notes
-----
- Heightfield-based terrains (CompoundWorld subclasses) expose `dims=(nrow,ncol)`
  and `floor_size=(width,height,depth)` in meters.
- `OlympicArena` uses a heightfield only for its rugged section; the resolution
  is `rugged_resolution` and the physical size is derived from
  `section_length` and `arena_width`.
- Plane terrains (e.g. SimpleFlatWorld) do not define an intrinsic grid; for
  those we report `cells=None` (you must pick your own discretization).
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import ariel.simulation.environments as envs


@dataclass(frozen=True)
class TerrainCellInfo:
    env_class: str
    kind: str  # "heightfield" | "heightfield_section" | "plane" | "unknown"
    nrow: int | None
    ncol: int | None
    cells: int | None
    cell_size_x_m: float | None
    cell_size_y_m: float | None
    notes: str | None = None


DEFAULT_PLANE_CELL_SIZE_M = 0.10


def _compoundworld_cell_info(cls: type[Any]) -> TerrainCellInfo | None:
    """Extract cell info for CompoundWorld-like terrains if possible."""
    dims = getattr(cls, "dims", None)
    floor_size = getattr(cls, "floor_size", None)
    if not isinstance(dims, tuple) or len(dims) != 2:
        return None
    if not isinstance(floor_size, tuple) or len(floor_size) < 2:
        return None

    nrow, ncol = int(dims[0]), int(dims[1])
    width_m, height_m = float(floor_size[0]), float(floor_size[1])
    cells = nrow * ncol
    cell_size_x_m = width_m / max(ncol, 1)
    cell_size_y_m = height_m / max(nrow, 1)

    return TerrainCellInfo(
        env_class=cls.__name__,
        kind="heightfield",
        nrow=nrow,
        ncol=ncol,
        cells=cells,
        cell_size_x_m=cell_size_x_m,
        cell_size_y_m=cell_size_y_m,
    )


def _plane_world_cell_info(
    cls: type[Any], *, cell_size_m: float = DEFAULT_PLANE_CELL_SIZE_M
) -> TerrainCellInfo | None:
    """Discretize a plane world using its `floor_size` and a chosen cell size."""
    floor_size = getattr(cls, "floor_size", None)
    if not isinstance(floor_size, tuple) or len(floor_size) < 2:
        return None

    width_m, height_m = float(floor_size[0]), float(floor_size[1])
    safe = max(float(cell_size_m), 1e-9)
    ncol = max(1, int(math.ceil(width_m / safe)))
    nrow = max(1, int(math.ceil(height_m / safe)))
    cells = nrow * ncol

    return TerrainCellInfo(
        env_class=cls.__name__,
        kind="plane",
        nrow=nrow,
        ncol=ncol,
        cells=cells,
        cell_size_x_m=safe,
        cell_size_y_m=safe,
        notes=f"Plane discretized with cell_size={safe}m using floor_size=({width_m}m,{height_m}m).",
    )


def _olympic_arena_cell_info(cls: type[Any]) -> TerrainCellInfo | None:
    """Extract the rugged-section heightfield discretization for OlympicArena."""
    if cls.__name__ != "OlympicArena":
        return None

    sig = inspect.signature(cls.__init__)
    defaults: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.default is not inspect.Parameter.empty:
            defaults[name] = p.default

    rugged_resolution = int(defaults.get("rugged_resolution", 0) or 0)
    arena_width = float(defaults.get("arena_width", 0.0) or 0.0)
    section_length = float(defaults.get("section_length", 0.0) or 0.0)

    if rugged_resolution <= 0 or arena_width <= 0.0 or section_length <= 0.0:
        return TerrainCellInfo(
            env_class=cls.__name__,
            kind="unknown",
            nrow=None,
            ncol=None,
            cells=None,
            cell_size_x_m=None,
            cell_size_y_m=None,
            notes="Could not infer OlympicArena defaults from signature.",
        )

    nrow = ncol = rugged_resolution
    cells = nrow * ncol

    # In `olympic_arena.py`, the heightfield is created with:
    # size=[section_length, arena_width/2, ...]
    # MuJoCo hfield size is given as half-extents in x and y.
    rugged_section_length_m = 2.0 * section_length
    rugged_section_width_m = arena_width
    cell_size_x_m = rugged_section_length_m / max(ncol, 1)
    cell_size_y_m = rugged_section_width_m / max(nrow, 1)

    return TerrainCellInfo(
        
        env_class=cls.__name__,
        kind="heightfield_section",
        nrow=nrow,
        ncol=ncol,
        cells=cells,
        cell_size_x_m=cell_size_x_m,
        cell_size_y_m=cell_size_y_m,
        notes="Counts only the rugged heightfield section (flat/incline are not heightfields).",
    )


def terrain_cell_infos() -> list[TerrainCellInfo]:
    infos: list[TerrainCellInfo] = []

    for name, cls in inspect.getmembers(envs, inspect.isclass):
        # Only consider environments this module publicly exports
        if name not in getattr(envs, "__all__", []):
            continue
        # Skip base classes; we only want concrete terrains.
        if name in {"BaseWorld", "CompoundWorld"}:
            continue

        # Special-case OlympicArena (hybrid)
        if name == "OlympicArena":
            infos.append(_olympic_arena_cell_info(cls) or TerrainCellInfo(
                env_class=name,
                kind="unknown",
                nrow=None,
                ncol=None,
                cells=None,
                cell_size_x_m=None,
                cell_size_y_m=None,
            ))
            continue

        # Heightfield terrains (CompoundWorld-like)
        cw = _compoundworld_cell_info(cls)
        if cw is not None:
            infos.append(cw)
            continue

        # Plane-like terrains (SimpleFlatWorld / SimpleTiltedWorld)
        pw = _plane_world_cell_info(cls)
        infos.append(
            pw
            or TerrainCellInfo(
                env_class=name,
                kind="unknown",
                nrow=None,
                ncol=None,
                cells=None,
                cell_size_x_m=None,
                cell_size_y_m=None, 
            )
        )

    # Stable output
    infos.sort(key=lambda x: x.env_class)
    return infos


def main() -> None:
    infos = terrain_cell_infos()
    for info in infos:
        print(json.dumps(asdict(info), sort_keys=True))


if __name__ == "__main__":
    main()

