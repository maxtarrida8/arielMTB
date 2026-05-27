"""MuJoCo world: flat world with a chequerboard floor and boundary walls."""

# Standard library
from dataclasses import dataclass, field

# Third-party libraries
import mujoco

# Local libraries
from ariel.parameters.ariel_types import Dimension
from ariel.simulation.environments._base_world import BaseWorld


@dataclass
class SimpleFlatWorldWalled(BaseWorld):
    """A flat world with a chequerboard floor enclosed by four boundary walls.

    The walls sit flush with the floor edges and are tall enough to prevent
    the robot from leaving the arena.  They are thin, collidable box geoms
    with a neutral grey material so they are visually unobtrusive.

    Parameters
    ----------
    floor_size : Dimension
        (width, height, depth) of the floor plane in metres.  Walls are
        placed at ±width/2 and ±height/2 on the X and Y axes respectively.
    wall_height : float
        Full height of each wall in metres (default 1.0 m).  Should be
        tall enough that the robot cannot climb over.
    wall_thickness : float
        Full thickness of each wall in metres (default 0.1 m).
    checker_floor : bool
        Whether to apply the chequerboard texture to the floor.
    load_precompiled : bool
        Whether to load a precompiled XML file if one exists.
    """

    name: str = "simple-flat-world-walled"

    floor_size: Dimension = (10, 10, 1)  # metres (width, height, depth)
    wall_height: float = 1.0            # metres (full height)
    wall_thickness: float = 0.1         # metres (full thickness)
    checker_floor: bool = True

    # Whether to load precompiled XML (if it exists)
    load_precompiled: bool = True

    def __post_init__(self) -> None:
        # Initialise base class
        super().__init__(name=self.name, load_precompiled=self.load_precompiled)

        # If precompiled XML was loaded, skip regeneration
        if self.is_precompiled:
            return

        # Floor geom parameters
        self._floor_name = self.mujoco_config.floor_name
        width, height, depth = self.floor_size
        self._floor_kwargs = {
            "name": self._floor_name,
            "size": [width / 2, height / 2, depth / 2],  # MuJoCo: half-sizes from centre
            "type": mujoco.mjtGeom.mjGEOM_PLANE,
        }

        # Create checker texture if enabled
        self._create_checker_texture()

        # Expand the MuJoCo specification (floor + walls)
        self._expand_spec()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_checker_texture(self) -> None:
        self.spec.add_texture(
            name=self._floor_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.1, 0.2, 0.3],
            rgb2=[0.2, 0.3, 0.4],
            width=600,
            height=600,
        )
        self.spec.add_material(
            name=self._floor_name,
            textures=["", f"{self._floor_name}"],
            texrepeat=[3, 3],
            texuniform=True,
            reflectance=0,
        )
        self._floor_kwargs["material"] = self._floor_name

    def _expand_spec(self) -> None:
        # ---- Floor ----
        floor = self.spec.worldbody.add_body(
            name=self._floor_name,
            pos=[0, 0, 0],
        )
        floor.add_geom(**self._floor_kwargs)

        # ---- Walls ----
        width, height, _ = self.floor_size
        hw = width / 2          # floor half-width  (X axis)
        hh = height / 2         # floor half-height (Y axis)
        wh = self.wall_height / 2        # wall half-height (Z)
        wt = self.wall_thickness / 2     # wall half-thickness

        # East and West walls run along the full Y span (including corners).
        # North and South walls run between the East/West walls.
        wall_specs = [
            # (name,   pos_x,      pos_y,  pos_z, sx,       sy,       sz)
            ("wall_east",   hw + wt,   0.0,   wh,   wt,  hh + wt,   wh),
            ("wall_west",  -hw - wt,   0.0,   wh,   wt,  hh + wt,   wh),
            ("wall_north",  0.0,   hh + wt,   wh,   hw,       wt,    wh),
            ("wall_south",  0.0,  -hh - wt,   wh,   hw,       wt,    wh),
        ]

        for w_name, px, py, pz, sx, sy, sz in wall_specs:
            wall_body = self.spec.worldbody.add_body(
                name=w_name,
                pos=[px, py, pz],
            )
            wall_body.add_geom(
                name=f"{w_name}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx, sy, sz],
                rgba=[0.6, 0.6, 0.6, 1.0],  # neutral grey, fully opaque
            )
