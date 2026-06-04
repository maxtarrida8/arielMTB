"""MuJoCo world: flat walled arena with four waypoint target markers."""

# Standard library
from dataclasses import dataclass, field

# Third-party libraries
import mujoco

# Local libraries
from ariel.parameters.ariel_types import Dimension
from ariel.simulation.environments._base_world import BaseWorld

DEFAULT_TARGETS: list[tuple[float, float]] = [
    (3.0, 3.0),
    (3.0, -3.0),
    (-3.0, -3.0),
    (-3.0, 3.0),
]

TARGET_COLORS: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.6],  # red
    [0.0, 1.0, 0.0, 0.6],  # green
    [0.0, 0.0, 1.0, 0.6],  # blue
    [1.0, 1.0, 0.0, 0.6],  # yellow
]


@dataclass
class SimpleFlatWorldWalledWithTargets(BaseWorld):
    """A flat walled arena with four waypoint target markers.

    Combines ``SimpleFlatWorldWalled`` (floor + boundary walls) with four
    small cube markers placed at configurable XY positions.  The markers
    are purely visual (no collision effect) — they help interpret replay
    trajectories but do not influence the physics.

    Parameters
    ----------
    floor_size : Dimension
        (width, height, depth) of the floor plane in metres.
    wall_height : float
        Full height of each boundary wall in metres.
    wall_thickness : float
        Full thickness of each boundary wall in metres.
    checker_floor : bool
        Whether to apply the chequerboard texture to the floor.
    targets_xy : list[tuple[float, float]]
        XY positions of the waypoint markers.
    target_size : float
        Half-size of each cube marker in metres.
    load_precompiled : bool
        Whether to load a precompiled XML file if one exists.
    """

    name: str = "simple-flat-world-walled-with-targets"

    floor_size: Dimension = (10, 10, 1)
    wall_height: float = 1.0
    wall_thickness: float = 0.1
    checker_floor: bool = True

    targets_xy: list[tuple[float, float]] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    target_size: float = 0.15

    load_precompiled: bool = True

    def __post_init__(self) -> None:
        super().__init__(name=self.name, load_precompiled=self.load_precompiled)

        if self.is_precompiled:
            return

        self._floor_name = self.mujoco_config.floor_name
        width, height, depth = self.floor_size
        self._floor_kwargs = {
            "name": self._floor_name,
            "size": [width / 2, height / 2, depth / 2],
            "type": mujoco.mjtGeom.mjGEOM_PLANE,
        }

        self._create_checker_texture()
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
        hw = width / 2
        hh = height / 2
        wh = self.wall_height / 2
        wt = self.wall_thickness / 2

        wall_specs = [
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
                rgba=[0.6, 0.6, 0.6, 1.0],
            )

        # ---- Target markers ----
        s = self.target_size
        for i, (tx, ty) in enumerate(self.targets_xy):
            color = TARGET_COLORS[i % len(TARGET_COLORS)]
            body = self.spec.worldbody.add_body(
                name=f"target_{i}",
                pos=[tx, ty, s],
            )
            body.add_geom(
                name=f"target_{i}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[s, s, s],
                rgba=color,
                contype=0,
                conaffinity=0,
            )
