"""Re-evaluate best weights for runs missing coverage_curve.npy and patch the zips.

Usage
-----
    uv run examples/thesisMTB/backfill_curves.py
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "gecko_experiments"))
from NNexp import (
    ExplorationNetwork,
    _build_world,
    count_parameters,
    fill_parameters,
    run_exploration,
)

DATA_ROOT_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent / "__data__" / "Thesis Runs ",
    Path(__file__).resolve().parent.parent.parent / "__data__" / "Thesis Runs",
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if p.is_dir()), None)


def backfill_zip(zpath: Path) -> None:
    zf = zipfile.ZipFile(zpath, "r")
    names = zf.namelist()

    if any("coverage_curve.npy" in n for n in names):
        print(f"  SKIP (already has curve): {zpath.name}")
        return

    weights_name = next((n for n in names if n.endswith("best_weights.npy")), None)
    config_name = next((n for n in names if n.endswith("config.json")), None)
    if weights_name is None or config_name is None:
        print(f"  SKIP (missing weights or config): {zpath.name}")
        return

    weights = np.load(io.BytesIO(zf.read(weights_name)))
    config = json.loads(zf.read(config_name).decode())
    duration = config.get("args", {}).get("dur", config.get("duration_s", 300))

    net = ExplorationNetwork()
    n_nn = count_parameters(net)
    nn_w = weights[:n_nn]
    f_slow = float(np.clip(np.abs(weights[n_nn]), 0.1, 5.0))
    f_fast = float(np.clip(np.abs(weights[n_nn + 1]), 0.1, 10.0))

    fill_parameters(net, nn_w)
    model, data, core_id = _build_world(spawn_xy=(0.0, 0.0))
    _, _, _, curve, diverged = run_exploration(
        model, data, core_id, net, f_slow, f_fast, duration,
    )
    print(f"  Evaluated {zpath.name}: final_cov={curve[-1][1]:.4f}, diverged={diverged}")

    curve_arr = np.array(curve)
    prefix = weights_name.rsplit("/", 1)[0] if "/" in weights_name else ""
    curve_entry = f"{prefix}/coverage_curve.npy" if prefix else "coverage_curve.npy"

    buf = io.BytesIO()
    np.save(buf, curve_arr)
    curve_bytes = buf.getvalue()

    tmp = zpath.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zf.infolist():
            zout.writestr(item, zf.read(item.filename))
        zout.writestr(curve_entry, curve_bytes)
    zf.close()

    shutil.move(tmp, zpath)
    print(f"  Patched: {zpath.name}")


def main() -> None:
    if DATA_ROOT is None:
        print("Could not find 'Thesis Runs' directory")
        return

    for cond_dir in sorted(DATA_ROOT.iterdir()):
        if not cond_dir.is_dir():
            continue
        for zpath in sorted(cond_dir.glob("*.zip")):
            backfill_zip(zpath)

    print("\nDone.")


if __name__ == "__main__":
    main()
