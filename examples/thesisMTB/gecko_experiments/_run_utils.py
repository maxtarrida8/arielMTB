"""Tiny utilities shared by thesisMTB experiment scripts.

Each run gets its own folder under
    `__data__/<script_name>/runs/<YYYY-MM-DD_HH-MM-SS[__<run-name>]>/`
with `config.json` (all CLI args + git hash + start time) and
`summary.txt` (key results) sitting next to the run's outputs
(`best_params.npz`, `database.db`, etc.).

Designed to be imported with a `sys.path` insert in each script:

    sys.path.insert(0, str(Path(__file__).parent))
    from _run_utils import make_run_dir, save_run_config, save_run_summary
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def make_run_dir(script_path: Path, run_name: str | None = None) -> Path:
    """Create and return a fresh run folder for this script.

    Layout:
        ``Path.cwd() / "__data__" / <script_stem> / "runs" / <run_id>/``

    where ``<run_id>`` is the current local timestamp, optionally suffixed with
    ``__<run-name>``. Also tries to update a ``latest`` symlink one level up.
    """
    script_stem = Path(script_path).stem
    parent = Path.cwd() / "__data__" / script_stem / "runs"
    parent.mkdir(exist_ok=True, parents=True)

    run_id = _timestamp()
    if run_name:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
        run_id = f"{run_id}__{safe}"

    run_dir = parent / run_id
    run_dir.mkdir(exist_ok=True, parents=True)

    # Best-effort `latest` symlink (skip silently on Windows or restricted FS).
    latest = parent.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(Path("runs") / run_id, target_is_directory=True)
    except (OSError, NotImplementedError):
        pass

    return run_dir


def save_run_config(
    run_dir: Path,
    args: argparse.Namespace | dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> Path:
    """Write `config.json` capturing CLI args + run metadata."""
    if isinstance(args, argparse.Namespace):
        args_dict: dict[str, Any] = {k: _json_safe(v) for k, v in vars(args).items()}
    else:
        args_dict = {k: _json_safe(v) for k, v in dict(args).items()}

    payload: dict[str, Any] = {
        "script": sys.argv[0] if sys.argv else "",
        "command": " ".join(sys.argv),
        "args": args_dict,
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit_hash(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if extras:
        payload["extras"] = {k: _json_safe(v) for k, v in extras.items()}

    out = run_dir / "config.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return out


def save_run_summary(run_dir: Path, **fields: Any) -> Path:
    """Write a small human-readable `summary.txt` with the given fields."""
    out = run_dir / "summary.txt"
    lines: list[str] = ["# Run summary", ""]
    for key, value in fields.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    out.write_text("\n".join(lines))
    return out


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of arbitrary Python values to JSON-safe ones."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return repr(value)
