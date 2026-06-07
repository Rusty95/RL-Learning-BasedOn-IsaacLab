"""Play a trained checkpoint with Isaac Lab's RSL-RL play script."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "rl_lab_learning"
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", PROJECT_ROOT.parent / "isaaclab")).resolve()
ISAACLAB_PLAY = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"

if not ISAACLAB_PLAY.exists():
    raise FileNotFoundError(
        f"Could not find Isaac Lab RSL-RL play.py at {ISAACLAB_PLAY}. "
        "Set ISAACLAB_PATH to your Isaac Lab root."
    )

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import rl_lab_learning.tasks  # noqa: F401, E402

# Isaac Lab's delegated script imports its sibling cli_args.py by name.
if str(ISAACLAB_PLAY.parent) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_PLAY.parent))

runpy.run_path(str(ISAACLAB_PLAY), run_name="__main__")
