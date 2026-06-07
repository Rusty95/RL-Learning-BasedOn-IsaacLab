"""Train a registered task with Isaac Lab's RSL-RL training script."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "rl_lab_learning"
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", PROJECT_ROOT.parent / "isaaclab")).resolve()
ISAACLAB_TRAIN = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"

if not ISAACLAB_TRAIN.exists():
    raise FileNotFoundError(
        f"Could not find Isaac Lab RSL-RL train.py at {ISAACLAB_TRAIN}. "
        "Set ISAACLAB_PATH to your Isaac Lab root."
    )

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import rl_lab_learning.tasks  # noqa: F401, E402

# Isaac Lab's delegated script imports its sibling cli_args.py by name.
if str(ISAACLAB_TRAIN.parent) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_TRAIN.parent))

runpy.run_path(str(ISAACLAB_TRAIN), run_name="__main__")
