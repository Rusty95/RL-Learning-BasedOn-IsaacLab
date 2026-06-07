"""List Gymnasium tasks registered by this learning project."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source" / "rl_lab_learning"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import rl_lab_learning.tasks  # noqa: F401, E402


def main() -> None:
    task_ids = sorted(task_id for task_id in gym.registry if task_id.startswith("RLLab-"))
    if not task_ids:
        print("No RLLab-* tasks are registered.")
        return

    for task_id in task_ids:
        print(task_id)


if __name__ == "__main__":
    main()
