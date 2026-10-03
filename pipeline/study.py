"""The task settings and the active model's output paths."""
from __future__ import annotations

from pathlib import Path

from pipeline import config as C

MUTANTS_PER_SOLUTION = 24
GENERATED_TESTS_PER_SOLUTION = 10
FAULTS_PER_SOLUTION = 2


def path(task: str, filename: str) -> Path:
    """`data/<task>_<model>/<filename>` for the active model."""
    d = C.DATA / f"{task}_{C.ACTIVE_PROFILE}"
    d.mkdir(parents=True, exist_ok=True)
    return d / filename
