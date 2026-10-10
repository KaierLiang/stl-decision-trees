"""Load pinned original source from Git history for regression comparisons only."""

import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "bdc18ab1dfae70af521d9791e5ce3f13a788ab91"
FILES = {"main_belong.py", "main_belong_keep_temporal_feature.py", "utility.py"}


@lru_cache(maxsize=3)
def read_source(filename):
    if filename not in FILES:
        raise ValueError(f"Unknown reference source: {filename}")
    try:
        result = subprocess.run(
            ["git", "show", f"{REVISION}:{filename}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Source-comparison checks require a Git clone with the original commit. "
            "Use a full clone, or fetch missing history with git fetch --unshallow."
        ) from exc
    return result.stdout.decode("utf-8")


def load_utilities():
    namespace = {"__name__": "_reference_utilities"}
    exec(compile(read_source("utility.py"), f"{REVISION}:utility.py", "exec"), namespace)
    return namespace
