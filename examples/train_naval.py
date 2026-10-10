"""Run the real naval quickstart from any working directory after installation."""

import sys
from pathlib import Path

from stl_tree.cli import main

if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs" / "naval_quickstart.json"
    raise SystemExit(main(["train", "--config", str(config), "--trust-pickle", *sys.argv[1:]]))
