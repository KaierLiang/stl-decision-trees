#!/usr/bin/env bash
# Run this on a remote lab after installing the dev extra.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ruff check .
python -m ruff format --check src tests examples
python -m pytest -q
python -m build
python -m stl_tree inspect data/naval_dataset.pkl --trust-pickle
