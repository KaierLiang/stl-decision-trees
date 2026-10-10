# Remote validation

All execution checks below ran on the remote `lab` machine on **2026-10-10**.
No local Python tests, local optimization, or GitHub-hosted tests were run.

These checks validate the refactoring and onboarding workflow. They do not
reproduce all paper experiments. The ECG data were unavailable, the original
zero-robustness ambiguity remains, and a finite-budget solve of the full naval
configuration is not a global-optimality certificate.

## Environment

| Component | Version / setting |
| --- | --- |
| OS | Pop!_OS 22.04 LTS, Linux x86-64 |
| Python | 3.13.9 |
| NumPy | 2.5.3 |
| Gurobi | 13.0.3, academic license |
| pytest | 9.1.1 |
| Ruff | 0.17.0 |
| Solver threads | 1 |
| Sampling and solver seed | 42 |

A project-specific virtual environment was created under
`~/research/stl-decision-trees-20261010`. Dependencies were installed with `uv pip`
after pip's remote network requests stalled. The original shared environments
were not modified. NumPy and Gurobi versions in each run are saved in its summary.

## Completed checks

| Check | Result |
| --- | --- |
| Ruff lint and formatting | Passed; 15 source/notebook files checked/formatted |
| pytest | **15 passed** in 4.06 seconds |
| Original `exist` vs refactored model | Equal optimal objective, variable count, linear-constraint count, and general-constraint count |
| Original `belong` vs refactored model | Same checks passed |
| Source augmentation and sampling | Exact array equality on real naval data |
| Temporal windows | All 1,891 intervals retained; exact equality with original generator |
| No-incumbent path | Infeasible model produces a summary and no tree |
| Early feasible stop | `SOLUTION_LIMIT` incumbent exported without an optimality claim |
| One continuous channel | Model constructs and solves without the original channel-index error |
| Serialization and equality policy | JSON tree round-trip and explicit left/right/error tie handling passed |
| CLI | Config overrides, paths independent of cwd, NPZ conversion, inspection, and training passed |
| Distribution build | Wheel and source distribution built successfully |
| Wheel installation | Wheel installed in a second environment; CLI inspection ran from `/tmp` |
| Notebook | All six code cells executed successfully; trajectory plot visually inspected |

Both source-model regressions use eight **actual naval trajectories**, sampled
from the original pooled splits, followed by the identical augmentation. They
use depth 1 and the full time horizon/window family. The original code itself is
evaluated through objective construction. Only the `exist` numeric min-argument
call is adapted to Gurobi's explicit `constant` argument. This is a bounded
regression check, not an exhaustive equivalence proof or paper-scale evaluation.

## CLI quickstart

```bash
stl-tree train --config configs/naval_quickstart.json --trust-pickle --output outputs/quickstart-validation
```

The 12 selected training trajectories become 13 after augmentation. The solver
proved objective **13**, bound **13**, gap **0**, in **0.94 seconds**. The extracted
tree's deterministic training score was **12/13**, with **two samples reaching a
threshold tie** under tolerance `1e-6`. That difference is evidence of the inherited
zero-margin behavior, not something the refactoring conceals or changes.

The notebook separately ran the same setup and evaluated independently augmented
test data: 0.684 accuracy on 500 augmented test trajectories, with no prediction
ties. This small training sample is an onboarding example, not a paper benchmark.

Machine-readable quickstart evidence is in [quickstart.json](validation/quickstart.json).

## Full source naval configuration, finite solver budget

```bash
stl-tree train --config configs/naval_legacy.json --trust-pickle --time-limit 120 --output outputs/naval-legacy-validation
```

This run preserves 100 sampled trajectories from the pooled splits, the original
augmentation (123 total trajectories), depth 3, `belong` mode, and all 1,891
windows. It uses 29,176 variables, 18,472 linear constraints, and 4,305 general
constraints. Model construction took 4.44 seconds.

**The run reached its time limit; the optimum was not proved.** It returned a
feasible objective of **105**, upper bound **123**, and relative MIP gap **17.14%**
after 120.04 seconds. The flow fraction is 105/123; deterministic prediction with
the documented tie policy is 103/123, with three samples reaching a tie. The
incumbent tree and summary were exported successfully.

This checks the maintained full-size configuration and timeout path. It is not
a held-out test, nor evidence of reproducing a published score. Machine-readable
evidence is in [naval-legacy-120s.json](validation/naval-legacy-120s.json).

## Re-run on a remote lab

After installing the development dependencies, run:

```bash
python -m ruff check .
python -m ruff format --check src tests examples
python -m pytest -q
python -m build --no-isolation
```

The environment needs setuptools and wheel for the no-isolation build. The general
`scripts/validate.sh` entry point uses the default isolated build instead.
Use a fresh directory for each training run. Raw logs and the executed notebook
are retained in the remote project directory; solver credentials and license
details are not included in committed validation evidence.
