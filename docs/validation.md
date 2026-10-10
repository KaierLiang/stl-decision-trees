# Remote validation

All execution checks below ran on the remote `lab` machine on **2026-10-10**.
No local Python tests, local optimization, or GitHub-hosted tests were run.

**Scope:** naval only, under the author's stated noisy-data/nonzero-robustness
assumption. ECG is not a required experiment. The original training formulation,
data, hypothesis space, and all temporal windows are preserved.

**Current conclusion:** complete solves for both naval variants and both original
source models prove the same optimum. Numerically stable threshold export makes
deterministic predictions agree with all 123 admitted sample flows. The exported
assignments satisfy every original constraint and retain the optimal objective.
This validates the recorded naval runs; it is not a universal correctness proof
or a claim about held-out accuracy or reproduction of every paper experiment.

## Nonzero-robustness verification

The initial raw export's 121/123 score was investigated after the author clarified
the assumption. Its three near-zero routed decisions had robustness around
`1.02e-12` or `1.95e-12`, rather than exact floating-point zero. Prediction with
zero tolerance still gave 121/123: the solver's feasible numerical solution had
routed these small positive values right. This alone did **not** establish an
unavoidable gap between the optimal flow and achievable strict classification.

All relevant nodes had a nonempty strict threshold interval preserving the
incumbent's routing. Export now selects an interior threshold only at such
numerical boundaries, retaining the original variable/robustness bounds and
recording both raw and exported values. It introduces no new margin constraint.

| Complete model | Proven optimum / bound | Exported correct predictions | Prediction ties |
| --- | ---: | ---: | ---: |
| Refactored `exist` | 123 / 123 | 123/123 | 0 |
| Original `exist`, same export processing | 123 / 123 | 123/123 | 0 |
| Refactored `belong` | 123 / 123 | 123/123 | 0 |
| Original `belong`, same export processing | 123 / 123 | 123/123 | 0 |

The same complete instances below were rerun with no time limit and zero MIP gap,
using the earlier incumbent as a MIP start without fixing any model variables.
Every variable, linear row, MIN constraint, and objective was compared against
the original source again. Independent audits reconstructed the exported
thresholds' auxiliary variables and checked all original constraints, bounds,
integrality, objectives, and sample routes. Raw solver trees remain available;
the original model/solver assignment is not overwritten by export.
Across all four exports, minimum absolute routed continuous robustness is
`0.046817464383224205`; the largest linear constraint violation is approximately
`1.78e-15`. Every admitted sample follows the same branch as its incumbent flow.

Evidence: [complete nonzero naval audit](validation/nonzero-naval-audit.json),
[raw boundary diagnostic](validation/naval-boundary-audit.json), and portable
[`exist` tree](validation/naval-exist-tree.json) /
[`belong` tree](validation/naval-belong-tree.json).

After this change, **26 tests passed** on the remote lab, including both
operations/variants, preservation of bounds for unselected features, full
constraint audits, raw-tree export, and refusal to certify a collapsed strict
interval. The equality counterexample remains a passing boundary-detection test
outside the nonzero-robustness assumption, not a claimed supported data case.

The rebuilt wheel and source distribution also pass packaging checks, and all
six notebook code cells execute. **The small depth-1 quickstart is not a strict
routing validation:** its optimum uses an equality boundary with no open interval
for the same flows. A depth-2 diagnostic also retains an unresolved boundary.
Both are reported as unverified by the new check; neither substitutes for the
complete depth-3 naval results above. Threshold stabilization does not search
for a different tree or silently change the training formulation when an
incumbent violates the nonzero-robustness assumption.

## Initial raw export audit (historical)

The user requested complete runs instead of stopping at a solver time limit.
The follow-up retains **100 original samples, 123 after augmentation, depth 3,
both continuous features, the binary channel, all 61 time steps, and all 1,891
windows**. No time limit, positive gap tolerance, window pruning, or reduced tree
family was used for the completed runs.

| Full model | Flow objective | Upper bound | Gap | Fixed-policy predictions |
| --- | ---: | ---: | ---: | ---: |
| Refactored `exist` | 123 | 123 | 0 | 121/123 |
| Original `exist` | 123 | 123 | 0 | 121/123 |
| Refactored `belong` | 123 | 123 | 0 | 121/123 |
| Original `belong` | 123 | 123 | 0 | 121/123 |

The first `exist` solve started without a supplied incumbent and took 304.45
seconds with four threads. Its learned tree was then used as a **MIP start**, not
as fixed constraints, in the two full audit solves. On these constant binary
channels, a global `exist` node can seed `belong` with interval `[0,0]`. Every
interval remains available to the optimizer. Each original source model received
the corresponding refactored solution as a full MIP start and independently
proved the same bound. The duplicate `belong` solve without a supplied incumbent
was stopped after the complete warm-started proof was available; it had reached
122/123. Warm-started run times are recorded as such and are not performance
benchmarks against the initial solve.

The comparison ignores names and construction order and compares all variable
types, bounds, objective coefficients, linear row senses/RHS/coefficient multisets,
MIN operands/constants, objective sense, and objective constant. This covers
**6,571,444 nonzero coefficients for `exist` and 7,182,075 for `belong`**. Each
complete solution was then independently checked against every constraint and
its sample flow paths. These historical checks enforce the inherited equations,
which allow equality. Strict routing is verified separately in the newer audit
above under the author's clarified assumption.

Evidence:

- [Complete solves and constraint audits](validation/complete-naval-audit.json)
- [Initial complete `exist` solve without a supplied incumbent](validation/full-exist-cold.json)
- [Full `exist` coefficient comparison](validation/full-model-exist.json)
- [Full `belong` coefficient comparison](validation/full-model-belong.json)
- [Counterexample in all four implementations](validation/equality-counterexample.json)

### Equality boundary case outside the stated assumption

Two identical inputs have continuous channels `[1,1]` and `[2,2]`, the same binary
channel `[0,0]`, and opposite labels 0 and 1. Every deterministic classifier must
give them the same prediction, so at most one can be correct. Both original
models and both refactored models nevertheless prove a flow objective of two by
routing the samples in opposite directions at a zero-robustness predicate.

This is a counterexample if arbitrary zero robustness is allowed, not a
synthetic performance benchmark or evidence against the method under the
author's nonzero-robustness assumption. The original constraints remain
unchanged. Export detects that its strict interval is empty and does not
misrepresent this boundary case as verified strict classification.

### Refactoring bug fixed

`build_model` originally retained a reference to the caller's signal array. A
caller mutation after construction could make reported prediction scores use
different data from the already frozen MILP coefficients. The model now owns
read-only copies of its input signals and labels. A regression test failed on
the previous implementation and passes after the repair.

After the repair, **22 tests passed in 4.44 seconds** on the remote lab. Added
checks include an independent exhaustive depth-1 oracle, depth-2 coefficient
comparisons, multiclass/noncontiguous labels at depth zero, immutable input
snapshots, and the equality counterexample. The latter records behavior outside
the stated assumption; its passing test is not a supported-case accuracy result.

```bash
# Complete source/refactor solves, with every original candidate available.
python scripts/full_validation.py --mode exist --output outputs/full-exist-audit
python scripts/full_validation.py --mode belong --output outputs/full-belong-audit

# Inspect an equality boundary case outside the stated assumption.
python scripts/reproduce_equality_issue.py --output outputs/equality-issue
```

The earlier checks below are retained as historical evidence.

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
stl-tree train --config configs/naval.json --trust-pickle --time-limit 120 --output outputs/naval-validation
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
evidence is in [naval-120s.json](validation/naval-120s.json).

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
