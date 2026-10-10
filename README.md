# STL Decision Trees

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Solver](https://img.shields.io/badge/Solver-Gurobi-CC0000)](https://www.gurobi.com/)
[![Paper](https://img.shields.io/badge/arXiv-2407.21090-B31B1B)](https://arxiv.org/abs/2407.21090)

Code accompanying **Learning Optimal Signal Temporal Logic Decision Trees for
Classification: A Max-Flow MILP Formulation**, by Kaier Liang, Gustavo A. Cardona,
Disha Kamale, and Cristian-Ioan Vasile (CDC 2024).

This repository organizes the supplied `attachment2` research scripts into
reusable Python modules. It contains the **exist** and **temporal belong** variants,
the naval dataset, a CLI, a Python API, and the original scripts for comparison.

> **Scope.** This is a refactoring of the supplied experiment code, not a claim
> that every paper experiment has been reproduced. The ECG data are not included.
> The supplied augmentation changes both data and labels; the legacy experiment
> pools the original train/test splits. See [experiment semantics](docs/semantics.md)
> before interpreting any score. Gurobi and a suitable license are required to train.

## Install

```bash
git clone https://github.com/KaierLiang/stl-decision-trees.git
cd stl-decision-trees
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
The repository is currently private, so cloning requires access.
The solver license must accommodate the model size; see
[installation and troubleshooting](docs/getting-started.md).

## Quickstart: inspect → train → reuse

### 1. Inspect the bundled real dataset

```bash
stl-tree inspect data/naval_dataset.pkl --trust-pickle
```

The dataset has 1,600 training and 400 test trajectories, each with two continuous
channels and 61 time steps. Legacy labels `-1` and `1` are mapped to `0` and `1`.
Only trust pickle files whose source you know. For your own datasets, prefer NPZ.

### 2. Run a small, time-limited example

```bash
stl-tree train --config configs/naval_quickstart.json --trust-pickle --output outputs/quickstart
```

This example samples **12 real training trajectories**, applies the original
augmentation, and learns a depth-1 tree with a 30-second solver limit. It retains
**all 1,891 temporal windows**. It is an onboarding example, not paper validation.
Use a fresh output directory for every run.

### 3. Read the result

```text
outputs/quickstart/
├── run.json              # input fingerprint, data settings, model/solver configuration
├── sample_indices.npz    # sampled rows and augmentation parent mapping
├── summary.json          # status, objective, bound, gap, timing, and separate scores
├── solver.log            # solver output (when logging is enabled)
├── tree.json             # portable tree; only present when an incumbent exists
└── tree.txt              # readable active tree
```

`max_flow_training_fraction` is the in-sample optimization statistic.
`deterministic_training_accuracy` evaluates the extracted tree with an explicit
left-branch tie policy. These can differ: the original MILP allows either branch
at zero robustness. `tie_samples` counts affected samples on the prediction path.
Always check `status`, `has_incumbent`, `objective_bound`, and `mip_gap`.

### 4. Reuse a saved tree without starting a solver

```python
from stl_tree import LearnedTree, augment_with_discrete_feature, load_dataset

dataset = load_dataset("data/naval_dataset.pkl", trusted_pickle=True)
X_test, y_test, parents = augment_with_discrete_feature(*dataset.select("test"))
tree = LearnedTree.load("outputs/quickstart/tree.json")
predictions, ties = tree.predict(X_test, tie_break="left", return_ties=True)
print("Augmented held-out accuracy:", (predictions == y_test).mean())
print("Samples reaching a threshold tie:", ties.sum())
```

This held-out example is appropriate after the **train-only quickstart**. A tree
trained with `split="combined"` has already seen the original test partition;
its score there is not held-out evaluation. The example applies the experimental
augmentation independently inside the test split.

## Python building blocks

```python
from stl_tree import (
    ModelConfig, SolverConfig, load_dataset, sample_data,
    augment_with_discrete_feature, build_model, solve,
)

# 1. Load and select data.
dataset = load_dataset("data/naval_dataset.pkl", trusted_pickle=True)
X, y, selected = sample_data(*dataset.select("train"), samples=12, seed=42)

# 2. Apply the source experiment's augmentation explicitly.
X, y, parents = augment_with_discrete_feature(X, y)

# 3. Build the MILP. You can inspect or export it before optimization.
artifacts = build_model(X, y, ModelConfig(depth=1, discrete_mode="exist"))
try:
    # 4. Solve, inspect status, and save an incumbent when available.
    result = solve(artifacts, SolverConfig(time_limit=30, threads=1))
    print(result.status, result.objective, result.objective_bound)
    if result.tree is not None:
        print(result.tree.format())
    result.save("outputs/api-example")
finally:
    artifacts.model.dispose()
```

Runnable versions: [CLI example](examples/train_naval.py),
[Python API example](examples/python_api.py), and
[cell-by-cell notebook](examples/quickstart.ipynb).

## Experiment presets

| Preset | Split / sample count before augmentation | Depth | Discrete variant | Purpose |
| --- | --- | --- | --- | --- |
| `naval_quickstart.json` | train / 12 | 1 | exist | 30-second onboarding run |
| `naval_legacy.json` | combined / 100 | 3 | belong | source naval configuration |
| `ecg_legacy.json` | combined / 200 | 3 | exist | source ECG configuration; data missing |

```bash
# Original naval data settings and model constants; explicit solver budget.
stl-tree train --config configs/naval_legacy.json --trust-pickle --time-limit 600 --output outputs/naval

# Select the other supplied variant; no editing of the source is needed.
stl-tree train --config configs/naval_legacy.json --trust-pickle --discrete-mode exist --time-limit 600 --output outputs/naval-exist

# Inspect every setting and export the generated formulation if needed.
stl-tree train --help
```

CLI flags override JSON configuration. Dataset paths inside JSON are relative to
the JSON file. CLI paths are relative to the current working directory.
Omitting a time limit in a legacy preset allows the solver to run without a
time cap. `--max-window-span` bounds **end − start**, not the inclusive length;
the default 500 includes every interval of the supplied 61-step data.

## Bring your own data

```python
import numpy as np

# X arrays: (samples, features, time_steps); y arrays: (samples,).
# Both splits must have the same feature count and time horizon.
np.savez_compressed("my_data.npz", X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test)
```

```bash
stl-tree inspect my_data.npz
stl-tree train --dataset my_data.npz --split train --samples 100 --augmentation none --output outputs/custom
```

With `--augmentation none`, the final input channel must already be binary; it is
used for the discrete predicate. All preceding channels are continuous. With
`--augmentation legacy`, inputs contain continuous channels and labels must be
binary `0/1`; the source recipe appends the discrete channel and relabelled copies.
The model accepts nonnegative integer class IDs when no augmentation is applied.
Thresholds remain nonnegative and robustness bounds remain fixed by default;
consult [semantics and limitations](docs/semantics.md) before changing data scale.

## Repository map

```text
src/stl_tree/
  data.py          # validated loading, sampling, augmentation
  config.py        # model constants and solver controls
  primitives.py    # tree indexing, temporal windows, cached extrema
  model.py         # tree / flow / primitive / routing constraint blocks
  solution.py      # status-safe extraction, serialization, prediction
  cli.py           # inspect, convert, train
configs/           # explicit experiment settings
data/              # original naval pickle and provenance notes
examples/          # runnable Python examples and notebook cells
legacy/            # unmodified source scripts
tests/             # data and formulation regressions
docs/              # onboarding, semantics, remote validation, release notes
```

## Development and remote validation

```bash
python -m pip install -e ".[dev]"
bash scripts/validate.sh
```

Project validation is run on remote labs. The regression suite compares the
refactored MILP to the original constraint code on actual naval trajectories,
using the full window family. See [validation](docs/validation.md) for the recorded
environment and results, and [contributing](CONTRIBUTING.md) for the workflow.

## Citation

Paper: [arXiv:2407.21090](https://arxiv.org/abs/2407.21090).
Full author metadata is included in [CITATION.cff](CITATION.cff).

```bibtex
@misc{liang2024learningoptimalsignaltemporal,
  title={Learning Optimal Signal Temporal Logic Decision Trees for
         Classification: A Max-Flow MILP Formulation},
  author={Kaier Liang and Gustavo A. Cardona and Disha Kamale and Cristian-Ioan Vasile},
  year={2024},
  eprint={2407.21090},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.21090}
}
```

## Distribution status

The repository remains private. An open-source license and dataset redistribution
terms have not been selected or verified; no license grant is implied. See the
[release checklist](docs/release-checklist.md) before a future public release.
