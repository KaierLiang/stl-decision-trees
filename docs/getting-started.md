# Installation and troubleshooting

## Requirements

- Python 3.10 or newer; the tested version is recorded in [validation](validation.md).
- NumPy and the Gurobi Python API, installed by `pip install -e .`.
- A Gurobi license that supports the resulting model size.

Create a dedicated environment. Installing `gurobipy` does not by itself grant an
unrestricted solver license. Configure a license on the machine doing the solve
using the [official Gurobi setup guide](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python).
Do not commit a license file or credentials.

## Three commands to start

```bash
python -m pip install -e .
stl-tree inspect data/naval_dataset.pkl --trust-pickle
stl-tree train --config configs/naval_quickstart.json --trust-pickle --output outputs/first-run
```

`python -m stl_tree` is equivalent to `stl-tree`, which is useful if the script
directory is absent from your shell's PATH. No solve is started by importing
`stl_tree`. Data inspection, conversion, and saved-tree prediction need no solver
license.

## Notebooks

```bash
python -m pip install -e ".[notebook]"
jupyter lab examples/quickstart.ipynb
```

The notebook uses the same real naval data and model API as the CLI. Its cells
separate loading, augmentation, model construction, optimization, and evaluation.
Choose a Python kernel in the environment where the package is installed.

## Convert pickle data to NPZ

```bash
stl-tree convert data/naval_dataset.pkl data/naval_dataset.npz --trust-pickle
stl-tree inspect data/naval_dataset.npz
```

The converter refuses to overwrite an existing file and writes only numeric NPZ
arrays. A new NPZ has a different file hash from the original pickle even when
the numeric data match.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| License error or model too large for the license | Configure a suitable Gurobi license on the compute machine. Even the quickstart keeps all 1,891 windows. |
| Pickle opt-in error | Use `--trust-pickle` only for trusted input, or use NPZ. |
| Output directory already exists | Choose a new `--output` directory; runs do not overwrite earlier evidence. |
| `TIME_LIMIT` with an incumbent | Inspect the bound and gap. The saved tree is a feasible incumbent, not necessarily optimal. |
| No incumbent | Read `summary.json` and the solver log; no tree is exported. CLI exit status is 2. |
| Flow score differs from prediction accuracy | Inspect `threshold_stabilization`, `tree-raw.json`, and `tie_samples`; equality is excluded by the intended assumption but can occur in solver output. |
| Nonconstant binary channel in `belong` prediction | The inherited model equates both the interval min and max; see [semantics](semantics.md). |
| Unexpected infeasibility on differently scaled data | Check nonnegative thresholds, `max_rho`, and `big_m`; bounds affect the feasible model. |

The `train` time limit applies to Gurobi optimization, not file I/O or model
construction. `summary.json` records construction and optimization time separately.
Large depths increase node counts exponentially. Increasing temporal resolution
increases the number of intervals quadratically when all intervals are allowed.
