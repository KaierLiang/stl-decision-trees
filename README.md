# STL Decision Trees

Code for [Learning Optimal Signal Temporal Logic Decision Trees for Classification: A Max-Flow MILP Formulation](https://arxiv.org/abs/2407.21090).

## Installation

Requires **Python 3.10+** and a **Gurobi license** that supports the model size.

```bash
git clone https://github.com/KaierLiang/stl-decision-trees.git
cd stl-decision-trees
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, replace the activation command with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

```bash
stl-tree train --config configs/naval.json --trust-pickle --output outputs/naval
```

Results are saved to `outputs/naval/`. Use a new output directory for each run.

For the notebook:

```bash
python -m pip install -e ".[notebook]"
jupyter lab examples/quickstart.ipynb
```

[Python example](examples/python_api.py) · [Installation help](docs/getting-started.md)
