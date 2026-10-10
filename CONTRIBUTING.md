# Contributing

Use a private branch and describe the behavior changed by each patch. Preserve
the original model semantics unless a formulation change is explicitly intended
and documented. In particular, do not silently change the data augmentation,
window family, objective, threshold bounds, or equality routing.

Install development dependencies in an isolated environment on a remote lab:

```bash
python -m pip install -e ".[dev]"
bash scripts/validate.sh
```

The `solver` tests need a working Gurobi license. To run only data and prediction
checks, use `python -m pytest -m 'not solver'`; this does not validate optimization.
Solver tests use the actual naval file and compare against the original equations.
Small constructed inputs in boundary-condition tests check software mechanisms,
not experimental classification performance.

Keep `legacy/` unchanged. Do not commit solver output directories, license files,
credentials, or local environment folders. New experiment evidence should state
the data split, augmentation, seed, window family, depth, time limit, status, and
optimality gap. Always distinguish in-sample flow from deterministic prediction
and from held-out evaluation.

No automatic cloud CI or publishing workflow is enabled. See
[remote validation](docs/validation.md) for the authorized test environment.
