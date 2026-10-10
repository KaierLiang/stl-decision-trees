# Original experiment snapshot

These three Python files are unmodified copies of the supplied source. The initial
Git commit also contains them in their original locations. They are preserved for
provenance and regression comparison, not used as importable package entry points.

- `main_belong.py`: ECG-oriented experiment, global `exist` discrete feature.
- `main_belong_keep_temporal_feature.py`: naval experiment, temporal `belong` variant.
- `utility.py`: original preprocessing and helper functions.

The scripts assume data under `../original/` and immediately load data and optimize
when executed/imported. Those relative paths were already unavailable in the supplied
directory. Use the CLI or examples in the repository root for maintained entry points.

The regression harness evaluates the original model-construction statements with
controlled real-data inputs, correcting only the numeric constant argument in one
Gurobi min call for API compatibility. See `tests/test_solver_regression.py`.
