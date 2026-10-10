"""Compare optima against the actual original constraint code on real naval data.

Run on remote labs. The source is evaluated only from tree construction through
objective construction, skipping its hard-coded I/O, optimization, and output.
The exist script needs one Gurobi API correction: a numeric min operand is passed
via 'constant'. These tests use two continuous channels, for which the legacy
len(feature) index is already the correct discrete channel.
"""

import ast
import runpy
from pathlib import Path

import numpy as np
import pytest

from stl_tree import (
    ModelConfig,
    SolverConfig,
    augment_with_discrete_feature,
    build_model,
    load_dataset,
    sample_data,
    solve,
)
from stl_tree.primitives import generate_windows

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.solver


def legacy_model(x, y, mode, *, depth=1, return_namespace=False):
    import gurobipy as gp

    filename = "main_belong.py" if mode == "exist" else "main_belong_keep_temporal_feature.py"
    source = (ROOT / "legacy" / filename).read_text(encoding="utf-8")
    if mode == "exist":
        old = "[op_vars[f'{op}_{feature}'][node], get_discrete_feature(data[i][feature_idx+len(feature)])],"
        new = (
            "[op_vars[f'{op}_{feature}'][node]], "
            "constant=float(get_discrete_feature(data[i][feature_idx+len(feature)])),"
        )
        assert source.count(old) == 1
        source = source.replace(old, new)
    statements = ast.parse(source).body
    start = next(
        i
        for i, node in enumerate(statements)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Tuple)
        and [n.id for n in node.targets[0].elts] == ["n", "decision_nodes", "leaf_nodes"]
    )
    stop = next(
        i
        for i, node in enumerate(statements)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "optimize"
    )
    namespace = runpy.run_path(str(ROOT / "legacy/utility.py"))
    namespace.update(
        Model=gp.Model,
        GRB=gp.GRB,
        quicksum=gp.quicksum,
        data=x,
        true_labels=y,
        data_count=len(y),
        num_features=x.shape[1] - 1,
        num_labels=len(np.unique(y)),
        comb=generate_windows(x.shape[2], 500),
        depth=depth,
    )
    exec(
        compile(ast.Module(body=statements[start:stop], type_ignores=[]), filename, "exec"),
        namespace,
    )
    return namespace if return_namespace else namespace["m"]


@pytest.mark.parametrize("mode", ["exist", "belong"])
def test_original_and_refactored_optima_agree(mode):
    dataset = load_dataset(ROOT / "data/naval_dataset.pkl", trusted_pickle=True)
    x, y, _ = sample_data(*dataset.select("combined"), 8, 42)
    x, y, _ = augment_with_discrete_feature(x, y)
    original = legacy_model(x, y, mode)
    artifacts = build_model(x, y, ModelConfig(depth=1, discrete_mode=mode))
    try:
        original.Params.OutputFlag = 0
        original.Params.Threads = 1
        original.Params.Seed = 42
        original.Params.TimeLimit = 60
        original.Params.MIPGap = 0
        original.optimize()
        result = solve(artifacts, SolverConfig(time_limit=60, output_flag=False))
        assert original.Status == 2, "Original regression instance did not reach an optimum"
        assert result.proven_optimal
        assert result.objective == pytest.approx(original.ObjVal, abs=1e-6)
        assert artifacts.model.NumVars == original.NumVars
        assert artifacts.model.NumConstrs == original.NumConstrs
        assert artifacts.model.NumGenConstrs == original.NumGenConstrs
        assert len(artifacts.windows) == 1891
    finally:
        original.dispose()
        artifacts.model.dispose()


def test_no_incumbent_does_not_access_solution_values(tmp_path):
    # A deliberate infeasible constraint checks an error path, not classification quality.
    x = np.array([[[1, 2], [0, 0]], [[3, 4], [0, 0]]], dtype=float)
    artifacts = build_model(x, [0, 1], ModelConfig(depth=0))
    try:
        artifacts.model.addConstr(artifacts.variables["source"][0] >= 2)
        result = solve(artifacts, SolverConfig(output_flag=False))
        assert result.status == "INFEASIBLE"
        assert not result.has_incumbent and result.tree is None
        assert result.objective is None and result.mip_gap is None
        result.save(tmp_path)
        assert (tmp_path / "summary.json").exists()
        assert not (tmp_path / "tree.json").exists()
    finally:
        artifacts.model.dispose()


def test_discrete_channel_index_is_not_tied_to_feature_name_length():
    # One continuous channel used to address index 2 (out of range) in exist mode.
    x = np.array([[[1, 2], [0, 0]], [[1, 2], [1, 1]]], dtype=float)
    artifacts = build_model(x, [0, 1], ModelConfig(depth=1))
    try:
        result = solve(artifacts, SolverConfig(output_flag=False))
        assert result.proven_optimal and result.objective == pytest.approx(2)
    finally:
        artifacts.model.dispose()


def test_feasible_early_stop_is_exported_without_claiming_optimality(tmp_path):
    dataset = load_dataset(ROOT / "data/naval_dataset.pkl", trusted_pickle=True)
    x, y, _ = sample_data(*dataset.select("train"), 8, 42)
    x, y, _ = augment_with_discrete_feature(x, y)
    artifacts = build_model(x, y, ModelConfig(depth=1))
    try:
        artifacts.model.Params.SolutionLimit = 1
        result = solve(artifacts, SolverConfig(time_limit=60, output_flag=False))
        assert result.status == "SOLUTION_LIMIT"
        assert result.has_incumbent and not result.proven_optimal
        assert result.objective is not None and result.tree is not None
        result.save(tmp_path)
        assert (tmp_path / "tree.json").exists()
        with pytest.raises(FileExistsError):
            result.save(tmp_path)
    finally:
        artifacts.model.dispose()
