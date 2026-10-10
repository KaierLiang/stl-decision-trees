"""Numerical export checks; constructed inputs exercise boundaries, not accuracy."""

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from stl_tree import ModelConfig, SolverConfig, build_model, solve

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.solver


@pytest.mark.parametrize("mode", ["exist", "belong"])
@pytest.mark.parametrize("operation", ["less", "greater"])
def test_export_preserves_flows_and_all_original_bounds(mode, operation, tmp_path):
    # The unselected second feature tightens the threshold's feasible interval
    # to [2, 3]. A midpoint using only the selected feature would ignore this.
    x = np.array([[[1], [8], [0]], [[3], [2], [0]]], dtype=float)
    artifacts = build_model(x, [0, 1], ModelConfig(depth=1, discrete_mode=mode, max_rho=6, big_m=6))
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        audit = runpy.run_path(str(ROOT / "scripts/model_audit.py"))
    finally:
        sys.path.pop(0)
    try:
        v = artifacts.variables
        v["op"][operation, 0][0].LB = 1
        left_label = 0 if operation == "less" else 1
        v["label"][left_label, 1].LB = 1
        v["label"][1 - left_label, 2].LB = 1
        for variable in v["source"].values():
            variable.LB = 1
        # A custom linear objective selects the endpoint threshold 3 solely for
        # this mechanism test; normal training uses the unchanged flow objective.
        artifacts.model.setObjective(v["threshold"][0], -1)
        result = solve(artifacts, SolverConfig(output_flag=False))
        assert result.raw_tree.nodes[0]["threshold"] == pytest.approx(3)
        assert result.tree.nodes[0]["threshold"] == pytest.approx(2.5)
        assert result.threshold_stabilization["verified_strict_routing"]
        np.testing.assert_array_equal(result.tree.predict(x), [0, 1])
        # Audit linear/general constraints directly. Its objective remains the
        # raw optimized threshold, so the separate objective check must differ.
        report = audit["audit_solution"](
            artifacts, assignment=audit["exported_assignment"](artifacts, result.tree)
        )
        assert report["linear_max_violation"] <= 1e-10
        assert report["bound_max_violation"] <= 1e-10
        assert report["min_max_violation"] <= 1e-10
        assert report["integrality_max_violation"] <= 1e-10
        assert report["bad_routes"] == []
        assert report["zero_robustness_routed_decisions"] == 0
        result.save(tmp_path)
        assert (tmp_path / "tree-raw.json").exists()
        raw_result = solve(artifacts, SolverConfig(output_flag=False), stabilize=False)
        assert raw_result.tree.to_dict() == raw_result.raw_tree.to_dict()
        assert raw_result.threshold_stabilization is None
    finally:
        artifacts.model.dispose()
