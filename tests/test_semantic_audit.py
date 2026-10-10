"""Independent small-instance oracles and known semantic failure witnesses.

Constructed inputs below test boundary mechanisms only. Full validation uses
the supplied naval dataset and every original temporal window.
"""

import runpy
from pathlib import Path

import numpy as np
import pytest

from stl_tree import ModelConfig, SolverConfig, build_model, load_dataset, sample_data, solve

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.solver


def _exhaustive_stump(x, y, mode):
    """Enumerate the complete source depth-1 search, including zero-margin ties.

    This oracle uses direct signal extrema and label counts, never solver
    variables or constraint construction. Inputs are bounded within [0, 80], so
    the preserved +/-200 robustness bounds do not remove any threshold in [0,81].
    """
    classes = np.unique(y)
    best = max(int(np.sum(y == label)) for label in classes)

    def score(values):
        return max(
            int(np.sum(((values >= 0) & (y == left)) | ((values <= 0) & (y == right))))
            for left in classes
            for right in classes
        )

    # The discrete channel is 0/1; its branch truth has no threshold ambiguity.
    if mode == "exist":
        best = max(best, score(2 * x[:, -1].max(axis=1) - 1))
    for a in range(x.shape[2]):
        for b in range(a, x.shape[2]):
            binary = x[:, -1, a : b + 1]
            if mode == "belong":
                if np.any(binary.min(axis=1) != binary.max(axis=1)):
                    continue
                best = max(best, score(2 * binary.max(axis=1) - 1))
            for feature in range(x.shape[1] - 1):
                for operation in ("less", "greater"):
                    interval = x[:, feature, a : b + 1]
                    values = interval.min(axis=1) if operation == "less" else interval.max(axis=1)
                    breakpoints = np.unique(np.concatenate(([0.0, 81.0], values)))
                    thresholds = np.concatenate(
                        (breakpoints, (breakpoints[:-1] + breakpoints[1:]) / 2)
                    )
                    for threshold in thresholds:
                        rho = threshold - values if operation == "less" else values - threshold
                        best = max(best, score(rho))
    return best


@pytest.mark.parametrize("mode", ["exist", "belong"])
def test_independent_exhaustive_stump_matches_milp(mode):
    dataset = load_dataset(ROOT / "data/naval_dataset.pkl", trusted_pickle=True)
    x, y, _ = sample_data(*dataset.select("train"), 5, seed=19)
    # Use a short prefix ONLY for this exhaustive mechanism oracle; full runs
    # and source-equivalence audits retain the entire horizon.
    x = np.concatenate((x[:, :, :4], np.array([[[0, 0, 1, 1]]] * len(x))), axis=1)
    expected = _exhaustive_stump(x, y, mode)
    artifacts = build_model(x, y, ModelConfig(depth=1, discrete_mode=mode))
    try:
        result = solve(artifacts, SolverConfig(output_flag=False))
        assert result.proven_optimal and result.objective == pytest.approx(expected)
    finally:
        artifacts.model.dispose()


def test_known_zero_margin_ambiguity_can_split_identical_inputs():
    # A correctness counterexample for deterministic classification in the
    # inherited formulation, explicitly retained rather than hidden as accuracy.
    x = np.array([[[1, 1], [0, 0]], [[1, 1], [0, 0]]], dtype=float)
    artifacts = build_model(x, [0, 1], ModelConfig(depth=1))
    try:
        result = solve(artifacts, SolverConfig(output_flag=False))
        assert result.proven_optimal and result.objective == pytest.approx(2)
        assert result.deterministic_training_accuracy == pytest.approx(0.5)
        assert result.tie_samples == 2
    finally:
        artifacts.model.dispose()


def test_depth_zero_and_noncontiguous_multiclass_labels():
    x = np.array([[[2], [0]], [[3], [1]], [[4], [0]], [[5], [1]]], dtype=float)
    artifacts = build_model(x, [3, 8, 8, 12], ModelConfig(depth=0))
    try:
        result = solve(artifacts, SolverConfig(output_flag=False))
        assert result.proven_optimal and result.objective == pytest.approx(2)
        np.testing.assert_array_equal(result.tree.predict(x), [8, 8, 8, 8])
    finally:
        artifacts.model.dispose()


def test_data_snapshot_is_not_mutated_by_the_caller():
    x = np.array([[[1, 2], [0, 0]], [[3, 4], [1, 1]]], dtype=float)
    original_x = x.copy()
    artifacts = build_model(x, [0, 1], ModelConfig(depth=1))
    try:
        x[:] = 77
        np.testing.assert_array_equal(artifacts.signals, original_x)
        with pytest.raises(ValueError, match="read-only"):
            artifacts.signals[0, 0, 0] = 10
    finally:
        artifacts.model.dispose()


@pytest.mark.parametrize("mode", ["exist", "belong"])
def test_every_constraint_matches_source_at_depth_two(mode):
    import sys

    # The audit's implementation is also used by the full-size remote checks.
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        audit = runpy.run_path(str(ROOT / "scripts/model_audit.py"))
    finally:
        sys.path.pop(0)
    dataset = load_dataset(ROOT / "data/naval_dataset.pkl", trusted_pickle=True)
    x, y, _ = sample_data(*dataset.select("train"), 4, seed=19)
    x = np.concatenate((x[:, :, :4], np.zeros((len(x), 1, 4))), axis=1)
    config = ModelConfig(depth=2, discrete_mode=mode)
    refactored = build_model(x, y, config)
    original = audit["original_artifacts"](x, y, config)
    try:
        assert audit["compare_models"](original, refactored)["identical"]
        result = solve(refactored, SolverConfig(output_flag=False))
        assert result.proven_optimal
        assert audit["audit_solution"](refactored)["passed"]
    finally:
        original.model.dispose()
        refactored.model.dispose()
