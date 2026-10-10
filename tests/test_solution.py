"""Mechanism checks for explicit semantics, separate from real-data validation."""

import numpy as np
import pytest

from stl_tree.solution import LearnedTree


def test_prediction_ties_and_serialization(tmp_path):
    tree = LearnedTree(
        {
            0: {
                "kind": "continuous",
                "feature": 0,
                "window": [0, 1],
                "operation": "greater",
                "threshold": 2.0,
            },
            1: {"kind": "label", "label": 1},
            2: {"kind": "label", "label": 0},
        },
        2,
        2,
        "exist",
    )
    x = np.array([[[0, 3], [0, 0]], [[0, 1], [0, 0]], [[0, 2], [0, 0]]])
    with pytest.raises(ValueError, match="Zero-robustness"):
        tree.predict(x)
    left, ties = tree.predict(x, tie_break="left", return_ties=True)
    np.testing.assert_array_equal(left, [1, 0, 1])
    np.testing.assert_array_equal(ties, [False, False, True])
    np.testing.assert_array_equal(tree.predict(x, tie_break="right"), [1, 0, 0])
    path = tmp_path / "tree.json"
    tree.save(path)
    np.testing.assert_array_equal(LearnedTree.load(path).predict(x, tie_break="left"), left)


def test_belong_does_not_silently_relax_original_equalities():
    tree = LearnedTree(
        {
            0: {"kind": "belong", "feature": 1, "window": [0, 1]},
            1: {"kind": "label", "label": 1},
            2: {"kind": "label", "label": 0},
        },
        2,
        2,
        "belong",
    )
    with pytest.raises(ValueError, match="constant binary"):
        tree.predict(np.array([[[0, 1], [0, 1]]]))
