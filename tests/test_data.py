"""Data integrity and regression checks; no optimization is performed here."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from stl_tree import augment_with_discrete_feature, load_dataset, sample_data
from stl_tree.primitives import generate_windows

ROOT = Path(__file__).resolve().parents[1]


def test_naval_shape_labels_and_pickle_opt_in():
    path = ROOT / "data/naval_dataset.pkl"
    with pytest.raises(ValueError, match="trusted_pickle"):
        load_dataset(path)
    dataset = load_dataset(path, trusted_pickle=True)
    assert dataset.train_signals.shape == (1600, 2, 61)
    assert dataset.test_signals.shape == (400, 2, 61)
    assert set(dataset.train_labels) == {0, 1}
    combined, labels = dataset.select("combined")
    assert combined.shape == (2000, 2, 61)
    np.testing.assert_array_equal(labels[:1600], dataset.train_labels)


def test_augmentation_and_sampling_match_original_on_real_data():
    original = runpy.run_path(str(ROOT / "legacy/utility.py"))
    dataset = load_dataset(ROOT / "data/naval_dataset.pkl", trusted_pickle=True)
    x, y = dataset.select("combined")
    np.random.seed(42)
    expected_indices = np.random.choice(len(x), 100, replace=False)
    selected_x, selected_y, indices = sample_data(x, y, 100, 42)
    np.testing.assert_array_equal(indices, expected_indices)
    expected = original["augment_data_with_new_feature"](x[indices], y[indices])
    actual = augment_with_discrete_feature(selected_x, selected_y)
    for old, new in zip(expected, actual):
        np.testing.assert_array_equal(old, new)


def test_window_family_is_not_restricted():
    original = runpy.run_path(str(ROOT / "legacy/utility.py"))
    windows = generate_windows(61, 500)
    assert len(windows) == 1891
    assert windows == original["generate_combinations"](0, 60, 500)
    assert (60, 60) in windows and (0, 60) in windows
    assert generate_windows(4, 1) == original["generate_combinations"](0, 3, 1)


def test_npz_roundtrip(tmp_path):
    dataset = load_dataset(ROOT / "data/naval_dataset.pkl", trusted_pickle=True)
    path = tmp_path / "data.npz"
    np.savez_compressed(
        path,
        X_train=dataset.train_signals,
        y_train=dataset.train_labels,
        X_test=dataset.test_signals,
        y_test=dataset.test_labels,
    )
    restored = load_dataset(path)
    np.testing.assert_array_equal(restored.train_signals, dataset.train_signals)
    np.testing.assert_array_equal(restored.test_labels, dataset.test_labels)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="nonempty shape"):
        sample_data(np.zeros((2, 3)), [0, 1])
    with pytest.raises(ValueError, match="samples must"):
        sample_data(np.zeros((2, 1, 3)), [0, 1], samples=3)
    with pytest.raises(ValueError, match="binary labels"):
        augment_with_discrete_feature(np.zeros((2, 1, 3)), [0, 2])
