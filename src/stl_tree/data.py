"""Dataset loading, validation, sampling, and the original augmentation recipe."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def validate_data(signals, labels):
    """Validate an N × F × T finite signal tensor and nonnegative integer labels."""
    x = np.asarray(signals, dtype=float)
    y = np.asarray(labels)
    if x.ndim != 3 or any(size == 0 for size in x.shape):
        raise ValueError("signals must have nonempty shape (samples, features, time_steps)")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("labels must have shape (samples,) matching signals")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("signals and labels must be finite")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("labels must be nonnegative integers (map -1 to 0 when loading)")
    return x, y.astype(np.int64)


@dataclass(frozen=True)
class Dataset:
    train_signals: np.ndarray
    train_labels: np.ndarray
    test_signals: np.ndarray
    test_labels: np.ndarray

    def select(self, split="train"):
        if split == "train":
            return self.train_signals.copy(), self.train_labels.copy()
        if split == "test":
            return self.test_signals.copy(), self.test_labels.copy()
        if split == "combined":
            return (
                np.concatenate((self.train_signals, self.test_signals)),
                np.concatenate((self.train_labels, self.test_labels)),
            )
        raise ValueError("split must be 'train', 'test', or 'combined'")

    def summary(self):
        def describe(x, y):
            values, counts = np.unique(y, return_counts=True)
            return {
                "shape": list(x.shape),
                "range": [float(x.min()), float(x.max())],
                "labels": {str(int(k)): int(v) for k, v in zip(values, counts)},
            }

        return {
            "train": describe(self.train_signals, self.train_labels),
            "test": describe(self.test_signals, self.test_labels),
        }


def load_dataset(path, *, trusted_pickle=False):
    """Load four split arrays from NPZ, or from an explicitly trusted pickle.

    NPZ keys: X_train, y_train, X_test, y_test. Original -1 labels map to 0.
    Pickle loading can execute code: only enable it for files you trust.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = ("X_train", "y_train", "X_test", "y_test")
            missing = set(keys) - set(archive.files)
            if missing:
                raise ValueError(f"NPZ is missing keys: {', '.join(sorted(missing))}")
            arrays = [archive[key] for key in keys]
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        if not trusted_pickle:
            raise ValueError("Pickle requires trusted_pickle=True / --trust-pickle")
        with path.open("rb") as stream:
            arrays = pickle.load(stream)
        if not isinstance(arrays, (list, tuple)) or len(arrays) != 4:
            raise ValueError("Pickle must contain [X_train, y_train, X_test, y_test]")
    else:
        raise ValueError("Dataset extension must be .npz, .pkl, or .pickle")
    train_x, train_y, test_x, test_y = arrays
    train_x, train_y = validate_data(train_x, np.where(np.asarray(train_y) == -1, 0, train_y))
    test_x, test_y = validate_data(test_x, np.where(np.asarray(test_y) == -1, 0, test_y))
    if train_x.shape[1:] != test_x.shape[1:]:
        raise ValueError("Train and test must have matching feature/time dimensions")
    return Dataset(train_x, train_y, test_x, test_y)


def sample_data(signals, labels, samples=None, seed=42):
    """Use the original RandomState sampling sequence; return source indices too."""
    x, y = validate_data(signals, labels)
    if samples is None:
        indices = np.arange(len(x))
    else:
        if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= len(x):
            raise ValueError(f"samples must be an integer from 1 to {len(x)}, or None")
        indices = np.random.RandomState(seed).choice(len(x), samples, replace=False)
    return x[indices].copy(), y[indices].copy(), indices


def augment_with_discrete_feature(signals, labels):
    """Reproduce augment_data_with_new_feature exactly.

    Append a constant-zero channel to every original trajectory. Duplicate the
    first floor(n_label_1 / 2) label-1 trajectories, set their new channel to one,
    and relabel those copies as zero. This is an experimental data construction,
    not a generic preprocessing step. Source indices identify each copy's parent.
    """
    x, y = validate_data(signals, labels)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Discrete-feature augmentation requires binary labels 0 and 1")
    count, _, horizon = x.shape
    originals = np.concatenate((x, np.zeros((count, 1, horizon))), axis=1)
    positive_indices = np.flatnonzero(y == 1)
    copied_indices = positive_indices[: len(positive_indices) // 2]
    copies = np.concatenate((x[copied_indices], np.ones((len(copied_indices), 1, horizon))), axis=1)
    return (
        np.concatenate((originals, copies)),
        np.concatenate((y, np.zeros(len(copied_indices), dtype=np.int64))),
        np.concatenate((np.arange(count), copied_indices)),
    )
