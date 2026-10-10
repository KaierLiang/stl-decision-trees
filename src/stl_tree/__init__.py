"""Reusable building blocks for the supplied max-flow MILP experiments.

Importing this package does not load data, create a solver, or start training.
"""

from .config import ModelConfig, SolverConfig
from .data import augment_with_discrete_feature, load_dataset, sample_data
from .model import build_model
from .solution import LearnedTree, solve

__all__ = [
    "ModelConfig",
    "SolverConfig",
    "augment_with_discrete_feature",
    "load_dataset",
    "sample_data",
    "build_model",
    "LearnedTree",
    "solve",
]
__version__ = "0.1.0"
