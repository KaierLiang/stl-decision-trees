"""Explicit model constants and solver controls; defaults match the supplied scripts."""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class ModelConfig:
    depth: int = 3
    max_window_span: int = 500
    discrete_mode: str = "exist"
    max_rho: float = 200.0
    big_m: float = 250.0

    def __post_init__(self):
        for name in ("depth", "max_window_span"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.discrete_mode not in {"exist", "belong"}:
            raise ValueError("discrete_mode must be 'exist' or 'belong'")
        for name in ("max_rho", "big_m"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.big_m < self.max_rho:
            raise ValueError("big_m must be at least max_rho")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SolverConfig:
    time_limit: float | None = None
    mip_gap: float = 0.0
    threads: int = 1
    seed: int = 42
    output_flag: bool = True

    def __post_init__(self):
        if self.time_limit is not None and (not isfinite(self.time_limit) or self.time_limit <= 0):
            raise ValueError("time_limit must be finite and positive, or None")
        if not isfinite(self.mip_gap) or not 0 <= self.mip_gap <= 1:
            raise ValueError("mip_gap must be between 0 and 1")
        for name in ("threads", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def to_dict(self):
        return asdict(self)
