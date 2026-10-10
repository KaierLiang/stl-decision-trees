"""Incumbent-safe solution extraction, portable trees, and explicit tie handling."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import SolverConfig


@dataclass
class LearnedTree:
    nodes: dict[int, dict]
    feature_count: int
    horizon: int
    discrete_mode: str

    def to_dict(self):
        return {
            "schema_version": 1,
            "nodes": self.nodes,
            "feature_count": self.feature_count,
            "horizon": self.horizon,
            "discrete_mode": self.discrete_mode,
        }

    @classmethod
    def from_dict(cls, value):
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported tree schema_version")
        return cls(
            {int(k): v for k, v in value["nodes"].items()},
            value["feature_count"],
            value["horizon"],
            value["discrete_mode"],
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def predict(self, signals, *, tie_break="error", tolerance=1e-6, return_ties=False):
        """Route augmented signals; ties error by default, or choose left/right.

        The training MILP permits either branch at zero robustness. Prediction
        must choose an explicit policy. A tolerance marks near-zero values too.
        The returned tie mask marks samples hitting such a continuous predicate.
        """
        if tie_break not in {"error", "left", "right"}:
            raise ValueError("tie_break must be 'error', 'left', or 'right'")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        x = np.asarray(signals, dtype=float)
        if x.ndim != 3 or x.shape[1:] != (self.feature_count, self.horizon):
            raise ValueError(f"Expected shape (N, {self.feature_count}, {self.horizon})")
        if not np.isfinite(x).all() or not np.isin(x[:, -1], [0, 1]).all():
            raise ValueError("Signals must be finite with a final binary channel")
        predictions = np.empty(len(x), dtype=np.int64)
        tied = np.zeros(len(x), dtype=bool)
        for sample, signal in enumerate(x):
            node_id = 0
            for _ in range(len(self.nodes) + 1):
                node = self.nodes[node_id]
                if node["kind"] == "label":
                    predictions[sample] = node["label"]
                    break
                feature = node["feature"]
                if node["kind"] == "continuous":
                    a, b = node["window"]
                    values = signal[feature, a : b + 1]
                    rho = (
                        node["threshold"] - values.min()
                        if node["operation"] == "less"
                        else values.max() - node["threshold"]
                    )
                    is_tie = abs(rho) <= tolerance
                    tied[sample] |= is_tie
                    if is_tie and tie_break == "error":
                        raise ValueError(
                            f"Zero-robustness tie at sample {sample}, node {node_id}; "
                            "choose tie_break='left' or 'right' explicitly"
                        )
                    take_left = tie_break == "left" if is_tie else rho > 0
                elif node["kind"] == "exist":
                    take_left = bool(signal[feature].max())
                elif node["kind"] == "belong":
                    a, b = node["window"]
                    values = signal[feature, a : b + 1]
                    if values.min() != values.max():
                        raise ValueError(
                            "The original belong equations require a constant binary "
                            f"channel within window {a, b} at node {node_id}"
                        )
                    take_left = bool(values.max())
                else:
                    raise ValueError(f"Unknown predicate kind: {node['kind']}")
                node_id = 2 * node_id + (1 if take_left else 2)
            else:
                raise ValueError("Tree has no reachable label")
        return (predictions, tied) if return_ties else predictions

    def format(self):
        """Readable active tree, excluding descendants of nodes already labelled."""
        lines = []

        def visit(node_id, indent, branch):
            node = self.nodes[node_id]
            if node["kind"] == "label":
                text = f"label {node['label']}"
            elif node["kind"] == "continuous":
                a, b = node["window"]
                relation = "<=" if node["operation"] == "less" else ">="
                text = f"F[{a},{b}](s{node['feature'] + 1} {relation} {node['threshold']:.8g})"
            else:
                interval = f"{tuple(node['window'])}" if "window" in node else ""
                text = f"{node['kind']}(s{node['feature'] + 1}{interval})"
            lines.append(f"{indent}{branch}node {node_id}: {text}")
            if node["kind"] != "label":
                visit(2 * node_id + 1, indent + "  ", "true  -> ")
                visit(2 * node_id + 2, indent + "  ", "false -> ")

        visit(0, "", "")
        return "\n".join(lines)


@dataclass
class SolveResult:
    status: str
    status_code: int
    has_incumbent: bool
    proven_optimal: bool
    runtime_seconds: float
    objective: float | None
    objective_bound: float | None
    mip_gap: float | None
    training_samples: int
    max_flow_training_fraction: float | None
    deterministic_training_accuracy: float | None
    tie_samples: int | None
    tree: LearnedTree | None
    solver_config: dict
    model_config: dict
    model_size: dict

    def to_dict(self):
        value = asdict(self)
        value.pop("tree")
        value["prediction_tie_break"] = "left"
        value["prediction_tolerance"] = 1e-6
        value["notes"] = [
            "Max-flow fraction is an in-sample optimization statistic, not held-out accuracy.",
            "The original zero-margin constraints permit either branch at robustness zero.",
        ]
        return value

    def save(self, directory, *, metadata=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any((directory / name).exists() for name in ("summary.json", "tree.json", "tree.txt")):
            raise FileExistsError(f"Result already exists in {directory}; choose a fresh directory")
        summary = self.to_dict()
        summary["metadata"] = metadata or {}
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if self.tree is not None:
            self.tree.save(directory / "tree.json")
            (directory / "tree.txt").write_text(self.tree.format() + "\n", encoding="utf-8")


def _extract_tree(artifacts):
    variables = artifacts.variables
    nodes = {}

    def visit(node):
        label = next((k for k in artifacts.classes if variables["label"][k, node].X > 0.5), None)
        if label is not None:
            nodes[node] = {"kind": "label", "label": label}
            return
        if variables["discrete"][node].X > 0.5:
            item = {
                "kind": artifacts.config.discrete_mode,
                "feature": artifacts.signals.shape[1] - 1,
            }
        else:
            operation, feature = next(
                key for key, values in variables["op"].items() if values[node].X > 0.5
            )
            item = {
                "kind": "continuous",
                "operation": operation,
                "feature": feature,
                "threshold": float(variables["threshold"][node].X),
            }
        if item["kind"] != "exist":
            item["window"] = list(
                next(
                    interval
                    for interval in artifacts.windows
                    if variables["window"][node][interval].X > 0.5
                )
            )
        nodes[node] = item
        visit(2 * node + 1)
        visit(2 * node + 2)

    visit(0)
    return LearnedTree(
        nodes,
        artifacts.signals.shape[1],
        artifacts.signals.shape[2],
        artifacts.config.discrete_mode,
    )


def solve(artifacts, config=None, *, log_file=None):
    """Solve and report any incumbent, including after a time limit.

    Never access solution values without an incumbent. The caller owns the
    model lifetime; call artifacts.model.dispose() after exporting what you need.
    """
    import gurobipy as gp

    config = config or SolverConfig()
    model = artifacts.model
    model.Params.OutputFlag = int(config.output_flag)
    model.Params.Threads = config.threads
    model.Params.Seed = config.seed
    model.Params.MIPGap = config.mip_gap
    model.Params.TimeLimit = gp.GRB.INFINITY if config.time_limit is None else config.time_limit
    if log_file is not None:
        model.Params.LogFile = str(log_file)
    model.optimize()
    names = [
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "INPROGRESS",
        "USER_OBJ_LIMIT",
        "WORK_LIMIT",
        "MEM_LIMIT",
    ]
    status = next(
        (name for name in names if getattr(gp.GRB, name, None) == model.Status),
        f"STATUS_{model.Status}",
    )
    incumbent = model.SolCount > 0
    tree = _extract_tree(artifacts) if incumbent else None
    objective = float(model.ObjVal) if incumbent else None
    try:
        bound = float(model.ObjBound)
        if not np.isfinite(bound) or abs(bound) >= gp.GRB.INFINITY:
            bound = None
    except (AttributeError, gp.GurobiError):
        bound = None
    gap = float(model.MIPGap) if incumbent else None
    if gap is not None and (not np.isfinite(gap) or gap >= gp.GRB.INFINITY):
        gap = None
    accuracy, tie_count = None, None
    if tree is not None:
        predicted, ties = tree.predict(artifacts.signals, tie_break="left", return_ties=True)
        accuracy = float(np.mean(predicted == artifacts.labels))
        tie_count = int(ties.sum())
    # OPTIMAL with a positive requested MIPGap need not certify an exact optimum.
    proven = (
        model.Status == gp.GRB.OPTIMAL
        and objective is not None
        and bound is not None
        and abs(objective - bound) <= 1e-6
    )
    return SolveResult(
        status,
        int(model.Status),
        incumbent,
        proven,
        float(model.Runtime),
        objective,
        bound,
        gap,
        len(artifacts.labels),
        objective / len(artifacts.labels) if incumbent else None,
        accuracy,
        tie_count,
        tree,
        config.to_dict(),
        artifacts.config.to_dict(),
        {
            "variables": model.NumVars,
            "linear_constraints": model.NumConstrs,
            "general_constraints": model.NumGenConstrs,
            "windows": len(artifacts.windows),
        },
    )
