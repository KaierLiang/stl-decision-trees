"""Max-flow MILP, organized into the constraint blocks used by the source scripts.

Both discrete variants share the continuous primitive, tree, and flow blocks.
No pruning, window subsampling, objective penalties, or margin are introduced.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ModelConfig
from .data import validate_data
from .primitives import generate_nodes, generate_windows, window_extrema


@dataclass
class ModelArtifacts:
    model: Any
    config: ModelConfig
    signals: np.ndarray
    labels: np.ndarray
    classes: list[int]
    windows: list[tuple[int, int]]
    decision_nodes: list[int]
    leaf_nodes: list[int]
    variables: dict


def _tree_block(m, gp, count, decisions, leaves, classes, features):
    """One label or predicate at internal nodes; exactly one label at leaves."""
    operations = ("less", "greater")
    op = {
        (kind, feature): m.addVars(decisions, vtype=gp.GRB.BINARY, name=f"{kind}_s{feature + 1}")
        for kind in operations
        for feature in range(features)
    }
    discrete = m.addVars(decisions, vtype=gp.GRB.BINARY, name="discrete")
    label = m.addVars(classes, range(count), vtype=gp.GRB.BINARY, name="label")
    for node in leaves:
        m.addConstr(gp.quicksum(label[k, node] for k in classes) == 1, f"leaf_label_{node}")
    for node in decisions:
        m.addConstr(
            gp.quicksum(variables[node] for variables in op.values())
            + discrete[node]
            + gp.quicksum(label[k, node] for k in classes)
            == 1,
            f"node_choice_{node}",
        )
    return op, discrete, label


def _flow_block(m, gp, labels, count, decisions, leaves, label):
    """Conserve sample flow, permitting a sink only at its ground-truth label."""
    samples = len(labels)
    flow = m.addVars(samples, count, vtype=gp.GRB.BINARY, name="z")
    sink = m.addVars(samples, count, vtype=gp.GRB.BINARY, name="z_t")
    source = m.addVars(samples, vtype=gp.GRB.BINARY, name="z_s")
    for sample, target in enumerate(labels):
        m.addConstr(source[sample] == flow[sample, 0], f"source_{sample}")
        for node in range(count):
            m.addConstr(sink[sample, node] <= label[int(target), node], f"sink_{sample}_{node}")
        for node in decisions:
            m.addConstr(
                flow[sample, node]
                == flow[sample, 2 * node + 1] + flow[sample, 2 * node + 2] + sink[sample, node],
                f"flow_{sample}_{node}",
            )
        for node in leaves:
            m.addConstr(flow[sample, node] == sink[sample, node], f"leaf_flow_{sample}_{node}")
    return flow, sink, source


def _primitive_block(m, gp, x, config, decisions, windows, op, discrete):
    """Continuous F primitives and the original exist/belong discrete equations."""
    samples, feature_count, _ = x.shape
    continuous_count = feature_count - 1
    # Deliberately preserve Gurobi's default nonnegative threshold lower bound.
    threshold = m.addVars(decisions, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="threshold")
    window = {
        node: m.addVars(windows, vtype=gp.GRB.BINARY, name=f"window_{node}") for node in decisions
    }
    for node in decisions:
        selected = gp.quicksum(variables[node] for variables in op.values())
        if config.discrete_mode == "belong":
            selected += discrete[node]
        m.addConstr(selected == window[node].sum(), f"window_choice_{node}")

    rho = {
        key: m.addVars(
            samples,
            decisions,
            lb=-config.max_rho,
            ub=config.max_rho,
            vtype=gp.GRB.CONTINUOUS,
            name=f"rho_{key[0]}_{key[1]}",
        )
        for key in op
    }
    discrete_rho = m.addVars(samples, decisions, vtype=gp.GRB.CONTINUOUS, name="rho_discrete")
    minima, maxima = window_extrema(x, windows) if decisions else (None, None)
    for sample in range(samples):
        for node in decisions:
            for feature in range(continuous_count):
                for kind, sign, coefficients in (
                    ("less", 1, -minima[sample, feature]),
                    ("greater", -1, maxima[sample, feature]),
                ):
                    m.addConstr(
                        gp.quicksum(
                            float(coefficients[k]) * window[node][interval]
                            for k, interval in enumerate(windows)
                        )
                        + sign * threshold[node]
                        == rho[kind, feature][sample, node],
                        f"rho_{kind}_{feature}_{sample}_{node}",
                    )
            if config.discrete_mode == "exist":
                # Use the explicit constant argument, supported by current Gurobi.
                m.addGenConstrMin(
                    discrete_rho[sample, node],
                    [discrete[node]],
                    constant=float(x[sample, -1].max()),
                    name=f"exist_{sample}_{node}",
                )
            else:
                # Preserve BOTH equalities in the temporal variant. They require
                # the binary channel to be constant within the selected window.
                for name, coefficients in (
                    ("min", minima[sample, -1]),
                    ("max", maxima[sample, -1]),
                ):
                    m.addConstr(
                        gp.quicksum(
                            float(coefficients[k]) * window[node][interval]
                            for k, interval in enumerate(windows)
                        )
                        == discrete_rho[sample, node],
                        f"belong_{name}_{sample}_{node}",
                    )
    return threshold, window, rho, discrete_rho


def _routing_block(m, gp, samples, decisions, config, op, discrete, rho, discrete_rho, flow):
    """Gate left/right flow using zero-margin robustness, as in the originals."""
    sign = {
        key: m.addVars(samples, decisions, vtype=gp.GRB.BINARY, name=f"sign_{key[0]}_{key[1]}")
        for key in op
    }
    active = {
        key: m.addVars(samples, decisions, vtype=gp.GRB.BINARY, name=f"active_{key[0]}_{key[1]}")
        for key in op
    }
    discrete_active = (
        m.addVars(samples, decisions, vtype=gp.GRB.BINARY, name="discrete_active")
        if config.discrete_mode == "belong"
        else discrete_rho
    )
    for sample in range(samples):
        for node in decisions:
            for key, variables in op.items():
                m.addConstr(
                    rho[key][sample, node] <= config.big_m * sign[key][sample, node],
                    f"sign_upper_{key}_{sample}_{node}",
                )
                m.addConstr(
                    -rho[key][sample, node] <= config.big_m * (1 - sign[key][sample, node]),
                    f"sign_lower_{key}_{sample}_{node}",
                )
                m.addGenConstrMin(
                    active[key][sample, node],
                    [variables[node], sign[key][sample, node]],
                    name=f"active_{key}_{sample}_{node}",
                )
            if config.discrete_mode == "belong":
                m.addGenConstrMin(
                    discrete_active[sample, node],
                    [discrete[node], discrete_rho[sample, node]],
                    name=f"discrete_active_{sample}_{node}",
                )
            left_capacity = gp.quicksum(v[sample, node] for v in active.values())
            left_capacity += discrete_active[sample, node]
            m.addConstr(flow[sample, 2 * node + 1] <= left_capacity, f"left_{sample}_{node}")
            m.addConstr(flow[sample, 2 * node + 2] <= 1 - left_capacity, f"right_{sample}_{node}")
            m.addConstr(
                flow[sample, 2 * node + 2]
                <= gp.quicksum(v[node] for v in op.values()) + discrete[node],
                f"right_active_{sample}_{node}",
            )
    return sign, active, discrete_active


def build_model(signals, labels, config=None, *, env=None):
    """Build, but do not solve, a max-flow MILP.

    signals is N × F × T; its LAST channel must be binary. Use
    augment_with_discrete_feature for the supplied experiment's construction.
    Labels may be any nonnegative integers. A Gurobi license is required here.
    """
    import gurobipy as gp

    config = config or ModelConfig()
    x, y = validate_data(signals, labels)
    # Coefficients are frozen at build time. Keep the data used for extraction
    # and scoring equally stable if the caller later reuses its input buffers.
    x, y = x.copy(), y.copy()
    x.setflags(write=False)
    y.setflags(write=False)
    if x.shape[1] < 2:
        raise ValueError("Provide at least one continuous channel and a final binary channel")
    if not np.isin(x[:, -1, :], [0, 1]).all():
        raise ValueError("The final channel must be binary (0 or 1)")
    classes = [int(value) for value in np.unique(y)]
    count, decisions, leaves = generate_nodes(config.depth)
    windows = generate_windows(x.shape[2], config.max_window_span)
    m = gp.Model("STL_MaxFlow", env=env)
    try:
        op, discrete, label = _tree_block(m, gp, count, decisions, leaves, classes, x.shape[1] - 1)
        flow, sink, source = _flow_block(m, gp, y, count, decisions, leaves, label)
        threshold, window, rho, discrete_rho = _primitive_block(
            m,
            gp,
            x,
            config,
            decisions,
            windows,
            op,
            discrete,
        )
        sign, active, discrete_active = _routing_block(
            m,
            gp,
            len(y),
            decisions,
            config,
            op,
            discrete,
            rho,
            discrete_rho,
            flow,
        )
        # Source objective: maximize correctly labelled sample flow, no penalty.
        m.setObjective(sink.sum(), gp.GRB.MAXIMIZE)
        m.update()
    except Exception:
        m.dispose()
        raise
    return ModelArtifacts(
        m,
        config,
        x,
        y,
        classes,
        windows,
        decisions,
        leaves,
        {
            "op": op,
            "discrete": discrete,
            "label": label,
            "flow": flow,
            "sink": sink,
            "source": source,
            "threshold": threshold,
            "window": window,
            "rho": rho,
            "discrete_rho": discrete_rho,
            "sign": sign,
            "active": active,
            "discrete_active": discrete_active,
        },
    )
