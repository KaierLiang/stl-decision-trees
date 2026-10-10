"""Select numerically stable thresholds without changing the fitted tree's flow.

This does not add margins to the MILP. It selects another threshold within the
existing feasible interval, only when a routed robustness is near zero. If no
strict interval exists, the boundary is reported and the raw threshold retained.
"""

from copy import deepcopy

import numpy as np


def stabilize_thresholds(artifacts, tree, *, tolerance=1e-6):
    """Return a separate export tree and a report; leave the solver untouched.

    The interval accounts for every sample/continuous feature's robustness bounds,
    the threshold variable's bounds, and every nonzero sample flow at this node.
    Operations, windows, labels, topology, data, and admitted flows are unchanged.
    This assumes the base formulation built by this package; caller-added threshold
    constraints are not supported by this postprocessing step.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    exported = deepcopy(tree)
    changes, unresolved = [], []
    v = artifacts.variables
    for node_id, node in exported.nodes.items():
        if node["kind"] != "continuous":
            continue
        a, b = node["window"]
        signals = artifacts.signals[:, :-1, a : b + 1]
        minima, maxima = signals.min(axis=2), signals.max(axis=2)
        less = node["operation"] == "less"
        extrema = (minima if less else maxima)[:, node["feature"]]
        left = np.array([v["flow"][i, 2 * node_id + 1].X > 0.5 for i in range(len(signals))])
        right = np.array([v["flow"][i, 2 * node_id + 2].X > 0.5 for i in range(len(signals))])
        routed = left | right
        rho = (node["threshold"] - extrema) * (1 if less else -1)
        if not np.any(routed & (np.abs(rho) <= tolerance)):
            continue
        variable = v["threshold"][node_id]
        # Both operations exist for every continuous feature, even if unselected.
        lower = max(float(variable.LB), float(maxima.max()) - artifacts.config.max_rho)
        upper = min(float(variable.UB), float(minima.min()) + artifacts.config.max_rho)
        below, above = (left, right) if less else (right, left)
        if below.any():
            lower = max(lower, float(extrema[below].max()))
        if above.any():
            upper = min(upper, float(extrema[above].min()))
        candidate = lower + (upper - lower) / 2
        candidate_rho = (candidate - extrema) * (1 if less else -1)
        stable = (
            lower < candidate < upper
            and np.all(candidate_rho[left] > tolerance)
            and np.all(candidate_rho[right] < -tolerance)
        )
        if not stable:
            unresolved.append(
                {
                    "node": node_id,
                    "lower": lower,
                    "upper": upper,
                    "reason": "No threshold with stable strict routing in the feasible interval",
                }
            )
            continue
        changes.append(
            {
                "node": node_id,
                "raw_threshold": node["threshold"],
                "export_threshold": candidate,
                "feasible_lower": lower,
                "feasible_upper": upper,
                "minimum_routed_absolute_robustness": float(np.abs(candidate_rho[routed]).min()),
            }
        )
        node["threshold"] = candidate

    # Verify each admitted flow's branch, not just its final predicted label.
    minimum = None
    mismatches = 0
    near_zero = 0
    for node_id, node in exported.nodes.items():
        if node["kind"] != "continuous":
            continue
        a, b = node["window"]
        values = artifacts.signals[:, node["feature"], a : b + 1]
        rho = (
            node["threshold"] - values.min(axis=1)
            if node["operation"] == "less"
            else values.max(axis=1) - node["threshold"]
        )
        for i, value in enumerate(rho):
            left = v["flow"][i, 2 * node_id + 1].X > 0.5
            right = v["flow"][i, 2 * node_id + 2].X > 0.5
            if not (left or right):
                continue
            magnitude = abs(float(value))
            minimum = magnitude if minimum is None else min(minimum, magnitude)
            near_zero += int(magnitude <= tolerance)
            mismatches += int(left != (value > 0))
    return exported, {
        "method": "interior_thresholds_preserving_incumbent_flow",
        "tolerance": tolerance,
        "changed_nodes": changes,
        "unresolved_nodes": unresolved,
        "routed_branch_mismatches": mismatches,
        "near_zero_routed_decisions": near_zero,
        "minimum_routed_absolute_robustness": minimum,
        "verified_strict_routing": mismatches == 0 and near_zero == 0 and not unresolved,
    }
