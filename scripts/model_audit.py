"""Independent model comparison and numerical solution checks for remote validation.

Variable/row names and creation order are ignored. Every coefficient, bound,
objective coefficient, variable type, and MIN constraint is compared explicitly.
This module is development tooling, not a substitute optimization formulation.
"""

import hashlib
import json
import runpy
import struct
from collections import Counter
from pathlib import Path

from stl_tree.model import ModelArtifacts

ROOT = Path(__file__).resolve().parents[1]


def original_artifacts(x, y, config):
    reference = runpy.run_path(str(ROOT / "tests/test_solver_regression.py"))
    ns = reference["legacy_model"](
        x, y, config.discrete_mode, depth=config.depth, return_namespace=True
    )
    features = ns["features"]
    discrete_key = f"{config.discrete_mode}_{ns['discrete_features'][0]}"
    op = {
        (kind, j): ns["op_vars"][f"{kind}_{feature}"]
        for kind in ("less", "greater")
        for j, feature in enumerate(features)
    }
    variables = {
        "op": op,
        "discrete": ns["op_vars"][discrete_key],
        "label": {(k, n): v for k, values in ns["w"].items() for n, v in values.items()},
        "flow": ns["z"],
        "sink": ns["z_t"],
        "source": ns["z_s"],
        "threshold": ns["threshold"],
        "window": ns["time_window"],
        "discrete_rho": ns["rho_vars"][f"rho_{discrete_key}"],
    }
    for new, old, prefix in (
        ("rho", "rho_vars", "rho"),
        ("sign", "y_vars", "y"),
        ("active", "min_vars", "min"),
    ):
        variables[new] = {(kind, j): ns[old][f"{prefix}_{kind}_{features[j]}"] for kind, j in op}
    variables["discrete_active"] = (
        ns["min_vars_discrete"][f"min_{discrete_key}"]
        if config.discrete_mode == "belong"
        else variables["discrete_rho"]
    )
    ns["m"].update()
    return ModelArtifacts(
        ns["m"],
        config,
        x,
        y,
        sorted(ns["w"]),
        ns["comb"],
        ns["decision_nodes"],
        ns["leaf_nodes"],
        variables,
    )


def canonical_variables(artifacts):
    result = {}
    for group, collection in artifacts.variables.items():
        if group == "discrete_active" and artifacts.config.discrete_mode == "exist":
            continue  # It aliases discrete_rho, rather than creating new variables.
        if group in {"op", "rho", "sign", "active", "window"}:
            for key, values in collection.items():
                for index, variable in values.items():
                    result[f"{group}|{key}|{index}"] = variable
        else:
            for index, variable in collection.items():
                result[f"{group}|{index}"] = variable
    if len(result) != artifacts.model.NumVars:
        raise AssertionError("Canonical variable mapping does not cover the entire model")
    if len({v.index for v in result.values()}) != len(result):
        raise AssertionError("Duplicate variables in canonical mapping")
    return result


def model_signature(artifacts):
    import gurobipy as gp

    model = artifacts.model
    model.update()
    canonical = canonical_variables(artifacts)
    ordered = sorted(canonical)
    index_to_id = {canonical[name].index: i for i, name in enumerate(ordered)}
    attributes = {
        name: (var.VType, float(var.LB), float(var.UB), float(var.Obj))
        for name, var in canonical.items()
    }
    rows = Counter()
    nonzeros = 0
    for constraint in model.getConstrs():
        expression = model.getRow(constraint)
        coefficients = sorted(
            (index_to_id[expression.getVar(j).index], expression.getCoeff(j))
            for j in range(expression.size())
        )
        nonzeros += len(coefficients)
        digest = hashlib.sha256()
        digest.update(constraint.Sense.encode())
        digest.update(struct.pack("<d", float(constraint.RHS)))
        for index, coefficient in coefficients:
            digest.update(struct.pack("<qd", index, coefficient))
        rows[digest.hexdigest()] += 1
    general = Counter()
    for constraint in model.getGenConstrs():
        if constraint.GenConstrType != gp.GRB.GENCONSTR_MIN:
            raise AssertionError("Unexpected general constraint type")
        output, operands, constant = model.getGenConstrMin(constraint)
        general[
            (
                index_to_id[output.index],
                tuple(sorted(index_to_id[v.index] for v in operands)),
                float(constant),
            )
        ] += 1
    return {
        "variables": attributes,
        "linear_rows": rows,
        "general_constraints": general,
        "model_sense": model.ModelSense,
        "objective_constant": float(model.ObjCon),
        "nonzeros": nonzeros,
    }


def compare_models(left, right):
    signatures = [model_signature(item) for item in (left, right)]
    checks = {key: signatures[0][key] == signatures[1][key] for key in signatures[0]}
    if not all(checks.values()):
        raise AssertionError(f"Source/refactor differ: {checks}")
    row_fingerprint = hashlib.sha256(
        json.dumps(sorted(signatures[0]["linear_rows"].items())).encode()
    ).hexdigest()
    return {
        "identical": True,
        "checks": checks,
        "variables": left.model.NumVars,
        "linear_constraints": left.model.NumConstrs,
        "general_constraints": left.model.NumGenConstrs,
        "nonzeros": signatures[0]["nonzeros"],
        "linear_row_sha256": row_fingerprint,
    }


def exported_assignment(artifacts, tree):
    """Reconstruct auxiliaries for an exported tree in the ORIGINAL formulation."""
    values = artifacts.model.getAttr("X", artifacts.model.getVars())
    v = artifacts.variables
    for node, predicate in tree.nodes.items():
        if predicate["kind"] != "continuous":
            continue
        threshold = predicate["threshold"]
        values[v["threshold"][node].index] = threshold
        a, b = predicate["window"]
        for sample, signal in enumerate(artifacts.signals):
            for key, operation in v["op"].items():
                kind, feature = key
                window = signal[feature, a : b + 1]
                rho = threshold - window.min() if kind == "less" else window.max() - threshold
                sign = float(rho > 0)
                values[v["rho"][key][sample, node].index] = float(rho)
                values[v["sign"][key][sample, node].index] = sign
                values[v["active"][key][sample, node].index] = min(operation[node].X, sign)
    return values


def audit_solution(artifacts, *, tolerance=1e-5, assignment=None):
    """Re-evaluate every constraint and each routed predicate independently."""
    import gurobipy as gp

    model = artifacts.model
    if not model.SolCount:
        raise ValueError("No incumbent to audit")
    values = model.getAttr("X", model.getVars()) if assignment is None else assignment

    def get_value(variable):
        return values[variable.index]

    bound_error = integer_error = row_error = min_error = 0.0
    for variable, value in zip(model.getVars(), values):
        bound_error = max(bound_error, variable.LB - value, value - variable.UB)
        if variable.VType != gp.GRB.CONTINUOUS:
            integer_error = max(integer_error, abs(value - round(value)))
    for constraint in model.getConstrs():
        row = model.getRow(constraint)
        lhs = sum(row.getCoeff(j) * values[row.getVar(j).index] for j in range(row.size()))
        residual = lhs - constraint.RHS
        violation = (
            abs(residual)
            if constraint.Sense == "="
            else residual
            if constraint.Sense == "<"
            else -residual
        )
        row_error = max(row_error, violation)
    for constraint in model.getGenConstrs():
        output, operands, constant = model.getGenConstrMin(constraint)
        target = min([constant, *(values[v.index] for v in operands)])
        min_error = max(min_error, abs(values[output.index] - target))
    objective = model.ObjCon + sum(v.Obj * values[v.index] for v in model.getVars())
    objective_error = abs(objective - model.ObjVal)

    v = artifacts.variables
    routed = 0
    equality_routed = 0
    bad_routes = []
    for sample, (signal, target) in enumerate(zip(artifacts.signals, artifacts.labels)):
        if get_value(v["source"][sample]) < 0.5:
            continue
        routed += 1
        node = 0
        for _ in range(artifacts.config.depth + 1):
            if get_value(v["sink"][sample, node]) > 0.5:
                if get_value(v["label"][int(target), node]) <= 0.5:
                    bad_routes.append([sample, node, "incorrect_sink"])
                break
            left = get_value(v["flow"][sample, 2 * node + 1]) > 0.5
            if get_value(v["discrete"][node]) > 0.5:
                if artifacts.config.discrete_mode == "exist":
                    truth = signal[-1].max()
                else:
                    a, b = next(
                        w for w in artifacts.windows if get_value(v["window"][node][w]) > 0.5
                    )
                    window = signal[-1, a : b + 1]
                    if window.min() != window.max():
                        bad_routes.append([sample, node, "nonconstant_binary_interval"])
                    truth = window.max()
                if left != bool(truth):
                    bad_routes.append([sample, node, "wrong_discrete_branch"])
            else:
                kind, feature = next(
                    key for key, op in v["op"].items() if get_value(op[node]) > 0.5
                )
                a, b = next(w for w in artifacts.windows if get_value(v["window"][node][w]) > 0.5)
                window = signal[feature, a : b + 1]
                threshold = get_value(v["threshold"][node])
                rho = threshold - window.min() if kind == "less" else window.max() - threshold
                if abs(rho) <= tolerance:
                    equality_routed += 1
                elif left != (rho > 0):
                    bad_routes.append([sample, node, "wrong_continuous_branch"])
            node = 2 * node + (1 if left else 2)
        else:
            bad_routes.append([sample, node, "missing_sink"])
    report = {
        "bound_max_violation": float(bound_error),
        "integrality_max_violation": float(integer_error),
        "linear_max_violation": float(row_error),
        "min_max_violation": float(min_error),
        "objective_error": float(objective_error),
        "routed_samples": routed,
        "zero_robustness_routed_decisions": equality_routed,
        "bad_routes": bad_routes,
        "tolerance": tolerance,
    }
    report["passed"] = (
        max(bound_error, integer_error, row_error, min_error, objective_error) <= tolerance
        and not bad_routes
        and abs(routed - model.ObjVal) <= tolerance
    )
    return report
