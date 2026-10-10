"""Check an equality case outside the stated nonzero-robustness assumption."""

import argparse
import json
from pathlib import Path

import numpy as np
from model_audit import original_artifacts

from stl_tree import ModelConfig, SolverConfig, build_model, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # A logical counterexample for verification, not a synthetic benchmark.
    x = np.array([[[1, 1], [2, 2], [0, 0]], [[1, 1], [2, 2], [0, 0]]], dtype=float)
    y = np.array([0, 1])
    evidence = {
        "scope": "boundary check outside the intended nonzero-robustness assumption",
        "signals": x.tolist(),
        "labels": y.tolist(),
        "deterministic_classifier_upper_bound": 1,
        "runs": [],
    }
    for mode in ("exist", "belong"):
        config = ModelConfig(depth=1, discrete_mode=mode)
        for variant, builder in (("original", original_artifacts), ("refactored", build_model)):
            artifacts = builder(x, y, config)
            try:
                result = solve(artifacts, SolverConfig(output_flag=False))
                assert result.proven_optimal and abs(result.objective - 2) <= 1e-6
                left = result.tree.predict(x, tie_break="left")
                right = result.tree.predict(x, tie_break="right")
                evidence["runs"].append(
                    {
                        "mode": mode,
                        "variant": variant,
                        "status": result.status,
                        "flow_objective": result.objective,
                        "objective_bound": result.objective_bound,
                        "left_policy_predictions": left.tolist(),
                        "right_policy_predictions": right.tolist(),
                        "left_policy_correct": int(np.sum(left == y)),
                        "right_policy_correct": int(np.sum(right == y)),
                        "tree": result.tree.to_dict(),
                        "sample_left_flow": [
                            float(artifacts.variables["flow"][i, 1].X) for i in range(2)
                        ],
                        "sample_right_flow": [
                            float(artifacts.variables["flow"][i, 2].X) for i in range(2)
                        ],
                    }
                )
            finally:
                artifacts.model.dispose()
    (args.output / "equality-counterexample.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("CONFIRMED: all four models claim 2/2 flow for identical signals with opposite labels.")
    print("Every deterministic classifier can classify at most 1/2 of these two samples correctly.")


if __name__ == "__main__":
    main()
