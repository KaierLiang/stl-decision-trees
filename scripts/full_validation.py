"""Run complete source-vs-refactor comparisons on a remote lab, without a time cap.

Example: python scripts/full_validation.py --mode belong --output outputs/audit-belong
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from model_audit import (
    audit_solution,
    canonical_variables,
    compare_models,
    exported_assignment,
    original_artifacts,
)

from stl_tree import (
    ModelConfig,
    SolverConfig,
    augment_with_discrete_feature,
    build_model,
    load_dataset,
    sample_data,
    solve,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/naval_dataset.pkl"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--mode", choices=("exist", "belong"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument(
        "--tree-start", type=Path, help="Use a saved incumbent as a partial MIP start"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = load_dataset(args.dataset, trusted_pickle=True)
    x, y, selected = sample_data(*dataset.select("combined"), args.samples, args.seed)
    x, y, parents = augment_with_discrete_feature(x, y)
    np.savez_compressed(args.output / "indices.npz", selected=selected, parents=parents)
    config = ModelConfig(depth=args.depth, discrete_mode=args.mode)
    artifacts = build_model(x, y, config)
    original = original_artifacts(x, y, config)
    try:
        start = time.perf_counter()
        comparison = compare_models(original, artifacts)
        comparison.update(
            samples_before_augmentation=args.samples,
            samples_after_augmentation=len(y),
            depth=args.depth,
            windows=len(artifacts.windows),
            mode=args.mode,
            dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            comparison_seconds=time.perf_counter() - start,
        )
        (args.output / "model-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2), flush=True)
        if args.compare_only:
            return
        if args.tree_start:
            tree = json.loads(args.tree_start.read_text())
            v = artifacts.variables
            for key, node in tree["nodes"].items():
                n = int(key)
                if node["kind"] == "label":
                    v["label"][node["label"], n].Start = 1
                else:
                    if node["kind"] == "continuous":
                        v["op"][node["operation"], node["feature"]][n].Start = 1
                        v["threshold"][n].Start = node["threshold"]
                    else:
                        v["discrete"][n].Start = 1
                    if "window" in node:
                        v["window"][n][tuple(node["window"])].Start = 1
                    elif args.mode == "belong":
                        # A global-exist incumbent on constant binary channels
                        # can seed belong using any interval, without fixing it.
                        v["window"][n][0, 0].Start = 1
        solver = SolverConfig(threads=args.threads, seed=args.seed)
        result = solve(artifacts, solver, log_file=args.output / "refactored.log")
        result.save(args.output / "refactored", metadata={"comparison": comparison})
        first_audit = audit_solution(artifacts) if result.has_incumbent else {"passed": False}
        (args.output / "refactored-audit.json").write_text(json.dumps(first_audit, indent=2) + "\n")
        if not first_audit["passed"]:
            raise AssertionError(f"Refactored solution audit failed: {first_audit}")
        export_audit = audit_solution(
            artifacts, assignment=exported_assignment(artifacts, result.tree)
        )
        (args.output / "refactored-export-audit.json").write_text(
            json.dumps(export_audit, indent=2) + "\n"
        )
        assert export_audit["passed"]
        assert result.threshold_stabilization["verified_strict_routing"]
        assert abs(result.deterministic_training_accuracy * len(y) - result.objective) <= 1e-6
        artifacts.model.write(str(args.output / "refactored.sol"))
        # A full MIP start accelerates the independent source solve without fixing
        # any variables or changing its feasible set, objective, or proof bound.
        ref_vars = canonical_variables(artifacts)
        for name, variable in canonical_variables(original).items():
            variable.Start = ref_vars[name].X
        reference_result = solve(original, solver, log_file=args.output / "original.log")
        reference_result.save(args.output / "original", metadata={"warm_start_from_refactor": True})
        original_audit = audit_solution(original)
        (args.output / "original-audit.json").write_text(
            json.dumps(original_audit, indent=2) + "\n"
        )
        assert original_audit["passed"]
        original_export_audit = audit_solution(
            original, assignment=exported_assignment(original, reference_result.tree)
        )
        (args.output / "original-export-audit.json").write_text(
            json.dumps(original_export_audit, indent=2) + "\n"
        )
        assert original_export_audit["passed"]
        assert reference_result.threshold_stabilization["verified_strict_routing"]
        assert (
            abs(
                reference_result.deterministic_training_accuracy * len(y)
                - reference_result.objective
            )
            <= 1e-6
        )
        assert result.proven_optimal and reference_result.proven_optimal
        assert abs(result.objective - reference_result.objective) <= 1e-6
        print(
            "FULL COMPARISON PASSED: matching proven optima and independent constraint audits",
            flush=True,
        )
    finally:
        original.model.dispose()
        artifacts.model.dispose()


if __name__ == "__main__":
    main()
