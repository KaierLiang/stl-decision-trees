"""The complete workflow as separate, reusable Python blocks."""

from datetime import datetime
from pathlib import Path

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
    # Block 1: load and sample only the training split.
    root = Path(__file__).resolve().parents[1]
    dataset = load_dataset(root / "data/naval_dataset.pkl", trusted_pickle=True)
    x, y, indices = sample_data(*dataset.select("train"), samples=12, seed=42)

    # Block 2: apply the original discrete-channel experiment explicitly.
    x, y, parents = augment_with_discrete_feature(x, y)

    # Block 3: build the model; all 1,891 windows of the 61-step data remain.
    artifacts = build_model(x, y, ModelConfig(depth=1, discrete_mode="exist"))
    try:
        # Block 4: solve with a finite budget and inspect the incumbent/status.
        result = solve(artifacts, SolverConfig(time_limit=30, threads=1))
        print(result.to_dict())
        if result.tree:
            print(result.tree.format())
            # Block 5: predict with an explicit policy for zero-robustness ties.
            predictions, ties = result.tree.predict(x, tie_break="left", return_ties=True)
            print("Deterministic training accuracy:", (predictions == y).mean())
            print("Samples encountering a tie:", ties.sum())
        # Block 6: persist the tree and the original source index of each copy.
        output = root / "outputs" / ("python-api-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        result.save(
            output,
            metadata={
                "split": "train",
                "augmentation": "discrete",
                "source_indices": indices[parents].tolist(),
            },
        )
    finally:
        artifacts.model.dispose()


if __name__ == "__main__":
    main()
