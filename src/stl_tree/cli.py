"""Command-line entry points for inspection, conversion, and reproducible runs."""

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from .config import ModelConfig, SolverConfig
from .data import augment_with_discrete_feature, load_dataset, sample_data
from .model import build_model
from .solution import solve


def _parser():
    parser = argparse.ArgumentParser(description="Learn STL decision trees using a max-flow MILP")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "convert"):
        command = sub.add_parser(name, help=f"{name.capitalize()} a dataset without a solver")
        command.add_argument("dataset", type=Path)
        command.add_argument("--trust-pickle", action="store_true")
        if name == "convert":
            command.add_argument("output", type=Path, help="Destination .npz file (must not exist)")
    train = sub.add_parser("train", help="Build and solve an experiment")
    train.add_argument("--config", type=Path, help="JSON config; explicit CLI flags override it")
    train.add_argument("--dataset", type=Path)
    train.add_argument("--trust-pickle", action="store_true")
    train.add_argument("--split", choices=("train", "test", "combined"))
    train.add_argument("--samples", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument(
        "--augmentation",
        choices=("legacy", "none"),
        help="legacy: append/copy; none: input already has a final binary channel",
    )
    train.add_argument("--depth", type=int)
    train.add_argument("--max-window-span", type=int)
    train.add_argument("--discrete-mode", choices=("exist", "belong"))
    train.add_argument("--max-rho", type=float)
    train.add_argument("--big-m", type=float)
    train.add_argument("--time-limit", type=float)
    train.add_argument("--mip-gap", type=float)
    train.add_argument("--threads", type=int)
    train.add_argument("--quiet", action="store_true", help="Suppress solver progress")
    train.add_argument("--output", type=Path, help="New directory for this run")
    train.add_argument("--export-model", action="store_true", help="Write model.lp before solving")
    return parser


def _run_settings(args):
    settings = {
        "dataset": "data/naval_dataset.pkl",
        "split": "train",
        "samples": 100,
        "seed": 42,
        "augmentation": "legacy",
        "model": {},
        "solver": {},
    }
    if args.config:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        unknown = set(loaded) - set(settings)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        settings.update(loaded)
        # Paths written in config are relative to the config file, not the cwd.
        if "dataset" in loaded:
            settings["dataset"] = args.config.resolve().parent / loaded["dataset"]
    for key in ("dataset", "split", "samples", "seed", "augmentation"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if settings["augmentation"] not in {"legacy", "none"}:
        raise ValueError("augmentation must be 'legacy' or 'none'")
    for item in fields(ModelConfig):
        value = getattr(args, item.name)
        if value is not None:
            settings["model"][item.name] = value
    for key in ("time_limit", "mip_gap", "threads"):
        value = getattr(args, key)
        if value is not None:
            settings["solver"][key] = value
    settings["solver"].setdefault("seed", settings["seed"])
    if args.seed is not None:
        settings["solver"]["seed"] = args.seed
    if args.quiet:
        settings["solver"]["output_flag"] = False
    return settings, ModelConfig(**settings["model"]), SolverConfig(**settings["solver"])


def _train(args):
    settings, model_config, solver_config = _run_settings(args)
    dataset_path = Path(settings["dataset"])
    dataset = load_dataset(dataset_path, trusted_pickle=args.trust_pickle)
    x, y = dataset.select(settings["split"])
    x, y, indices = sample_data(x, y, settings["samples"], settings["seed"])
    if settings["augmentation"] == "legacy":
        x, y, parents = augment_with_discrete_feature(x, y)
    else:
        parents = np.arange(len(x))
    output = args.output or Path("outputs") / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "dataset": dataset_path.name,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "split": settings["split"],
        "augmentation": settings["augmentation"],
        "requested_samples": settings["samples"],
        "sampling_seed": settings["seed"],
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    np.savez_compressed(
        output / "sample_indices.npz",
        selected_indices=indices,
        augmented_parent_indices=parents,
        source_indices=indices[parents],
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "model": model_config.to_dict(),
                "solver": solver_config.to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Building {model_config.discrete_mode} model: {len(y)} augmented samples, "
        f"depth {model_config.depth}, {x.shape[2]} time steps",
        flush=True,
    )
    start = time.perf_counter()
    artifacts = build_model(x, y, model_config)
    metadata["build_seconds"] = time.perf_counter() - start
    try:
        import gurobipy as gp

        metadata["gurobi"] = ".".join(map(str, gp.gurobi.version()))
        if args.export_model:
            artifacts.model.write(str(output / "model.lp"))
        result = solve(artifacts, solver_config, log_file=output / "solver.log")
        result.save(output, metadata=metadata)
        print(json.dumps(result.to_dict(), indent=2))
        if result.tree:
            print(result.tree.format())
        print(f"Saved run to {output}")
        return 0 if result.has_incumbent else 2
    finally:
        artifacts.model.dispose()


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return _train(args)
        dataset = load_dataset(args.dataset, trusted_pickle=args.trust_pickle)
        if args.command == "inspect":
            print(json.dumps(dataset.summary(), indent=2))
        else:
            if args.output.suffix.lower() != ".npz":
                raise ValueError("Conversion output must end in .npz")
            # Exclusive creation prevents overwriting an existing dataset.
            with args.output.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    X_train=dataset.train_signals,
                    y_train=dataset.train_labels,
                    X_test=dataset.test_signals,
                    y_test=dataset.test_labels,
                )
            print(f"Saved {args.output}")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(
            f"Missing dependency: {exc}. Install with python -m pip install -e .", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
