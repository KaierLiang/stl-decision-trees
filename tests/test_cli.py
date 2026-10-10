"""Configuration and I/O behavior that can silently change an experiment."""

import json
from pathlib import Path

from stl_tree.cli import _parser, _run_settings, main

ROOT = Path(__file__).resolve().parents[1]


def test_config_paths_and_explicit_overrides_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _parser().parse_args(
        [
            "train",
            "--config",
            str(ROOT / "configs/naval.json"),
            "--depth",
            "2",
            "--seed",
            "17",
            "--discrete-mode",
            "exist",
            "--time-limit",
            "10",
        ]
    )
    settings, model, solver = _run_settings(args)
    assert Path(settings["dataset"]).resolve() == ROOT / "data/naval_dataset.pkl"
    assert settings["split"] == "combined" and settings["samples"] == 100
    assert settings["augmentation"] == "discrete"
    assert model.depth == 2 and model.discrete_mode == "exist"
    assert model.max_window_span == 500
    assert solver.time_limit == 10 and solver.seed == settings["seed"] == 17


def test_unknown_config_key_is_not_silently_ignored(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"sampless": 20}), encoding="utf-8")
    assert main(["train", "--config", str(path)]) == 1
    assert "Unknown config keys" in capsys.readouterr().err


def test_convert_and_inspect_cli(tmp_path, capsys):
    path = tmp_path / "naval.npz"
    assert main(["convert", str(ROOT / "data/naval_dataset.pkl"), str(path), "--trust-pickle"]) == 0
    capsys.readouterr()
    assert main(["inspect", str(path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["train"]["shape"] == [1600, 2, 61]
    assert main(["convert", str(ROOT / "data/naval_dataset.pkl"), str(path), "--trust-pickle"]) == 1
