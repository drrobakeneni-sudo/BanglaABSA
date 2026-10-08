from dataclasses import asdict

import pytest

from bangla_absa.cli import main
from bangla_absa.config import TrainingConfig, tiny_config
from bangla_absa.data.importers import write_records
from bangla_absa.io import read_json, write_json


def test_documented_cli_flow(records, tmp_path):
    source = tmp_path / "source.jsonl"
    write_records(source, records)
    prepared, run = tmp_path / "prepared", tmp_path / "run"
    main(["prepare", "--input", str(source), "--out", str(prepared)])
    config = tmp_path / "m4.json"
    write_json(
        config,
        {
            "mode": "synthetic",
            "model": asdict(tiny_config()),
            "training": asdict(TrainingConfig(epochs=1, batch_size=32)),
        },
    )
    args = ["--data", str(prepared / "reviews.jsonl"), "--manifest", str(prepared / "splits.json")]
    main(["train", *args, "--config", str(config), "--out", str(run)])
    for command in ("validate", "test"):
        main(
            [
                command,
                "--data",
                str(prepared / "reviews.jsonl"),
                "--run",
                str(run),
                "--out",
                str(tmp_path / command),
            ]
        )
    predictions = tmp_path / "test" / "predictions.json"
    main(
        ["evaluate", "--predictions", str(predictions), "--out", str(tmp_path / "plots"), "--plots"]
    )
    main(["export-tables", "--predictions", str(predictions), "--out", str(tmp_path / "table.csv")])
    main(
        [
            "benchmark",
            "--data",
            str(source),
            "--run",
            str(run),
            "--out",
            str(tmp_path / "bench.json"),
        ]
    )
    main(
        [
            "predict",
            "--data",
            str(source),
            "--run",
            str(run),
            "--out",
            str(tmp_path / "predicted.json"),
        ]
    )
    assert read_json(tmp_path / "bench.json")["parameters"] > 0
    with pytest.raises(SystemExit) as error:
        main(["train", *args, "--config", str(config), "--out", str(run)])
    assert error.value.code == 2


def test_cli_suite_repeat_cv_and_smoke(tmp_path, records):
    data = tmp_path / "records.jsonl"
    write_records(data, records)
    from bangla_absa.data.splits import make_splits

    write_json(tmp_path / "split.json", make_splits(records))
    cfgdir = tmp_path / "configs"
    write_json(
        cfgdir / "m1.json",
        {
            "mode": "synthetic",
            "model": asdict(tiny_config("m1")),
            "training": asdict(TrainingConfig(epochs=1)),
        },
    )
    args = ["--data", str(data), "--manifest", str(tmp_path / "split.json")]
    main(["suite", *args, "--configs", str(cfgdir), "--out", str(tmp_path / "suite")])
    main(["ablations", *args, "--configs", str(cfgdir), "--out", str(tmp_path / "ablations")])
    main(
        [
            "repeat",
            *args,
            "--config",
            str(cfgdir / "m1.json"),
            "--seeds",
            "17",
            "29",
            "--out",
            str(tmp_path / "repeat"),
        ]
    )
    main(["cv", *args, "--config", str(cfgdir / "m1.json"), "--out", str(tmp_path / "cv")])
    main(["smoke", "--out", str(tmp_path / "smoke")])
    assert (tmp_path / "smoke" / "smoke-report.json").exists()
