from dataclasses import asdict

import numpy as np

from bangla_absa.config import TrainingConfig, tiny_config
from bangla_absa.data.splits import make_splits
from bangla_absa.experiments import execute
from bangla_absa.io import write_json
from bangla_absa.training.folds import run_cv
from bangla_absa.training.trainer import predict_run


def test_configuration_dispatch_five_seed_vote_and_ensemble(records, tmp_path):
    tc = TrainingConfig(epochs=1, batch_size=32, learning_rate=0.001)
    for name in ("m4", "m5", "m6", "m5_five_seed_vote"):
        write_json(
            tmp_path / "configs" / f"{name}.json",
            {"mode": "synthetic", "model": asdict(tiny_config(name)), "training": asdict(tc)},
        )
    manifest = make_splits(records)
    for name in ("m6", "m5_five_seed_vote"):
        run = tmp_path / name
        execute(records, manifest, tmp_path / "configs" / f"{name}.json", run)
        p, _ = predict_run(run, records[:2])
        assert p.shape == (2, 3, 3)
        np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)


def test_ensemble_outer_fold_execution(records, tmp_path):
    tc = TrainingConfig(epochs=1, batch_size=32, learning_rate=0.001)
    result = run_cv(
        records,
        make_splits(records),
        tiny_config("m6"),
        tc,
        tmp_path,
        mode="synthetic",
        folds=2,
        ensemble_configs=([tiny_config("m4"), tiny_config("m5")], [tc, tc]),
    )
    assert len(result) == 2 and all(r["metrics"]["decisions"] > 0 for r in result)
