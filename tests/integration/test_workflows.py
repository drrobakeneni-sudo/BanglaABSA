from dataclasses import replace

import numpy as np
import pytest

from bangla_absa.config import TrainingConfig, tiny_config
from bangla_absa.data.schema import labels_array
from bangla_absa.data.splits import make_splits, select
from bangla_absa.evaluation.benchmark import benchmark
from bangla_absa.evaluation.plots import plot_history, plot_predictions
from bangla_absa.experiments import evaluate_run
from bangla_absa.io import read_json
from bangla_absa.training.checkpoints import verify
from bangla_absa.training.ensemble import (
    assert_held_out,
    ensemble_manifest,
    fit_weights,
    prediction_cache,
    soft_vote,
    train_ensemble,
)
from bangla_absa.training.folds import development_folds, run_cv
from bangla_absa.training.trainer import fit_run, load_run, predict_run


def tc():
    return TrainingConfig(epochs=1, batch_size=32, learning_rate=0.001)


def test_neural_roundtrip_resume_and_unlabeled(records, tmp_path):
    manifest = make_splits(records)
    run = tmp_path / "run"
    cfg = tiny_config("m4")
    fit_run(records, manifest, cfg, tc(), run, mode="synthetic")
    p1, _ = predict_run(run, records[:3])
    unlabeled = [{"review_id": r["review_id"], "text": r["text"]} for r in records[:3]]
    p2, _ = predict_run(run, unlabeled)
    np.testing.assert_allclose(p1, p2)
    fit_run(records, manifest, cfg, tc(), run, mode="synthetic", resume=True)
    p3, _ = predict_run(run, unlabeled)
    np.testing.assert_allclose(p1, p3)
    evaluate_run(run, records, "test", tmp_path / "evaluation")
    with pytest.raises(ValueError, match="mismatch"):
        fit_run(records, manifest, cfg, replace(tc(), seed=99), run, mode="synthetic", resume=True)
    model, features, pipeline, _ = load_run(run)
    data, _ = pipeline.transform(records[:2])
    result = benchmark(model, features.encode(data), warmup=1, repetitions=2)
    assert result["parameters"] > 0 and result["flops"] is None
    plot_predictions(labels_array(records[:3]), p1, tmp_path / "figures")
    plot_history(read_json(run / "history.json"), tmp_path / "history.png")
    assert (tmp_path / "figures" / "evaluation.svg").exists()
    (run / "preprocessing.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify(run)


def test_ensemble_isolated_partitions_cache_and_resume(records, tmp_path):
    manifest = make_splits(records)
    combined = ensemble_manifest(records, manifest)
    assert set(combined.values()) == {
        "train",
        "weight_fit",
        "meta_train",
        "meta_validation",
        "test",
    }
    cfgs = [tiny_config("m4"), tiny_config("m5")]
    run = tmp_path / "m6"
    meta = train_ensemble(records, manifest, cfgs, [tc(), tc()], tc(), run, mode="synthetic")
    for branch in ("m4", "m5"):
        bmeta = read_json(run / branch / "run.json")
        assert not set(bmeta["train_ids"]) & set(meta["train_ids"])
        assert set(bmeta["selection_ids"]) == set(meta["selection_ids"])
    p, _ = predict_run(run, records[:2])
    assert p.shape == (2, 3, 3)
    train_ensemble(records, manifest, cfgs, [tc(), tc()], tc(), run, mode="synthetic", resume=True)
    np.testing.assert_allclose(p, predict_run(run, records[:2])[0])
    wf = select(records, combined, "weight_fit")
    cache = run / "cache" / "weight_fit-m4.npz"
    prediction_cache(cache, wf, run / "m4")
    with pytest.raises(ValueError, match="cache"):
        prediction_cache(cache, list(reversed(wf)), run / "m4")
    with pytest.raises(ValueError, match="leakage"):
        assert_held_out([{"train_ids": ["x"], "selection_ids": []}], ["x"])
    with pytest.raises(ValueError):
        fit_weights(p, p, np.zeros((2, 3), dtype=int), role="test")


@pytest.mark.parametrize(
    "fusion", ["equal_vote", "weighted_vote", "unweighted_stacking", "logistic_stacking"]
)
def test_ensemble_controls(fusion, records, tmp_path):
    manifest = make_splits(records)
    train_ensemble(
        records,
        manifest,
        [tiny_config("m4"), tiny_config("m5")],
        [tc(), tc()],
        tc(),
        tmp_path / fusion,
        mode="synthetic",
        fusion=fusion,
    )
    p, _ = predict_run(tmp_path / fusion, records[:2])
    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)
    np.testing.assert_allclose(soft_vote([p] * 5, 5), p)
    with pytest.raises(ValueError):
        soft_vote([p] * 2, 5)


def test_cv_excludes_original_test(records, tmp_path):
    manifest = make_splits(records)
    original_test = {r["review_id"] for r in select(records, manifest, "test")}
    for dev, split in development_folds(records, manifest):
        assert not original_test & set(split)
        assert {r["review_id"] for r in dev} == set(split)
    output = run_cv(records, manifest, tiny_config("m1"), tc(), tmp_path, mode="synthetic", folds=2)
    assert len(output) == 2
