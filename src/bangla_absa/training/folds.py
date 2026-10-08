from pathlib import Path

from bangla_absa.data.schema import labels_array
from bangla_absa.data.splits import make_splits, select, validate_splits
from bangla_absa.evaluation.metrics import evaluate
from bangla_absa.io import write_json
from bangla_absa.training.trainer import fit_run, predict_run


def development_folds(records, manifest, folds=5, seed=17):
    validate_splits(records, manifest, ["train", "validation", "test"])
    if folds < 2:
        raise ValueError("At least two outer folds required")
    development = [r for r in records if manifest[r["review_id"]] != "test"]
    partition = make_splits(development, {str(i): 1 / folds for i in range(folds)}, seed)
    output = []
    for fold in range(folds):
        outer_train = [r for r in development if partition[r["review_id"]] != str(fold)]
        inner = make_splits(outer_train, {"train": 0.8, "validation": 0.2}, seed + fold)
        combined = {r["review_id"]: "outer_test" for r in development}
        combined.update(inner)
        validate_splits(development, combined)
        output.append((development, combined))
    return output


def run_cv(
    records,
    manifest,
    model_cfg,
    train_cfg,
    directory,
    mode="reconstruction",
    ensemble_configs=None,
    folds=5,
):
    output = []
    for i, (development, split) in enumerate(
        development_folds(records, manifest, folds, train_cfg.seed)
    ):
        # Outer evaluation records never enter the inner fit call, including signatures/resources.
        fit_records = [r for r in development if split[r["review_id"]] != "outer_test"]
        fit_manifest = {r["review_id"]: split[r["review_id"]] for r in fit_records}
        run = Path(directory) / f"fold-{i}"
        if ensemble_configs is not None:
            from bangla_absa.training.ensemble import train_ensemble

            configs, trainings = ensemble_configs
            train_ensemble(fit_records, fit_manifest, configs, trainings, train_cfg, run, mode)
        else:
            fit_run(fit_records, fit_manifest, model_cfg, train_cfg, run, mode=mode)
        test = select(development, split, "outer_test")
        p, _ = predict_run(run, test)
        metrics = evaluate(labels_array(test), p)
        output.append({"fold": i, "metrics": metrics})
    write_json(Path(directory) / "cv.json", output)
    return output
