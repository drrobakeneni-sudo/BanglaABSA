from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from bangla_absa.config import ModelConfig
from bangla_absa.data.schema import labels_array
from bangla_absa.data.splits import make_splits, select, validate_splits
from bangla_absa.io import fingerprint, read_json, write_json
from bangla_absa.models.m6_ensemble import Stacker, weighted_features
from bangla_absa.training.checkpoints import load_weights, seal, verify
from bangla_absa.training.seeds import seed_everything
from bangla_absa.training.trainer import fit_run, predict_run, train_neural


def ensemble_manifest(records, manifest, seed=17):
    validate_splits(records, manifest, allowed=["train", "validation", "test"])
    validation = select(records, manifest, "validation")
    inner = make_splits(
        validation, {"weight_fit": 0.25, "meta_train": 0.5, "meta_validation": 0.25}, seed
    )
    combined = {**manifest, **inner}
    validate_splits(records, combined)
    return combined


def fit_weights(p4, p5, labels, role="weight_fit"):
    if role != "weight_fit":
        raise ValueError("Fusion weights require designated weight_fit data")
    from bangla_absa.evaluation.metrics import validate_predictions

    labels, p4 = validate_predictions(labels, p4)
    _, p5 = validate_predictions(labels, p5)
    weights = []
    for a in range(3):
        valid = labels[:, a] >= 0
        if not valid.any():
            raise ValueError("Each aspect needs labels in the weight-fitting partition")
        y = labels[valid, a]
        indices = np.arange(len(y))

        def objective(weight):
            p = weight * p4[valid, a] + (1 - weight) * p5[valid, a]
            return float(-np.log(p[indices, y].clip(1e-12)).mean())

        optimum = minimize_scalar(objective, bounds=(0, 1), method="bounded")
        candidates = [0.0, float(optimum.x), 1.0]
        w = min(candidates, key=objective)
        weights.append([w, 1 - w])
    return np.asarray(weights, dtype=np.float32)


def assert_held_out(branch_metadata, ids):
    ids = set(ids)
    for meta in branch_metadata:
        if ids & (set(meta["train_ids"]) | set(meta["selection_ids"])):
            raise ValueError("In-sample or selection-set leakage in weight/meta training")


def prediction_cache(path, records, branch_directory):
    path = Path(path)
    meta = read_json(Path(branch_directory) / "run.json")
    identity = {
        "input_hash": fingerprint(records),
        "run_signature": meta["signature"],
        "checkpoint": read_json(Path(branch_directory) / "checksums.json"),
        "ids": [r["review_id"] for r in records],
    }
    if path.exists():
        if read_json(path.with_suffix(".json")) != identity:
            raise ValueError("Stale or misaligned prediction cache")
        with np.load(path, allow_pickle=False) as saved:
            p = saved["probabilities"]
        from bangla_absa.evaluation.metrics import validate_predictions

        validate_predictions(labels_array(records), p)
        return p
    p, _ = predict_run(branch_directory, records)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=p)
    write_json(path.with_suffix(".json"), identity)
    return p


class TensorFeatures:
    def __init__(self, records, features):
        self.features = {row["review_id"]: value for row, value in zip(records, features)}

    def encode(self, records):
        return {"features": torch.stack([self.features[row["review_id"]] for row in records])}


def train_ensemble(
    records,
    manifest,
    branch_configs,
    branch_training,
    training,
    directory,
    mode="reconstruction",
    fusion="stacking",
    resume=False,
):
    if fusion not in {
        "stacking",
        "equal_vote",
        "weighted_vote",
        "unweighted_stacking",
        "logistic_stacking",
    }:
        raise ValueError("Unknown ensemble fusion")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise ValueError("Ensemble output must be a new directory")
    directory.mkdir(parents=True, exist_ok=True)
    combined = ensemble_manifest(records, manifest, training.seed)
    roles = {
        role: select(records, combined, role)
        for role in ("weight_fit", "meta_train", "meta_validation")
    }
    branch_meta = []
    for name, cfg, tc in zip(("m4", "m5"), branch_configs, branch_training):
        if cfg.name != name:
            raise ValueError("Ensemble branches must be M4 then M5")
        path = directory / name
        if resume and (path / "run.json").exists():
            verify(path)
            saved = read_json(path / "run.json")
            if (
                saved["model"] != asdict(cfg)
                or saved["training"] != asdict(tc)
                or saved["input_hash"] != fingerprint(records)
                or saved["split_hash"] != fingerprint(combined)
            ):
                raise ValueError("Ensemble branch resume mismatch")
            branch_meta.append(saved)
        else:
            branch_meta.append(
                fit_run(records, combined, cfg, tc, path, "meta_validation", mode=mode)
            )
    for role in ("weight_fit", "meta_train"):
        assert_held_out(branch_meta, [r["review_id"] for r in roles[role]])
    probabilities = {
        role: [
            prediction_cache(directory / "cache" / f"{role}-{name}.npz", rows, directory / name)
            for name in ("m4", "m5")
        ]
        for role, rows in roles.items()
    }
    weights = np.full((3, 2), 0.5, dtype=np.float32)
    if fusion not in {"equal_vote", "unweighted_stacking", "logistic_stacking"}:
        weights = fit_weights(*probabilities["weight_fit"], labels_array(roles["weight_fit"]))
    metadata = {
        "kind": "ensemble",
        "fusion": fusion,
        "mode": mode,
        "weights": weights.tolist(),
        "training": asdict(training),
        "input_hash": fingerprint(records),
        "manifest": combined,
        "branch_signatures": [m["signature"] for m in branch_meta],
        "protocol": "validation split 25% weights, 50% meta train, 25% branch/meta selection",
        "train_ids": [r["review_id"] for r in roles["meta_train"]],
        "selection_ids": [r["review_id"] for r in roles["meta_validation"]],
    }
    metadata["signature"] = fingerprint(metadata)
    if (
        resume
        and (directory / "run.json").exists()
        and read_json(directory / "run.json")["signature"] != metadata["signature"]
    ):
        raise ValueError("Ensemble resume mismatch")
    all_meta = roles["meta_train"] + roles["meta_validation"]
    feature_parts = []
    for role in ("meta_train", "meta_validation"):
        p4, p5 = [torch.as_tensor(p, dtype=torch.float32) for p in probabilities[role]]
        feature_parts.append(weighted_features(p4, p5, torch.from_numpy(weights)))
    features = torch.cat(feature_parts)
    seed_everything(training.seed, training.threads)
    if fusion == "logistic_stacking":
        model = LogisticStacker().fit(
            features[: len(roles["meta_train"])].numpy(),
            labels_array(roles["meta_train"]),
            role="meta_train",
        )
        import skops.io as sio

        sio.dump(model.estimators, directory / "logistic.skops")
    elif "stacking" in fusion:
        model = Stacker()
        model.cfg = ModelConfig(name="m6")
        train_neural(
            model,
            TensorFeatures(all_meta, features),
            roles["meta_train"],
            roles["meta_validation"],
            training,
            directory,
            metadata["signature"],
            resume=resume and (directory / "last.pt").exists(),
        )
    write_json(directory / "run.json", metadata)
    write_json(
        directory / "partition_counts.json", {role: len(rows) for role, rows in roles.items()}
    )
    seal(directory)
    return metadata


class LogisticStacker:
    def fit(self, features, labels, role="meta_train"):
        if role != "meta_train":
            raise ValueError("Logistic stacker requires held-out meta_train predictions")
        if features.ndim != 2 or features.shape[1] != 18:
            raise ValueError("Exactly 18 branch probability features required")
        from sklearn.dummy import DummyClassifier

        self.estimators = []
        for a in range(3):
            valid = labels[:, a] >= 0
            classes = np.unique(labels[valid, a])
            if not len(classes):
                raise ValueError("Every aspect needs meta-training labels")
            model = (
                DummyClassifier(strategy="constant", constant=classes[0])
                if len(classes) == 1
                else (LogisticRegression(C=1, max_iter=1000))
            )
            self.estimators.append(model.fit(features[valid], labels[valid, a]))
        return self

    def predict_proba(self, features):
        p = np.zeros((len(features), 3, 3))
        for a, estimator in enumerate(self.estimators):
            p[:, a, estimator.classes_.astype(int)] = estimator.predict_proba(features)
        return p


def soft_vote(probabilities, expected_branches=None):
    if expected_branches is not None and len(probabilities) != expected_branches:
        raise ValueError("Unexpected number of voting branches")
    if len(probabilities) < 2 or len({p.shape for p in probabilities}) != 1:
        raise ValueError("Voting requires aligned probability arrays")
    from bangla_absa.evaluation.metrics import validate_predictions

    for p in probabilities:
        validate_predictions(np.full(p.shape[:2], -1), p)
    return np.mean(probabilities, axis=0)


def predict_ensemble(directory, records, device="cpu"):
    directory = Path(directory)
    verify(directory)
    meta = read_json(directory / "run.json")
    p4, p5 = [predict_run(directory / name, records, device)[0] for name in ("m4", "m5")]
    weights = torch.tensor(meta["weights"], dtype=torch.float32)
    features = weighted_features(
        torch.from_numpy(p4).float(), torch.from_numpy(p5).float(), weights
    )
    if meta["fusion"] == "logistic_stacking":
        import skops.io as sio

        model = LogisticStacker()
        model.estimators = sio.load(directory / "logistic.skops", trusted=[])
        p = model.predict_proba(features.numpy())
    elif "stacking" in meta["fusion"]:
        model = Stacker().eval().to(device)
        load_weights(model, directory / "best.safetensors")
        with torch.inference_mode():
            p = np.concatenate(
                [
                    model(part.to(device)).softmax(-1).cpu().numpy()
                    for part in features.split(meta["training"]["batch_size"])
                ]
            )
    else:
        w = weights.numpy()
        p = p4 * w[:, 0, None] + p5 * w[:, 1, None]
    return p, meta
