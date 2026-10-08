from dataclasses import replace
from pathlib import Path

from bangla_absa.config import load_config
from bangla_absa.data.schema import labels_array
from bangla_absa.data.splits import select
from bangla_absa.evaluation.metrics import evaluate
from bangla_absa.evaluation.predictions import save_predictions
from bangla_absa.io import fingerprint, read_json, write_json
from bangla_absa.training.ensemble import soft_vote, train_ensemble
from bangla_absa.training.trainer import fit_run, predict_run


def execute(records, manifest, config_path, directory, resume=False):
    cfg, tc, raw = load_config(config_path)
    mode = raw.get("mode", "reconstruction")
    if cfg.name in {
        "m6",
        "equal_vote",
        "weighted_vote",
        "unweighted_stacking",
        "logistic_stacking",
    }:
        # All ensembles use the same leakage-controlled branch protocol.
        root = Path(config_path).parent
        branch = [load_config(root / f"{name}.json") for name in ("m4", "m5")]
        fusion = "stacking" if cfg.name == "m6" else cfg.name
        return train_ensemble(
            records,
            manifest,
            [b[0] for b in branch],
            [b[1] for b in branch],
            tc,
            directory,
            mode,
            fusion,
            resume,
        )
    if cfg.name == "m5_five_seed_vote":
        base, base_tc, _ = load_config(Path(config_path).parent / "m5.json")
        seeds = raw.get("provenance", {}).get("proposed_seeds", [17, 29, 43, 71, 101])
        if len(seeds) != 5 or len(set(seeds)) != 5:
            raise ValueError("M5 five-seed vote requires five distinct seeds")
        runs = []
        for seed in seeds:
            path = Path(directory) / f"seed-{seed}"
            fit_run(records, manifest, base, replace(base_tc, seed=seed), path, mode=mode)
            runs.append(path.name)
        meta = {
            "kind": "voting",
            "runs": runs,
            "manifest": manifest,
            "input_hash": fingerprint(records),
            "training": {"seed": tc.seed},
            "selection": "validation",
            "mode": mode,
        }
        meta["signature"] = fingerprint(meta)
        write_json(Path(directory) / "run.json", meta)
        from bangla_absa.training.checkpoints import seal

        seal(directory)
        return meta
    return fit_run(records, manifest, cfg, tc, directory, resume=resume, mode=mode)


def evaluate_run(directory, records, role, output):
    meta = read_json(Path(directory) / "run.json")
    if fingerprint(records) != meta["input_hash"]:
        raise ValueError("Evaluation data differ from original run input; use predict for new data")
    selection = meta.get(
        "selection", "meta_validation" if meta.get("kind") == "ensemble" else "validation"
    )
    partition = selection if role == "validation" else role
    if role not in {"validation", "test"}:
        raise ValueError("Evaluation role must be validation or test")
    rows = select(records, meta["manifest"], partition)
    if not rows:
        raise ValueError("Requested evaluation partition is empty")
    p, _ = predict_run(directory, rows)
    y = labels_array(rows)
    metrics = evaluate(y, p)
    out = Path(output)
    save_predictions(
        out / "predictions.json",
        [r["review_id"] for r in rows],
        y,
        p,
        partition,
        meta["training"]["seed"],
        meta["signature"],
    )
    write_json(out / "metrics.json", {"mode": meta["mode"], **metrics})
    return metrics


def predict_vote(directory, records, device="cpu"):
    from bangla_absa.training.checkpoints import verify

    verify(directory)
    meta = read_json(Path(directory) / "run.json")
    root = Path(directory).resolve()
    runs = [(root / name).resolve() for name in meta["runs"]]
    if any(not path.is_relative_to(root) for path in runs):
        raise ValueError("Voting run path outside checkpoint directory")
    return soft_vote([predict_run(path, records, device)[0] for path in runs], 5), meta
