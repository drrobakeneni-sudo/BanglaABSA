"""Command-line workflows; all private artifacts default to user-chosen local paths."""

import argparse
from dataclasses import replace
from pathlib import Path

from bangla_absa.config import TrainingConfig, load_config, tiny_config
from bangla_absa.data.importers import read_records, write_records
from bangla_absa.data.schema import labels_array
from bangla_absa.data.splits import deduplicate, make_splits, validate_splits
from bangla_absa.evaluation.metrics import evaluate
from bangla_absa.evaluation.predictions import load_predictions, save_predictions
from bangla_absa.experiments import evaluate_run, execute
from bangla_absa.io import file_hash, fingerprint, read_json, write_json
from bangla_absa.preprocessing.pipeline import Pipeline
from bangla_absa.training.trainer import fit_run, predict_run


def prepare(args):
    records = read_records(args.input)
    records, summary = Pipeline().transform(records, args.min_words, args.max_words, args.language)
    if args.deduplicate:
        records = deduplicate(records)
    manifest = read_json(args.manifest) if args.manifest else make_splits(records, seed=args.seed)
    validate_splits(records, manifest, ["train", "validation", "test"])
    out = Path(args.out)
    write_records(out / "reviews.jsonl", records)
    write_json(out / "splits.json", manifest)
    summary.update(
        {
            "input_sha256": file_hash(args.input),
            "split_hash": fingerprint(manifest),
            "after_deduplication": len(records),
            "length_unit": "whitespace words",
            "split_origin": "provided" if args.manifest else "reconstructed",
            "partitions": {s: list(manifest.values()).count(s) for s in set(manifest.values())},
        }
    )
    write_json(out / "preparation.json", summary)
    print(f"Prepared {len(records)} reviews; outputs saved locally.")


def smoke(out):
    from bangla_absa.synthetic import synthetic_records
    from bangla_absa.training.ensemble import train_ensemble

    records = synthetic_records()
    records, _ = Pipeline().transform(records)
    manifest = make_splits(records)
    out = Path(out)
    write_records(out / "prepared" / "reviews.jsonl", records)
    write_json(out / "prepared" / "splits.json", manifest)
    tc = TrainingConfig(epochs=2, batch_size=16, learning_rate=0.001, threads=2)
    fit_run(records, manifest, tiny_config("m4"), tc, out / "m4", mode="synthetic")
    train_ensemble(
        records,
        manifest,
        [tiny_config("m4"), tiny_config("m5")],
        [tc, tc],
        replace(tc, epochs=3),
        out / "m6",
        mode="synthetic",
    )
    report = {"purpose": "Synthetic CPU integration only; not scientific validation", "models": {}}
    for name in ("m4", "m6"):
        evaluate_run(out / name, records, "validation", out / "evaluation" / name / "validation")
        report["models"][name] = evaluate_run(
            out / name, records, "test", out / "evaluation" / name / "test"
        )
        # Verify predictions are independent of the presence of gold labels.
        unlabeled = [{"review_id": r["review_id"], "text": r["text"]} for r in records[:3]]
        p, meta = predict_run(out / name, unlabeled)
        save_predictions(
            out / "evaluation" / name / "unlabeled.json",
            [r["review_id"] for r in unlabeled],
            labels_array(unlabeled),
            p,
            "predict",
            tc.seed,
            meta["signature"],
        )
    write_json(out / "smoke-report.json", report)
    print("CPU synthetic M4 and held-out M6 integration completed; see smoke-report.json.")
    return report


def parser():
    root = argparse.ArgumentParser(prog="bangla-absa", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=256)
    p.add_argument(
        "--language", choices=["any", "bangla", "bangla_or_latin"], default="bangla_or_latin"
    )
    p.add_argument("--deduplicate", action="store_true")
    for name in ("train", "ensemble", "cv", "repeat", "suite", "ablations"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--manifest", required=True)
        p.add_argument("--config", required=name not in {"suite", "ablations"})
        p.add_argument(
            "--configs", default="configs/models" if name == "suite" else "configs/ablations"
        )
        p.add_argument("--out", required=True)
        p.add_argument("--resume", action="store_true")
        if name == "repeat":
            p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 71, 101])
    for name in ("validate", "test", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plots", action="store_true")
    p = sub.add_parser("export-tables")
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("benchmark")
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--out", default="runs/smoke")
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "smoke":
            smoke(args.out)
        elif args.command in {"train", "ensemble"}:
            execute(
                read_records(args.data),
                read_json(args.manifest),
                args.config,
                args.out,
                args.resume,
            )
            print("Training complete; saved checkpoint and provenance.")
        elif args.command in {"validate", "test"}:
            evaluate_run(
                args.run,
                read_records(args.data),
                "validation" if args.command == "validate" else "test",
                args.out,
            )
            print("Evaluation complete; local metrics and predictions saved.")
        elif args.command == "predict":
            records = read_records(args.data)
            p, meta = predict_run(args.run, records)
            save_predictions(
                args.out,
                [r["review_id"] for r in records],
                labels_array(records),
                p,
                "predict",
                meta["training"]["seed"],
                meta["signature"],
            )
        elif args.command == "evaluate":
            _, y, p, _ = load_predictions(args.predictions)
            write_json(Path(args.out) / "metrics.json", evaluate(y, p))
            if args.plots:
                from bangla_absa.evaluation.plots import plot_predictions

                plot_predictions(y, p, args.out)
        elif args.command == "export-tables":
            from bangla_absa.evaluation.tables import export_tables

            export_tables(args.predictions, args.out)
        elif args.command == "benchmark":
            from bangla_absa.evaluation.benchmark import benchmark
            from bangla_absa.training.trainer import load_run

            model, features, pipeline, _ = load_run(args.run)
            if features is None:
                raise ValueError("Parameter/torch benchmark supports individual neural models")
            records, _ = pipeline.transform(read_records(args.data)[:8])
            write_json(args.out, benchmark(model, features.encode(records)))
        elif args.command == "cv":
            from bangla_absa.training.folds import run_cv

            cfg, tc, raw = load_config(args.config)
            branches = None
            if cfg.name == "m6":
                branch = [
                    load_config(Path(args.config).parent / f"{name}.json") for name in ("m4", "m5")
                ]
                branches = ([b[0] for b in branch], [b[1] for b in branch])
            run_cv(
                read_records(args.data),
                read_json(args.manifest),
                cfg,
                tc,
                args.out,
                raw.get("mode", "reconstruction"),
                branches,
            )
        elif args.command in {"suite", "ablations"}:
            paths = sorted(Path(args.configs).glob("*.json"))
            if not paths:
                raise ValueError("No configurations found")
            for path in paths:
                execute(
                    read_records(args.data),
                    read_json(args.manifest),
                    path,
                    Path(args.out) / path.stem,
                )
        elif args.command == "repeat":
            raw = read_json(args.config)
            for seed in args.seeds:
                variant = {**raw, "training": {**raw.get("training", {}), "seed": seed}}
                config_path = Path(args.out) / "configs" / f"seed-{seed}.json"
                write_json(config_path, variant)
                # Ensemble branch configs are copied into each configuration directory.
                for name in ("m4", "m5"):
                    branch = Path(args.config).parent / f"{name}.json"
                    if branch.exists():
                        b = read_json(branch)
                        b["training"]["seed"] = seed
                        write_json(config_path.parent / f"{name}.json", b)
                execute(
                    read_records(args.data),
                    read_json(args.manifest),
                    config_path,
                    Path(args.out) / f"seed-{seed}",
                )
    except (ValueError, FileNotFoundError) as exc:
        root.error(str(exc))


if __name__ == "__main__":
    main()
