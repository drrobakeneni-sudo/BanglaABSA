import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from bangla_absa.config import ModelConfig, strict_dataclass
from bangla_absa.data.schema import labels_array
from bangla_absa.data.splits import select, validate_splits
from bangla_absa.evaluation.metrics import evaluate
from bangla_absa.io import fingerprint, read_json, write_json
from bangla_absa.models.baselines.fasttext_bilstm import load_vectors
from bangla_absa.models.common import make_encoder, masked_loss
from bangla_absa.models.registry import CLASSICAL, build_model
from bangla_absa.preprocessing.pipeline import Pipeline
from bangla_absa.preprocessing.tokenization import Features
from bangla_absa.training.checkpoints import environment, load_weights, save_weights, seal, verify
from bangla_absa.training.seeds import restore_rng, rng_state, seed_everything


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def predict_neural(model, features, records, batch_size=32, device="cpu"):
    if not records:
        return np.empty((0, 3, 3))
    model.eval().to(device)
    output = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = features.encode(records[start : start + batch_size])
            output.append(model(to_device(batch, device)).softmax(-1).cpu().numpy())
    return np.concatenate(output)


def train_neural(model, features, train, validation, cfg, directory, signature, resume=False):
    directory = Path(directory)
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = max(1, math.ceil(len(train) / cfg.batch_size) * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_fraction)

    def rate(step):
        return (
            (step + 1) / max(1, warmup)
            if step < warmup
            else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    amp = cfg.mixed_precision and str(cfg.device).startswith("cuda")
    if cfg.mixed_precision and not amp:
        raise ValueError("Mixed precision is currently supported on CUDA only")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    y = torch.as_tensor(labels_array(train), dtype=torch.long)
    start_epoch, best, stale, history = 0, -math.inf, 0, []
    if resume:
        state = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Resume configuration/data/split mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
        start_epoch, best, stale, history = (
            state["epoch"] + 1,
            state["best"],
            state["stale"],
            state["history"],
        )
    reduction = "sum" if model.cfg.name == "m4" else "mean"
    smoothing = 0.2 if model.cfg.name == "aen" else 0.0
    for epoch in range(start_epoch, cfg.epochs):
        if stale >= cfg.patience:
            break
        model.train()
        losses = []
        order = torch.randperm(len(train)).tolist()
        for offset in range(0, len(order), cfg.batch_size):
            indices = order[offset : offset + cfg.batch_size]
            target = y[indices].to(cfg.device)
            if not (target >= 0).any():
                continue
            inputs = to_device(features.encode([train[i] for i in indices]), cfg.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                logits = model(inputs)
                loss = masked_loss(logits, target, reduction, smoothing)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
        probabilities = predict_neural(model, features, validation, cfg.batch_size, cfg.device)
        metrics = evaluate(labels_array(validation), probabilities)
        score = metrics["mean_aspect_macro_f1"]
        if score is None or not losses:
            raise ValueError("Training and selection partitions must contain observed labels")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_mean_aspect_macro_f1": score,
                "validation_decision_accuracy": metrics["decision_accuracy"],
            }
        )
        if score > best:
            best, stale = score, 0
            save_weights(model, directory / "best.safetensors")
        else:
            stale += 1
        torch.save(
            {
                "signature": signature,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(),
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "history": history,
            },
            directory / "last.pt",
        )
        write_json(directory / "history.json", history)
    load_weights(model, directory / "best.safetensors")
    return history


def fit_run(
    records,
    manifest,
    model_cfg,
    train_cfg,
    directory,
    validation_role="validation",
    resume=False,
    mode="reconstruction",
):
    if model_cfg.name == "m6":
        raise ValueError("Use the held-out ensemble workflow for M6")
    if validation_role not in {"validation", "meta_validation", "inner_validation"}:
        raise ValueError("Checkpoint selection may not consume TEST or meta-training records")
    validate_splits(records, manifest)
    train, validation = (
        select(records, manifest, "train"),
        select(records, manifest, validation_role),
    )
    if not train or not validation:
        raise ValueError("Nonempty training and validation partitions required")
    if mode not in {"synthetic", "reconstruction"}:
        raise ValueError("Exact reproduction unavailable; verify original artifacts first")
    if model_cfg.encoder == "tiny" and mode != "synthetic":
        raise ValueError("Tiny encoders are restricted to synthetic smoke mode")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()) and not resume:
        raise ValueError("Run directory is not empty; choose a new output or resume")
    seed_everything(train_cfg.seed, train_cfg.threads)
    signature = fingerprint(
        {
            "model": asdict(model_cfg),
            "training": asdict(train_cfg),
            "records": records,
            "manifest": manifest,
            "selection": validation_role,
        }
    )
    if resume:
        verify(directory)
        old = read_json(directory / "run.json")
        if old["signature"] != signature:
            raise ValueError("Resume configuration/data/split mismatch")
        pipeline = Pipeline.load(directory / "preprocessing.json")
    else:
        pipeline = Pipeline()
    train, _ = pipeline.transform(train)
    validation, _ = pipeline.transform(validation)
    # Validate grouping after deterministic normalization/transliteration as well.
    transformed, _ = pipeline.transform(records)
    validate_splits(transformed, manifest)
    if not resume:
        pipeline.fit(train).save(directory / "preprocessing.json")
    metadata = {
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "mode": mode,
        "signature": signature,
        "environment": environment(),
        "selection": validation_role,
        "train_ids": [r["review_id"] for r in train],
        "selection_ids": [r["review_id"] for r in validation],
        "input_hash": fingerprint(records),
        "split_hash": fingerprint(manifest),
        "manifest": manifest,
    }
    if model_cfg.name in CLASSICAL:
        if resume:
            raise ValueError("Classical fitting is atomic; resume is supported for neural models")
        model = build_model(model_cfg, train_cfg.seed).fit(train, validation)
        model.save(directory)
        history = []
    else:
        features = (
            Features.load(directory, model_cfg, pipeline)
            if resume
            else Features(model_cfg, pipeline).fit(train)
        )
        features.save(directory)
        local_config = (
            read_json(directory / "encoder.json") if resume and model_cfg.encoder == "hf" else None
        )
        encoder = make_encoder(model_cfg, local_config) if resume else None
        model = build_model(model_cfg, train_cfg.seed, encoder)
        if model_cfg.encoder == "hf":
            write_json(directory / "encoder.json", model.encoder.model.config.to_dict())
        if model_cfg.encoder == "static" and not resume:
            if model_cfg.fasttext_path:
                metadata["fasttext_vectors_found"] = load_vectors(
                    model_cfg.fasttext_path, features.vocabulary, model.encoder.embedding
                )
                from bangla_absa.io import file_hash

                metadata["fasttext_sha256"] = file_hash(model_cfg.fasttext_path)
            elif mode != "synthetic" and model_cfg.name in {
                "fasttext_bilstm",
                "memnet",
                "atae_lstm",
                "ian",
                "ram",
            }:
                raise ValueError(
                    "Provide a local fastText .vec in model.fasttext_path; see data/README.md"
                )
        history = train_neural(
            model, features, train, validation, train_cfg, directory, signature, resume
        )
    metadata["epochs_executed"] = len(history)
    write_json(directory / "run.json", metadata)
    seal(directory)
    return metadata


def load_run(directory):
    directory = Path(directory)
    verify(directory)
    meta = read_json(directory / "run.json")
    model_cfg = strict_dataclass(ModelConfig, meta["model"])
    pipeline = Pipeline.load(directory / "preprocessing.json")
    if model_cfg.name in CLASSICAL:
        model = build_model(model_cfg).load(directory)
        return model, None, pipeline, meta
    encoder_cfg = read_json(directory / "encoder.json") if model_cfg.encoder == "hf" else None
    model = build_model(model_cfg, encoder=make_encoder(model_cfg, encoder_cfg))
    load_weights(model, directory / "best.safetensors")
    features = Features.load(directory, model_cfg, pipeline)
    return model, features, pipeline, meta


def predict_run(directory, records, device="cpu"):
    kind = read_json(Path(directory) / "run.json").get("kind")
    if kind == "voting":
        from bangla_absa.experiments import predict_vote

        return predict_vote(directory, records, device)
    if kind == "ensemble":
        from bangla_absa.training.ensemble import predict_ensemble

        return predict_ensemble(directory, records, device)
    model, features, pipeline, meta = load_run(directory)
    records, _ = pipeline.transform(records)
    if features is None:
        p = model.predict_proba(records)
    else:
        p = predict_neural(model, features, records, meta["training"]["batch_size"], device)
    return p, meta
