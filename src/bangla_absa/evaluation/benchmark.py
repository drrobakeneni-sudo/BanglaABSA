import time

import numpy as np
import torch


def benchmark(model, batch, warmup=3, repetitions=10, device="cpu", includes_preprocessing=False):
    if warmup < 0 or repetitions < 1:
        raise ValueError("Invalid benchmark iteration counts")
    model = model.to(device).eval()
    batch = {k: v.to(device) for k, v in batch.items()}

    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        synchronize()
        for _ in range(repetitions):
            start = time.perf_counter()
            model(batch)
            synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    first = next(iter(batch.values()))
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "batch_latency_ms_mean": float(np.mean(samples)),
        "batch_latency_ms_sd": float(np.std(samples, ddof=1)) if repetitions > 1 else None,
        "amortized_ms_per_review": float(np.mean(samples) / len(first)),
        "batch_size": len(first),
        "padded_length": batch.get("input_ids", first).shape[-1],
        "device": str(device),
        "warmup": warmup,
        "repetitions": repetitions,
        "includes_preprocessing": includes_preprocessing,
        "flops": None,
        "flops_reason": "No complete validated FLOP counter for these mixed architectures",
    }
