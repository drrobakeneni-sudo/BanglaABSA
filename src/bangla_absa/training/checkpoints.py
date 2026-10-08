import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from bangla_absa.io import file_hash, read_json, write_json


def environment():
    versions = {}
    for name in ("torch", "transformers", "numpy", "scipy", "scikit-learn", "xgboost", "skops"):
        versions[name] = importlib.metadata.version(name)
    result = (
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if shutil.which("git")
        else None
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dependencies": versions,
        "git_commit": result.stdout.strip()
        if result is not None and result.returncode == 0
        else None,
    }


def save_weights(model, path):
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path))


def load_weights(model, path):
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)


def seal(directory):
    directory = Path(directory)
    hashes = {
        p.relative_to(directory).as_posix(): file_hash(p)
        for p in directory.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }
    write_json(directory / "checksums.json", hashes)


def verify(directory):
    directory = Path(directory).resolve()
    for name, expected in read_json(directory / "checksums.json").items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Checkpoint file missing, outside checkpoint, or checksum mismatch")
