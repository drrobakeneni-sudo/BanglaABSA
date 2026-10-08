from dataclasses import asdict, dataclass, fields
from typing import get_args

from bangla_absa.io import read_json


@dataclass
class ModelConfig:
    name: str = "m4"
    encoder: str = "hf"
    checkpoint: str = "csebuetnlp/banglabert"
    revision: str = "main"
    pretrained: bool = True
    width: int = 768
    vocab_size: int = 32000
    max_length: int = 256
    heads: int = 8
    ff_width: int = 3072
    lstm_hidden: int = 64
    classifier_hidden: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.3
    cue_width: int = 16
    cue_embeddings: bool = True
    refinement: bool = True
    bilstm: bool = True
    aspect_attention: bool = True
    dnn: bool = True
    extra_position: bool = True
    transformer_encoder: bool = True
    aasa: bool = True
    decoder: bool = True
    single_head: bool = False
    hops: int = 3
    fasttext_path: str | None = None
    lcf_window: int = 3
    trees: int = 0
    max_features: int = 50000
    svm_calibration_folds: int = 3

    def __post_init__(self):
        if self.encoder not in {"hf", "tiny", "static"}:
            raise ValueError("encoder must be hf, tiny, or static")
        positive = (
            self.width,
            self.vocab_size,
            self.max_length,
            self.heads,
            self.ff_width,
            self.lstm_hidden,
            self.classifier_hidden,
            self.encoder_layers,
            self.decoder_layers,
            self.hops,
            self.cue_width,
            self.max_features,
        )
        if any(v < 1 for v in positive) or self.width % self.heads:
            raise ValueError("Positive dimensions required; width must be divisible by heads")
        if self.max_length < 4 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid max_length or dropout")
        if self.encoder == "hf" and not self.checkpoint:
            raise ValueError("An explicit Hugging Face checkpoint is required")


@dataclass
class TrainingConfig:
    seed: int = 17
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    patience: int = 3
    gradient_clip: float = 1.0
    warmup_fraction: float = 0.1
    mixed_precision: bool = False
    device: str = "cpu"
    threads: int = 2

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.patience, self.threads) < 1:
            raise ValueError("Training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0,1)")


def strict_dataclass(cls, data):
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
        raise ValueError(f"Unknown fields in {cls.__name__}")
    for field in fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        types = get_args(field.type) or (field.type,)
        matches = any(
            (type(value) in (int, float) if t is float else type(value) is t) for t in types
        )
        if not matches:
            raise ValueError(f"Invalid type for {cls.__name__}.{field.name}")
    return cls(**data)


def load_config(path):
    data = read_json(path)
    if set(data) - {"model", "training", "provenance", "mode"}:
        raise ValueError("Unknown top-level config fields")
    if data.get("mode", "reconstruction") == "exact":
        raise ValueError(
            "Exact reproduction unavailable: original corpus, manifests, layer settings, "
            "parser, lexicons, seeds and stacking protocol have not been verified"
        )
    if data.get("mode", "reconstruction") not in {"reconstruction", "synthetic"}:
        raise ValueError("Invalid experiment mode")
    return (
        strict_dataclass(ModelConfig, data["model"]),
        strict_dataclass(TrainingConfig, data.get("training", {})),
        data,
    )


def tiny_config(name="m4", **kwargs):
    values = dict(
        name=name,
        encoder="tiny",
        width=24,
        vocab_size=128,
        max_length=32,
        heads=4,
        ff_width=48,
        lstm_hidden=12,
        classifier_hidden=24,
        dropout=0.0,
        cue_width=4,
        encoder_layers=1,
        decoder_layers=1,
    )
    values.update(kwargs)
    return ModelConfig(**values)


def config_dict(config):
    return asdict(config)
