import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_loss(logits, labels, reduction="mean", smoothing=0.0):
    if logits.shape != (*labels.shape, 3) or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("Expected logits [B,3,3] and labels [B,3]")
    valid = labels >= 0
    if (labels[valid] > 2).any() or (labels < -1).any():
        raise ValueError("Labels must be -1 or 0..2")
    if not valid.any():
        return logits.sum() * 0
    terms = F.cross_entropy(
        logits.reshape(-1, 3),
        labels.clamp_min(0).reshape(-1),
        reduction="none",
        label_smoothing=smoothing,
    ).reshape_as(labels)
    # Equal weight to each observed aspect; M4 sums means, M6 averages means.
    per_aspect = (terms * valid).sum(0) / valid.sum(0).clamp_min(1)
    if reduction == "sum":
        return per_aspect.sum()
    if reduction != "mean":
        raise ValueError("Loss reduction must be sum or mean")
    return per_aspect.sum() / valid.any(0).sum()


def masked_mean(x, mask):
    return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def run_lstm(lstm, x, mask):
    lengths = mask.sum(1).cpu()
    if (lengths == 0).any():
        raise ValueError("Each review needs at least one nonpadding token")
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    result, _ = lstm(packed)
    return pad_packed_sequence(result, batch_first=True, total_length=x.shape[1])[0]


class PositionEncoding(nn.Module):
    def forward(self, x):
        width = x.shape[-1]
        pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, width, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / width)
        )
        result = torch.zeros_like(x[:1])
        result[0, :, 0::2] = torch.sin(pos * div)
        result[0, :, 1::2] = torch.cos(pos * div[: width // 2])
        return x + result


class AdditiveAttention(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.projection = nn.Linear(width, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, mask):
        scores = self.score(torch.tanh(self.projection(x))).squeeze(-1)
        weights = scores.masked_fill(~mask.bool(), -torch.inf).softmax(-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1), weights


class TinyEncoder(nn.Module):
    """Random contextual encoder exclusively for offline synthetic checks."""

    def __init__(self, cfg):
        super().__init__()
        self.width = cfg.width
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width, padding_idx=0)
        self.position = PositionEncoding()
        self.block = nn.TransformerEncoderLayer(
            cfg.width, cfg.heads, cfg.ff_width, cfg.dropout, batch_first=True
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        x = self.position(self.embedding(input_ids))
        return self.block(x, src_key_padding_mask=~attention_mask.bool())


class StaticEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.width = cfg.width
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width, padding_idx=0)

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.embedding(input_ids)


class HFEncoder(nn.Module):
    def __init__(self, cfg, local_config=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if local_config is not None:
            values = dict(local_config)
            model_type = values.pop("model_type")
            self.model = AutoModel.from_config(AutoConfig.for_model(model_type, **values))
        elif cfg.pretrained:
            self.model = AutoModel.from_pretrained(cfg.checkpoint, revision=cfg.revision)
        else:
            self.model = AutoModel.from_config(
                AutoConfig.from_pretrained(cfg.checkpoint, revision=cfg.revision)
            )
        self.width = self.model.config.hidden_size
        if self.width != cfg.width:
            raise ValueError(
                f"Configured width {cfg.width} differs from checkpoint width {self.width}"
            )

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if (
            token_type_ids is not None
            and "token_type_ids" in self.model.forward.__code__.co_varnames
        ):
            inputs["token_type_ids"] = token_type_ids
        return self.model(**inputs).last_hidden_state


def make_encoder(cfg, local_config=None):
    if cfg.encoder == "tiny":
        return TinyEncoder(cfg)
    if cfg.encoder == "static":
        return StaticEncoder(cfg)
    return HFEncoder(cfg, local_config)


class TextModel(nn.Module):
    def __init__(self, cfg, encoder=None):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder if encoder is not None else make_encoder(cfg)
        self.cues = nn.Embedding(4, cfg.cue_width, padding_idx=0) if cfg.cue_embeddings else None
        self.cue_projection = nn.Linear(cfg.width + cfg.cue_width, cfg.width) if self.cues else None

    def encode(self, batch):
        mask = batch["attention_mask"].bool()
        if mask.ndim != 2 or (~mask.any(-1)).any():
            raise ValueError("Expected nonempty token sequences")
        x = self.encoder(**batch)
        if self.cues is not None:
            cues = batch.get("cue_ids", torch.zeros_like(batch["input_ids"]))
            x = self.cue_projection(torch.cat((x, self.cues(cues)), dim=-1))
        return x * mask.unsqueeze(-1), mask
