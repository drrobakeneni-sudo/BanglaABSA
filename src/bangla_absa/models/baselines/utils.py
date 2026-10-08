import torch
from torch import nn


def aspect_batch(batch, aspect, prefix="target"):
    return {
        key[len(prefix) + 1 :]: value[:, aspect]
        for key, value in batch.items()
        if key.startswith(prefix + "_")
    }


class QueryAttention(nn.Module):
    def __init__(self, memory_width, query_width, hidden):
        super().__init__()
        self.memory = nn.Linear(memory_width, hidden)
        self.query = nn.Linear(query_width, hidden, bias=False)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, memory, query, mask):
        scores = self.score(torch.tanh(self.memory(memory) + self.query(query).unsqueeze(1)))
        weights = scores.squeeze(-1).masked_fill(~mask.bool(), -torch.inf).softmax(-1)
        return torch.bmm(weights.unsqueeze(1), memory).squeeze(1)


def local_weights(alignment, mask, window=0):
    """Distance to text-derived aligned clause, never a fabricated target span."""
    n = mask.shape[1]
    positions = torch.arange(n, device=mask.device)
    distance = (positions[:, None] - positions[None, :]).abs().float()
    distance = distance.unsqueeze(0).expand(mask.shape[0], -1, -1)
    distance = distance.masked_fill(~alignment.bool().unsqueeze(1), float(n))
    nearest = distance.min(-1).values
    weight = (
        1 - (nearest - window).clamp_min(0) / mask.sum(1, keepdim=True).clamp_min(1)
    ).clamp_min(0)
    weight = torch.where(alignment.any(1, keepdim=True), weight, torch.ones_like(weight))
    return weight * mask
