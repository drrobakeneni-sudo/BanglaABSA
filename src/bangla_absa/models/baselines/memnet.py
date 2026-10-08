import torch
from torch import nn

from bangla_absa.models.baselines.utils import QueryAttention, aspect_batch
from bangla_absa.models.common import TextModel, masked_mean


class MemNet(TextModel):
    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        self.attention = QueryAttention(cfg.width, cfg.width, cfg.width)
        self.transition = nn.Linear(cfg.width, cfg.width)
        self.head = nn.Linear(cfg.width, 3)

    def forward(self, batch):
        memory, mask = self.encode(batch)
        outputs = []
        for a in range(3):
            target, tmask = self.encode(aspect_batch(batch, a))
            query = masked_mean(target, tmask)
            for _ in range(self.cfg.hops):
                query = self.transition(query) + self.attention(memory, query, mask)
            outputs.append(self.head(query))
        return torch.stack(outputs, 1)
