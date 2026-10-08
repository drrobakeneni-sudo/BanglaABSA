import torch
from torch import nn

from bangla_absa.models.common import TextModel, masked_mean


class PLM(TextModel):
    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.width, 9)

    def forward(self, batch):
        x, mask = self.encode(batch)
        return self.head(self.dropout(masked_mean(x, mask))).reshape(-1, 3, 3)


class ThreeHeads(TextModel):
    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.width, cfg.classifier_hidden),
                    nn.Tanh(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.classifier_hidden, 3),
                )
                for _ in range(3)
            ]
        )

    def forward(self, batch):
        x, mask = self.encode(batch)
        representation = masked_mean(x, mask)
        return torch.stack([head(representation) for head in self.heads], 1)
