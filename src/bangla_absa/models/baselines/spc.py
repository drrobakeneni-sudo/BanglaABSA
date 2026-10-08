import torch
from torch import nn

from bangla_absa.models.baselines.utils import aspect_batch
from bangla_absa.models.common import TextModel


class SPC(TextModel):
    """Review/category sentence pair. Encoder sees category tokens, not a target span."""

    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.width, 3))

    def forward(self, batch):
        outputs = []
        for a in range(3):
            x, _ = self.encode(aspect_batch(batch, a, prefix="pair"))
            outputs.append(self.classifier(x[:, 0]))
        return torch.stack(outputs, 1)
