import torch
from torch import nn

from bangla_absa.models.baselines.utils import aspect_batch
from bangla_absa.models.common import TextModel, masked_mean


class AEN(TextModel):
    """Three attentional interactions; category phrase replaces unavailable target span."""

    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        self.context_attention = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True)
        self.target_attention = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True)
        self.interaction = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True)
        self.context_ff = nn.Sequential(
            nn.Linear(cfg.width, cfg.ff_width), nn.ReLU(), nn.Linear(cfg.ff_width, cfg.width)
        )
        self.target_ff = nn.Sequential(
            nn.Linear(cfg.width, cfg.ff_width), nn.ReLU(), nn.Linear(cfg.ff_width, cfg.width)
        )
        self.head = nn.Linear(cfg.width * 3, 3)

    def forward(self, batch):
        x, mask = self.encode(batch)
        context = self.context_ff(
            self.context_attention(x, x, x, key_padding_mask=~mask, need_weights=False)[0]
        )
        outputs = []
        for a in range(3):
            target, tmask = self.encode(aspect_batch(batch, a))
            target = self.target_ff(
                self.target_attention(target, x, x, key_padding_mask=~mask, need_weights=False)[0]
            )
            interaction = self.interaction(
                context, target, target, key_padding_mask=~tmask, need_weights=False
            )[0]
            rep = torch.cat(
                (
                    masked_mean(context, mask),
                    masked_mean(interaction, mask),
                    masked_mean(target, tmask),
                ),
                -1,
            )
            outputs.append(self.head(rep))
        return torch.stack(outputs, 1)
