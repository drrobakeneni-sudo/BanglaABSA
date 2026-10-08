import torch
from torch import nn

from bangla_absa.models.baselines.utils import aspect_batch, local_weights
from bangla_absa.models.common import TextModel, masked_mean


class LCFBERT(TextModel):
    """Category-adapted local/global fusion with a shared contextual encoder.

    Local weighting uses input-only aligned clauses; absent cues retain full text.
    Global sentence-pair CLS is broadcast before fusion (documented adaptation).
    """

    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        self.fuse = nn.Linear(cfg.width * 2, cfg.width)
        self.attention = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True)
        self.head = nn.Linear(cfg.width, 3)

    def forward(self, batch):
        x, mask = self.encode(batch)
        outputs = []
        for a in range(3):
            paired, _ = self.encode(aspect_batch(batch, a, "pair"))
            weight = local_weights(batch["alignment"][:, a], mask, self.cfg.lcf_window)
            combined = self.fuse(
                torch.cat((x * weight.unsqueeze(-1), paired[:, :1].expand(-1, x.shape[1], -1)), -1)
            )
            combined = self.attention(
                combined, combined, combined, key_padding_mask=~mask, need_weights=False
            )[0]
            outputs.append(self.head(torch.tanh(masked_mean(combined, mask))))
        return torch.stack(outputs, 1)
