import torch
from torch import nn

from bangla_absa.models.baselines.utils import QueryAttention, aspect_batch
from bangla_absa.models.common import TextModel, masked_mean, run_lstm


class IAN(TextModel):
    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        h = cfg.lstm_hidden
        self.context_lstm = nn.LSTM(cfg.width, h, batch_first=True)
        self.target_lstm = nn.LSTM(cfg.width, h, batch_first=True)
        self.context_attention = QueryAttention(h, h, h)
        self.target_attention = QueryAttention(h, h, h)
        self.head = nn.Linear(2 * h, 3)

    def forward(self, batch):
        x, mask = self.encode(batch)
        context = run_lstm(self.context_lstm, x, mask)
        outputs = []
        for a in range(3):
            target, tmask = self.encode(aspect_batch(batch, a))
            target = run_lstm(self.target_lstm, target, tmask)
            c = self.context_attention(context, masked_mean(target, tmask), mask)
            t = self.target_attention(target, masked_mean(context, mask), tmask)
            outputs.append(self.head(torch.cat((c, t), -1)))
        return torch.stack(outputs, 1)
