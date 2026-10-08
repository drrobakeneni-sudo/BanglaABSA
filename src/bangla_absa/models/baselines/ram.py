import torch
from torch import nn

from bangla_absa.models.baselines.utils import QueryAttention, aspect_batch, local_weights
from bangla_absa.models.common import TextModel, masked_mean, run_lstm


class RAM(TextModel):
    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        memory_width = cfg.lstm_hidden * 2 + 1
        self.lstm = nn.LSTM(cfg.width, cfg.lstm_hidden, batch_first=True, bidirectional=True)
        self.attention = QueryAttention(memory_width, cfg.width * 2, cfg.width)
        self.recurrence = nn.GRUCell(memory_width, cfg.width)
        self.head = nn.Linear(cfg.width, 3)

    def forward(self, batch):
        x, mask = self.encode(batch)
        sequence = run_lstm(self.lstm, x, mask)
        outputs = []
        for a in range(3):
            target, tmask = self.encode(aspect_batch(batch, a))
            query = masked_mean(target, tmask)
            weights = local_weights(batch["alignment"][:, a], mask)
            memory = torch.cat((sequence * weights.unsqueeze(-1), (1 - weights).unsqueeze(-1)), -1)
            state = torch.zeros_like(query)
            for _ in range(self.cfg.hops):
                attended = self.attention(memory, torch.cat((state, query), -1), mask)
                state = self.recurrence(attended, state)
            outputs.append(self.head(state))
        return torch.stack(outputs, 1)
