import torch
from torch import nn

from bangla_absa.models.baselines.utils import QueryAttention, aspect_batch
from bangla_absa.models.common import TextModel, masked_mean, run_lstm


class ATAELSTM(TextModel):
    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        h = cfg.lstm_hidden
        self.lstm = nn.LSTM(cfg.width * 2, h, batch_first=True)
        self.attention = QueryAttention(h, cfg.width, h)
        self.context = nn.Linear(h, h)
        self.final_state = nn.Linear(h, h)
        self.head = nn.Linear(h, 3)

    def forward(self, batch):
        x, mask = self.encode(batch)
        outputs = []
        for a in range(3):
            target, tmask = self.encode(aspect_batch(batch, a))
            query = masked_mean(target, tmask)
            memory = run_lstm(
                self.lstm, torch.cat((x, query[:, None].expand(-1, x.shape[1], -1)), -1), mask
            )
            attended = self.attention(memory, query, mask)
            last = memory[torch.arange(x.shape[0], device=x.device), mask.sum(1) - 1]
            outputs.append(self.head(torch.tanh(self.context(attended) + self.final_state(last))))
        return torch.stack(outputs, 1)
