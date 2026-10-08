import torch
from torch import nn

from bangla_absa.models.common import AdditiveAttention, TextModel, masked_mean, run_lstm


class ABSANet(TextModel):
    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        self.refinements = (
            nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        cfg.width,
                        cfg.heads,
                        cfg.ff_width,
                        cfg.dropout,
                        batch_first=True,
                        norm_first=False,
                    )
                    for _ in range(3)
                ]
            )
            if cfg.refinement
            else nn.ModuleList()
        )
        self.lstm = (
            nn.LSTM(cfg.width, cfg.lstm_hidden, batch_first=True, bidirectional=True)
            if cfg.bilstm
            else None
        )
        width = cfg.lstm_hidden * 2 if self.lstm else cfg.width
        self.attentions = (
            nn.ModuleList([AdditiveAttention(width, cfg.classifier_hidden) for _ in range(3)])
            if cfg.aspect_attention
            else None
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, cfg.classifier_hidden),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.classifier_hidden, 3),
                )
                if cfg.dnn
                else nn.Linear(width, 3)
                for _ in range(3)
            ]
        )

    def forward(self, batch):
        x, mask = self.encode(batch)
        for block in self.refinements:
            x = block(x, src_key_padding_mask=~mask)
        if self.lstm is not None:
            x = run_lstm(self.lstm, x, mask)
        representations = (
            [att(x, mask)[0] for att in self.attentions]
            if self.attentions
            else [masked_mean(x, mask)] * 3
        )
        return torch.stack([head(rep) for head, rep in zip(self.heads, representations)], 1)
