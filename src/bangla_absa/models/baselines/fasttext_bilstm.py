import torch
from torch import nn

from bangla_absa.models.common import TextModel, masked_mean, run_lstm


def load_vectors(path, vocabulary, embedding):
    """Stream a local official fastText .vec; no entire embedding file in RAM."""
    found = 0
    with open(path, encoding="utf-8") as f, torch.no_grad():
        header = f.readline().split()
        if len(header) != 2 or int(header[1]) != embedding.embedding_dim:
            raise ValueError("fastText vector width does not match configured embedding width")
        for line in f:
            word, _, values = line.rstrip().partition(" ")
            if word in vocabulary:
                vector = torch.tensor([float(v) for v in values.split()])
                if vector.numel() != embedding.embedding_dim:
                    raise ValueError("Malformed fastText vector")
                embedding.weight[vocabulary[word]].copy_(vector)
                found += 1
    return found


class FastTextBiLSTM(TextModel):
    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        self.lstm = nn.LSTM(cfg.width, cfg.lstm_hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(cfg.lstm_hidden * 2, 9)

    def forward(self, batch):
        x, mask = self.encode(batch)
        return self.head(masked_mean(run_lstm(self.lstm, x, mask), mask)).reshape(-1, 3, 3)
