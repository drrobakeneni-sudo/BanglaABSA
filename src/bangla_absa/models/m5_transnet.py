import torch
from torch import nn

from bangla_absa.models.common import PositionEncoding, TextModel, masked_mean


class TransNet(TextModel):
    def __init__(self, cfg, encoder=None):
        super().__init__(cfg, encoder)
        self.position = PositionEncoding() if cfg.extra_position else nn.Identity()
        self.blocks = (
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
                    for _ in range(cfg.encoder_layers)
                ]
            )
            if cfg.transformer_encoder
            else []
        )
        self.aspect_queries = nn.Parameter(torch.randn(3, cfg.width) * 0.02)
        self.aasa = (
            nn.MultiheadAttention(cfg.width, cfg.heads, cfg.dropout, batch_first=True)
            if cfg.aasa
            else None
        )
        self.decoders = (
            nn.ModuleList(
                [
                    nn.TransformerDecoderLayer(
                        cfg.width, cfg.heads, cfg.ff_width, cfg.dropout, batch_first=True
                    )
                    for _ in range(cfg.decoder_layers)
                ]
            )
            if cfg.decoder
            else []
        )
        self.projection = nn.Linear(cfg.width, 128)
        self.heads = nn.ModuleList([nn.Linear(128, 3) for _ in range(1 if cfg.single_head else 3)])
        # Reconstructed causal order: Product -> Shop -> Transport.
        self.register_buffer("decoder_mask", torch.triu(torch.ones(3, 3, dtype=torch.bool), 1))

    def forward(self, batch):
        memory, mask = self.encode(batch)
        memory = self.position(memory)
        for block in self.blocks:
            memory = block(memory, src_key_padding_mask=~mask)
        query = self.aspect_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        if self.aasa is not None:
            aspects = (
                query
                + self.aasa(query, memory, memory, key_padding_mask=~mask, need_weights=False)[0]
            )
        else:
            aspects = masked_mean(memory, mask).unsqueeze(1).expand(-1, 3, -1)
        for block in self.decoders:
            aspects = block(
                aspects, memory, tgt_mask=self.decoder_mask, memory_key_padding_mask=~mask
            )
        representations = self.projection(aspects)
        if self.cfg.single_head:
            return self.heads[0](representations)
        return torch.stack([head(representations[:, a]) for a, head in enumerate(self.heads)], 1)
