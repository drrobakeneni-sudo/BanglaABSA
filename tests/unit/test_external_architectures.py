from dataclasses import replace

import numpy as np
import pytest
import torch

from bangla_absa.config import tiny_config
from bangla_absa.models.baselines.fasttext_bilstm import load_vectors
from bangla_absa.models.common import HFEncoder, StaticEncoder
from bangla_absa.models.registry import build_model
from bangla_absa.preprocessing.pipeline import Pipeline
from bangla_absa.preprocessing.tokenization import Features
from bangla_absa.training.checkpoints import load_weights, save_weights


@pytest.mark.parametrize("architecture", ["bert", "electra", "xlm-roberta", "deberta-v2"])
def test_real_backbone_classes_without_download(architecture, records):
    cfg = tiny_config("banglabert", encoder="hf", cue_embeddings=False)
    values = {
        "model_type": architecture,
        "hidden_size": 24,
        "embedding_size": 24,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 48,
        "vocab_size": 128,
        "max_position_embeddings": 64,
        "hidden_dropout_prob": 0,
        "attention_probs_dropout_prob": 0,
        "pad_token_id": 0,
    }
    encoder = HFEncoder(cfg, local_config=values)
    model = build_model(cfg, encoder=encoder).eval()
    batch = {
        "input_ids": torch.tensor([[2, 4, 5, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "token_type_ids": torch.zeros(1, 5, dtype=torch.long),
    }
    logits = model(batch)
    assert logits.shape == (1, 3, 3) and torch.isfinite(logits).all()


def test_hf_offsets_and_local_roundtrip(tmp_path, records):
    from transformers import BertTokenizerFast

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nপণ্য\nভালো\nকিন্তু\nদোকান\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.save_pretrained(tmp_path / "tokenizer_source")
    cfg = tiny_config("spc", encoder="hf", checkpoint=str(tmp_path / "tokenizer_source"))
    pipeline = Pipeline().fit(records)
    features = Features(cfg, pipeline).fit(records)
    batch = features.encode(records[:2])
    assert batch["pair_input_ids"].shape == (2, 3, 32)
    features.save(tmp_path / "saved")
    restored = Features.load(tmp_path / "saved", cfg, pipeline)
    torch.testing.assert_close(batch["cue_ids"], restored.encode(records[:2])["cue_ids"])


def test_static_fasttext_loader_and_embedding_ablation(tmp_path, records):
    cfg = tiny_config("fasttext_bilstm", encoder="static", width=24)
    encoder = StaticEncoder(cfg)
    path = tmp_path / "synthetic.vec"
    path.write_text("1 24\nপণ্য " + " ".join(["0.25"] * 24) + "\n", encoding="utf-8")
    assert load_vectors(path, {"পণ্য": 4}, encoder.embedding) == 1
    np.testing.assert_allclose(encoder.embedding.weight[4].detach().numpy(), 0.25)
    pipeline = Pipeline().fit(records)
    features = Features(cfg, pipeline).fit(records)
    model = build_model(cfg, encoder=encoder).eval()
    batch = features.encode(records[:2])
    p = model(batch)
    save_weights(model, tmp_path / "model.safetensors")
    load_weights(model, tmp_path / "model.safetensors")
    torch.testing.assert_close(p, model(batch))
    m4 = build_model(replace(cfg, name="m4"))
    assert m4(batch).shape == (2, 3, 3)
