from dataclasses import replace

import numpy as np
import pytest
import torch

from bangla_absa.config import tiny_config
from bangla_absa.data.schema import labels_array
from bangla_absa.models.common import AdditiveAttention, masked_loss
from bangla_absa.models.m6_ensemble import EnsembleNet, weighted_features
from bangla_absa.models.registry import CLASSICAL, NEURAL, build_model
from bangla_absa.preprocessing.pipeline import Pipeline
from bangla_absa.preprocessing.tokenization import Features


@pytest.mark.parametrize("name", list(NEURAL))
def test_every_neural_forward_gradient_and_padding(name, records):
    cfg = tiny_config(name)
    pipeline = Pipeline().fit(records[:60])
    features = Features(cfg, pipeline).fit(records[:60])
    batch = features.encode(records[:2])
    model = build_model(cfg).eval()
    logits = model(batch)
    assert logits.shape == (2, 3, 3)
    loss = masked_loss(logits, torch.tensor(labels_array(records[:2])))
    loss.backward()
    assert torch.isfinite(loss) and all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    modified = {k: v.clone() for k, v in batch.items()}
    for key in list(modified):
        if key.endswith("input_ids"):
            mask_name = key.replace("input_ids", "attention_mask")
            modified[key][modified[mask_name] == 0] = 20
    torch.testing.assert_close(logits, model(modified), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(2, 3))


@pytest.mark.parametrize("name", list(CLASSICAL))
def test_all_classical_models(name, records, tmp_path):
    cfg = tiny_config(name, trees=4)
    model = build_model(cfg).fit(records[:81], records[81:])
    p = model.predict_proba(records[:4])
    assert p.shape == (4, 3, 3)
    np.testing.assert_allclose(p.sum(-1), 1)
    model.save(tmp_path)
    restored = build_model(cfg).load(tmp_path)
    np.testing.assert_allclose(p, restored.predict_proba(records[:4]), atol=1e-6)


def test_classical_missing_classes(records):
    for r in records:
        r["labels"]["Product"] = "Positive"
    model = build_model(tiny_config("m1")).fit(records)
    p = model.predict_proba(records[:2])
    assert (p[:, 0, :2] == 0).all() and (p[:, 0, 2] == 1).all()
    with pytest.raises(ValueError, match="only fit"):
        model.fit(records, role="test")


def test_attention_equation_and_three_blocks(records):
    attn = AdditiveAttention(4, 2)
    x, mask = torch.randn(2, 5, 4), torch.tensor([[1, 1, 1, 0, 0]] * 2).bool()
    output, weights = attn(x, mask)
    expected = attn.score(torch.tanh(attn.projection(x))).squeeze(-1)
    expected = expected.masked_fill(~mask, -torch.inf).softmax(-1)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(output, (weights.unsqueeze(-1) * x).sum(1))
    model = build_model(tiny_config())
    assert len(model.refinements) == 3
    assert all(not block.norm_first for block in model.refinements)


@pytest.mark.parametrize(
    "name,field",
    [("m4", f) for f in ("refinement", "bilstm", "aspect_attention", "dnn")]
    + [("m5", f) for f in ("extra_position", "transformer_encoder", "aasa", "decoder")],
)
def test_ablations(name, field, records):
    cfg = replace(tiny_config(name), **{field: False})
    pipeline = Pipeline().fit(records)
    batch = Features(cfg, pipeline).fit(records).encode(records[:2])
    assert build_model(cfg)(batch).shape == (2, 3, 3)


def test_m5_heads_masks_and_conditioning(records):
    cfg = tiny_config("m5")
    pipeline = Pipeline().fit(records)
    batch = Features(cfg, pipeline).fit(records).encode(records[:2])
    model = build_model(cfg).eval()
    assert model.projection.out_features == 128
    assert torch.equal(model.decoder_mask, torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0]]).bool())
    p = model(batch)
    with torch.no_grad():
        model.aspect_queries[2].add_(1)
    changed = model(batch)
    torch.testing.assert_close(p[:, :2], changed[:, :2], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(p[:, 2], changed[:, 2])
    assert build_model(replace(cfg, single_head=True))(batch).shape == (2, 3, 3)


@pytest.mark.parametrize("mode", ["stacking", "equal_vote", "weighted_vote", "unweighted_stacking"])
def test_frozen_ensemble(mode, records):
    cfg = tiny_config()
    pipeline = Pipeline().fit(records)
    batch = Features(cfg, pipeline).fit(records).encode(records[:2])
    ensemble = EnsembleNet(
        build_model(cfg), build_model(replace(cfg, name="m5")), mode=mode
    ).train()
    assert not any(p.requires_grad for b in ensemble.branches for p in b.parameters())
    assert all(not b.training for b in ensemble.branches)
    assert ensemble(batch).shape == (2, 3, 3)
    with pytest.raises(ValueError):
        weighted_features(torch.ones(2, 3, 3), torch.ones(2, 3, 3), torch.ones(3, 2))
