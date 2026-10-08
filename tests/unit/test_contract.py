import numpy as np
import pytest
import torch

from bangla_absa.data.schema import labels_array, validate_records
from bangla_absa.data.splits import make_splits, validate_splits
from bangla_absa.evaluation.metrics import evaluate
from bangla_absa.models.common import masked_loss
from bangla_absa.models.m6_ensemble import Stacker, weighted_features


def record(i, text=None, labels=None):
    return {
        "review_id": str(i),
        "text": text or f"সিনথেটিক পণ্য নমুনা {i}",
        "labels": labels or {"Product": "Positive", "Shop": None, "Transport": "Negative"},
    }


def test_schema_null_is_not_neutral():
    rows = validate_records([record(1)])
    assert labels_array(rows).tolist() == [[2, -1, 0]]
    with pytest.raises(ValueError):
        validate_records([record(1), record(1)])
    with pytest.raises(ValueError):
        validate_records([record(1, labels={"Product": "happy"})])


def test_split_group_and_duplicate_leakage():
    rows = [record(i) for i in range(40)]
    rows[1]["text"] = rows[0]["text"]
    rows[2]["group_id"] = rows[1]["group_id"] = "family"
    split = make_splits(rows, seed=17)
    assert split == make_splits(list(reversed(rows)), seed=17)
    validate_splits(rows, split)
    assert len({split[str(i)] for i in range(3)}) == 1
    bad = dict(split)
    bad["1"] = "test" if split["0"] != "test" else "train"
    with pytest.raises(ValueError, match="group|duplicate"):
        validate_splits(rows, bad)


def test_m6_contract_and_order():
    m = Stacker()
    assert sum(p.numel() for p in m.parameters()) == 462729
    p = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    p = p.softmax(-1)
    w = torch.tensor([[0.2, 0.8]] * 3)
    f = weighted_features(p, p.flip(0), w)
    assert f.shape == (2, 18)
    torch.testing.assert_close(f[:, :9], (p * 0.2).flatten(1))
    q = m(f).softmax(-1)
    assert q.shape == (2, 3, 3)
    torch.testing.assert_close(q.sum(-1), torch.ones(2, 3))


def test_missing_loss_and_metrics():
    logits = torch.randn(2, 3, 3, requires_grad=True)
    zero = masked_loss(logits, torch.full((2, 3), -1))
    zero.backward()
    assert zero.item() == 0 and torch.isfinite(logits.grad).all()
    y = np.array([[0, -1, 2], [1, 1, -1]])
    p = np.eye(3)[np.array([[0, 2, 2], [2, 1, 0]])]
    out = evaluate(y, p)
    assert out["reviews"] == 2 and out["decisions"] == 4
    assert out["decision_accuracy"] == 0.75
