import numpy as np
import pytest

from bangla_absa.evaluation.agreement import branch_analysis
from bangla_absa.evaluation.metrics import curves, evaluate
from bangla_absa.evaluation.predictions import align_predictions, load_predictions, save_predictions
from bangla_absa.evaluation.statistics import (
    cluster_bootstrap,
    holm,
    mcnemar,
    paired_seed_test,
    summarize_seeds,
)
from bangla_absa.evaluation.tables import export_tables


def test_known_metrics_and_undefined_auc():
    y = np.array([[0, 1, 2], [2, -1, 0]])
    p = np.eye(3)[np.array([[0, 1, 2], [0, 2, 0]])]
    result = evaluate(y, p)
    assert result["decision_accuracy"] == 0.8
    assert result["exact_match_review_accuracy"] == 1
    assert result["fully_labeled_reviews"] == 1
    assert result["per_aspect"]["Shop"]["macro_auc"] is None
    assert len(curves(y, p)) == 3
    assert evaluate(np.full((2, 3), -1), p)["decision_accuracy"] is None
    with pytest.raises(ValueError):
        evaluate(y, p * 2)


def test_predictions_alignment_and_tables(tmp_path):
    y = np.array([[0, 1, 2], [1, -1, 0]])
    p = np.eye(3)[y.clip(0)]
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    save_predictions(first, ["a", "b"], y, p, "test", 17, "synthetic-a")
    save_predictions(second, ["b", "a"], y[::-1], p[::-1], "test", 17, "synthetic-b")
    aligned = align_predictions(load_predictions(first), load_predictions(second))
    np.testing.assert_array_equal(aligned[2], aligned[3])
    assert len(export_tables([first, second], tmp_path / "table.csv")) == 2
    save_predictions(second, ["b", "c"], y, p, "test", 17, "c")
    with pytest.raises(ValueError, match="matched"):
        align_predictions(load_predictions(first), load_predictions(second))
    with pytest.raises(ValueError):
        save_predictions(first, ["a", "a"], y, p, "test", 17, "a")


def test_statistics_and_cluster_resampling():
    result = summarize_seeds({17: 0.4, 29: 0.6, 43: 0.5})
    assert result["mean"] == pytest.approx(0.5)
    assert result["sample_sd"] == pytest.approx(0.1)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert paired_seed_test({1: 1, 2: 2}, {1: 0, 2: 0})["p"] > 0
    with pytest.raises(ValueError):
        paired_seed_test({1: 1}, {2: 2})
    assert paired_seed_test({1: 1, 2: 1}, {1: 1, 2: 1})["p"] is None
    y = np.array([[0, 1, 2], [2, -1, 0], [1, 1, 1]])
    a, b = y.clip(0), np.zeros_like(y)
    one = cluster_bootstrap(y, a, b, repetitions=40)
    assert one == cluster_bootstrap(y, a, b, repetitions=40)
    assert mcnemar(y, a, b)["exact_binomial_p"] <= 1
    assert mcnemar(y, a, a)["exact_binomial_p"] == 1


def test_branch_disagreement():
    y = np.array([[0, 1, 2]])
    a = np.eye(3)[np.array([[0, 0, 0]])]
    b = np.eye(3)[np.array([[0, 1, 1]])]
    c = np.eye(3)[y]
    result = branch_analysis(y, a, b, c)
    assert result["agree_correct"] == 1 and result["m5_only_correct"] == 1
    assert result["disagree_neither_correct"] == 1 and result["m6_neither_and_correct"] == 1
    assert result["hard_label_oracle_accuracy"] == pytest.approx(2 / 3)
