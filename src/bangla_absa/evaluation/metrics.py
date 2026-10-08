import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from bangla_absa import ASPECTS, SENTIMENTS


def validate_predictions(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities)
    if y.ndim != 2 or y.shape[1] != 3 or p.shape != (*y.shape, 3):
        raise ValueError("Expected labels [N,3] and probabilities [N,3,3]")
    if not np.isin(y, [-1, 0, 1, 2]).all():
        raise ValueError("Invalid gold label")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-5):
        raise ValueError("Invalid probabilities")
    return y, p


def _classification(y, p):
    if len(y) == 0:
        return {
            "decisions": 0,
            "accuracy": None,
            "macro_f1": None,
            "confusion": np.zeros((3, 3), dtype=int).tolist(),
            "classes": {},
        }
    pred = p.argmax(-1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], zero_division=0
    )
    classes = {}
    for c, name in enumerate(SENTIMENTS):
        binary = y == c
        defined = len(np.unique(binary)) == 2
        classes[name] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
            "auc": float(roc_auc_score(binary, p[:, c])) if defined else None,
            "ap": float(average_precision_score(binary, p[:, c])) if defined else None,
            "curve_reason": None if defined else "One-vs-rest labels have only one class",
        }
    aucs = [v["auc"] for v in classes.values()]
    return {
        "decisions": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_auc": float(np.mean(aucs)) if all(v is not None for v in aucs) else None,
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "classes": classes,
    }


def evaluate(y, probabilities):
    y, p = validate_predictions(y, probabilities)
    valid = y >= 0
    pooled = _classification(y[valid], p[valid])
    per_aspect = {
        a: _classification(y[valid[:, i], i], p[valid[:, i], i]) for i, a in enumerate(ASPECTS)
    }
    aspect_f1 = [v["macro_f1"] for v in per_aspect.values() if v["macro_f1"] is not None]
    full = valid.all(1)
    exact = (p.argmax(-1)[full] == y[full]).all(1)
    return {
        "reviews": len(y),
        "reviews_with_labels": int(valid.any(1).sum()),
        "decisions": int(valid.sum()),
        "decision_accuracy": pooled["accuracy"],
        "pooled_sentiment_macro_f1": pooled["macro_f1"],
        "mean_aspect_macro_f1": float(np.mean(aspect_f1)) if aspect_f1 else None,
        "fully_labeled_reviews": int(full.sum()),
        "exact_match_review_accuracy": float(exact.mean()) if len(exact) else None,
        "pooled": pooled,
        "per_aspect": per_aspect,
        "macro_f1_policy": "Fixed three-class macro; absent-class F1=0; no-label aspect excluded",
    }


def curves(y, p):
    y, p = validate_predictions(y, p)
    mask = y >= 0
    y, p = y[mask], p[mask]
    result = {}
    for i, name in enumerate(SENTIMENTS):
        binary = y == i
        if len(np.unique(binary)) < 2:
            result[name] = {"reason": "One-vs-rest labels have only one class"}
        else:
            fpr, tpr, _ = roc_curve(binary, p[:, i])
            precision, recall, _ = precision_recall_curve(binary, p[:, i])
            result[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
            }
    return result
