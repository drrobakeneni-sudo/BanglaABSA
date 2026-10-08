import numpy as np

from bangla_absa import ASPECTS, SENTIMENTS
from bangla_absa.evaluation.metrics import validate_predictions
from bangla_absa.io import read_json, write_json


def save_predictions(path, ids, y, p, split, seed, checkpoint_id):
    y, p = validate_predictions(y, p)
    if len(ids) != len(y) or len(ids) != len(set(ids)):
        raise ValueError("Prediction IDs must be unique and aligned")
    rows = [
        {
            "review_id": rid,
            "aspect": aspect,
            "observed": bool(y[i, a] >= 0),
            "gold": SENTIMENTS[y[i, a]] if y[i, a] >= 0 else None,
            "predicted": SENTIMENTS[int(p[i, a].argmax())],
            "probabilities": p[i, a].tolist(),
        }
        for i, rid in enumerate(ids)
        for a, aspect in enumerate(ASPECTS)
    ]
    write_json(
        path,
        {
            "aspect_order": list(ASPECTS),
            "sentiment_order": list(SENTIMENTS),
            "split": split,
            "seed": seed,
            "checkpoint_id": checkpoint_id,
            "rows": rows,
        },
    )


def load_predictions(path):
    data = read_json(path)
    if data["aspect_order"] != list(ASPECTS) or data["sentiment_order"] != list(SENTIMENTS):
        raise ValueError("Prediction class order mismatch")
    ids = list(dict.fromkeys(r["review_id"] for r in data["rows"]))
    index = {rid: i for i, rid in enumerate(ids)}
    y = np.full((len(ids), 3), -1)
    p = np.zeros((len(ids), 3, 3))
    seen = set()
    for row in data["rows"]:
        key = (row["review_id"], row["aspect"])
        if key in seen:
            raise ValueError("Duplicate prediction decision")
        seen.add(key)
        i, a = index[key[0]], ASPECTS.index(key[1])
        y[i, a] = SENTIMENTS.index(row["gold"]) if row["gold"] is not None else -1
        if row["observed"] != bool(y[i, a] >= 0):
            raise ValueError("Observed mask disagrees with gold label")
        p[i, a] = row["probabilities"]
        if row["predicted"] != SENTIMENTS[int(p[i, a].argmax())]:
            raise ValueError("Stored predicted label disagrees with probabilities")
    validate_predictions(y, p)
    return ids, y, p, {k: v for k, v in data.items() if k != "rows"}


def align_predictions(left, right):
    ids, y, p, meta = left
    ids2, y2, p2, meta2 = right
    if len(ids) != len(set(ids)) or len(ids2) != len(set(ids2)) or set(ids) != set(ids2):
        raise ValueError("Model predictions do not have matched unique IDs")
    if meta.get("split") != meta2.get("split") or meta.get("seed") != meta2.get("seed"):
        raise ValueError("Model predictions must have matched split and seed")
    lookup = {rid: i for i, rid in enumerate(ids2)}
    order = [lookup[rid] for rid in ids]
    if not np.array_equal(y, y2[order]):
        raise ValueError("Model gold labels or masks disagree")
    return ids, y, p, p2[order]
