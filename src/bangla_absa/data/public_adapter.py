"""Explicit mapping for a locally obtained related dataset, never the 200k corpus."""

from bangla_absa import ASPECTS
from bangla_absa.data.schema import validate_records

MAPPING = {"product quality": "Product", "seller service": "Shop", "delivery": "Transport"}


def adapt_long_rows(
    rows,
    id_column="review_id",
    text_column="text",
    aspect_column="aspect",
    sentiment_column="sentiment",
    conflict="error",
):
    """User maps the downloaded columns into this long-form interface.

    No claim about uninspected upstream column names; price/packaging are excluded.
    conflict=missing preserves contradictory labels as missing and counts them.
    """
    if conflict not in {"error", "missing"}:
        raise ValueError("Conflict policy must be error or missing")
    result, observed, conflicts, excluded = {}, {}, set(), 0
    for row in rows:
        rid, text = str(row[id_column]), row[text_column]
        aspect = MAPPING.get(row[aspect_column].strip().lower())
        if rid in result and result[rid]["text"] != text:
            raise ValueError("One upstream ID has inconsistent text")
        result.setdefault(
            rid,
            {
                "review_id": rid,
                "text": text,
                "labels": {a: None for a in ASPECTS},
                "provenance": {"dataset": "10.17632/n4n5y34p3s.1", "adapter": "long-form-v1"},
            },
        )
        if aspect is None:
            excluded += 1
            continue
        sentiment = row[sentiment_column].strip().capitalize()
        key = (rid, aspect)
        observed.setdefault(key, set()).add(sentiment)
        if len(observed[key]) > 1:
            conflicts.add(key)
            if conflict == "error":
                raise ValueError("Conflicting aspect labels: adjudicate or select missing policy")
            result[rid]["labels"][aspect] = None
        elif key not in conflicts:
            result[rid]["labels"][aspect] = sentiment
    return validate_records(result.values()), {
        "excluded_aspect_rows": excluded,
        "conflicting_decisions": len(conflicts),
    }
