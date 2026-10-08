import numpy as np

from bangla_absa import ASPECTS, SENTIMENTS


def validate_records(records):
    """Validate without displaying text or identifiers in error messages."""
    records = list(records)
    seen = set()
    allowed = {"review_id", "text", "labels", "source_platform", "group_id", "date", "provenance"}
    for i, row in enumerate(records):
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError(f"Row {i}: expected a record with canonical fields")
        rid, text = row.get("review_id"), row.get("text")
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            raise ValueError(f"Row {i}: invalid or duplicate review_id")
        seen.add(rid)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Row {i}: text must be nonempty")
        for key in ("source_platform", "group_id", "date"):
            if key in row and row[key] is not None and not isinstance(row[key], str):
                raise ValueError(f"Row {i}: invalid {key}")
        labels = row.get("labels", {})
        if not isinstance(labels, dict) or set(labels) - set(ASPECTS):
            raise ValueError(f"Row {i}: invalid aspects")
        if any(v is not None and v not in SENTIMENTS for v in labels.values()):
            raise ValueError(f"Row {i}: invalid sentiment")
        if "provenance" in row and not isinstance(row["provenance"], dict):
            raise ValueError(f"Row {i}: provenance must be an object")
    return records


def labels_array(records):
    return np.array(
        [
            [
                SENTIMENTS.index(r.get("labels", {}).get(a))
                if r.get("labels", {}).get(a) is not None
                else -1
                for a in ASPECTS
            ]
            for r in records
        ],
        dtype=np.int64,
    ).reshape(-1, 3)
