import numpy as np


def nominal_agreement(ratings):
    """Rows=units, columns=annotators; null/NaN ratings are missing.

    Raw agreement is agreeing unordered observed pairs / all observed pairs.
    Alpha uses unit-normalized coincidence counts, excluding singleton units.
    """
    counts = {}
    observed_disagreement, paired_total, agreeing, pairs = 0.0, 0, 0, 0
    for row in ratings:
        valid = [
            v
            for v in row
            if v is not None and not (isinstance(v, (float, np.floating)) and np.isnan(v))
        ]
        m = len(valid)
        if m < 2:
            continue
        unit = {v: valid.count(v) for v in set(valid)}
        same_ordered = sum(n * (n - 1) for n in unit.values())
        observed_disagreement += (m * (m - 1) - same_ordered) / (m - 1)
        paired_total += m
        pairs += m * (m - 1) // 2
        agreeing += same_ordered // 2
        for value, count in unit.items():
            counts[value] = counts.get(value, 0) + count
    raw = agreeing / pairs if pairs else None
    if paired_total < 2:
        return {
            "alpha": None,
            "raw_pair_agreement": raw,
            "pairs": pairs,
            "reason": "No pairable units",
        }
    expected = (paired_total**2 - sum(n**2 for n in counts.values())) / (paired_total - 1)
    return {
        "alpha": 1 - observed_disagreement / expected if expected else None,
        "raw_pair_agreement": raw,
        "pairs": pairs,
        "reason": None if expected else "Expected disagreement is zero",
    }
