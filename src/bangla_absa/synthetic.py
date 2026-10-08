import itertools


def synthetic_records(count=135):
    """Newly authored fixtures. Numerical suffixes make examples distinct, not realistic."""
    sentiments = ("Negative", "Neutral", "Positive")
    aspects = ("Product", "Shop", "Transport")
    phrases = (
        ("পণ্য খারাপ", "পণ্য আছে", "পণ্য ভালো"),
        ("দোকান খারাপ", "দোকান আছে", "দোকান ভালো"),
        ("ডেলিভারি দেরি", "ডেলিভারি হয়েছে", "ডেলিভারি ভালো"),
    )
    rows = []
    values = itertools.cycle(itertools.product(range(3), repeat=3))
    for i in range(count):
        labels = next(values)
        text = " কিন্তু ".join(phrases[a][v] for a, v in enumerate(labels))
        if i % 5 == 0:
            text = text.replace("ভালো", "bhalo").replace("খারাপ", "kharap") + " 🙂"
        labels = dict(zip(aspects, [sentiments[v] for v in labels]))
        if i % 7 == 0:
            labels["Shop"] = None
        rows.append(
            {
                "review_id": f"synthetic-{i:04d}",
                "text": text + f" নমুনা {i}",
                "labels": labels,
                "provenance": {
                    "synthetic": True,
                    "purpose": "Authored software fixture; not collected human review",
                },
            }
        )
    return rows
