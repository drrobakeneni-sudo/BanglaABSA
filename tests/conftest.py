import itertools

import pytest
import torch


@pytest.fixture(autouse=True)
def limit_threads():
    torch.set_num_threads(2)


@pytest.fixture
def records():
    labels = ["Negative", "Neutral", "Positive"]
    aspects = ["Product", "Shop", "Transport"]
    phrases = [
        ("পণ্য খারাপ", "পণ্য আছে", "পণ্য ভালো"),
        ("দোকান খারাপ", "দোকান আছে", "দোকান ভালো"),
        ("ডেলিভারি দেরি", "ডেলিভারি হয়েছে", "ডেলিভারি ভালো"),
    ]
    return [
        {
            "review_id": f"synthetic-{i}",
            "text": " কিন্তু ".join(phrases[a][v] for a, v in enumerate(values)) + f" নমুনা {i}",
            "labels": dict(zip(aspects, [labels[v] for v in values])),
            "provenance": {"synthetic": True},
        }
        for i, values in enumerate(
            itertools.islice(itertools.cycle(itertools.product(range(3), repeat=3)), 108)
        )
    ]
