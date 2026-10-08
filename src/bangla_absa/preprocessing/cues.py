from collections import defaultdict

import numpy as np

from bangla_absa.data.schema import labels_array
from bangla_absa.preprocessing.normalize import tokens


class CueLexicon:
    """Train-only token/polarity co-occurrence reconstruction (IDs 1..3; 0=unknown)."""

    def __init__(self, mapping=None, min_count=2):
        self.mapping = dict(mapping or {})
        self.min_count = min_count

    def fit(self, records, role="train"):
        if role != "train":
            raise ValueError("Cue lexicon may only fit on train")
        counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))
        for row, labels in zip(records, labels_array(records)):
            for token in set(tokens(row["text"])):
                for label in labels[labels >= 0]:
                    counts[token][label] += 1
        self.mapping = {
            t: int(c.argmax()) + 1
            for t, c in counts.items()
            if c.sum() >= self.min_count and (c == c.max()).sum() == 1
        }
        return self

    def encode(self, text, offsets):
        # For each subword, inherit the cue ID of the containing full token.
        import re

        spans = [
            (m.start(), m.end(), self.mapping.get(m.group(), 0))
            for m in re.finditer(r"[\w\u0980-\u09ff]+|[^\w\s]", text)
        ]
        return [
            next((cue for left, right, cue in spans if left <= s < e <= right), 0)
            for s, e in offsets
        ]
