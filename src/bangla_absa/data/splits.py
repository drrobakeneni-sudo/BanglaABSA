"""Deterministic grouped multi-aspect stratification; identifiers stay local."""

from collections import defaultdict

import numpy as np

from bangla_absa.data.schema import labels_array, validate_records
from bangla_absa.preprocessing.normalize import normalize


def groups(records):
    """Union duplicate text and explicit groups, including transitive connections."""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(records):
        keys = [("text", normalize(row["text"]))]
        if row.get("group_id"):
            keys.append(("group", row["group_id"]))
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    grouped = defaultdict(list)
    for i in range(len(records)):
        grouped[root(i)].append(i)
    return list(grouped.values())


def make_splits(records, ratios=None, seed=17):
    records = sorted(validate_records(records), key=lambda r: r["review_id"])
    ratios = ratios or {"train": 0.7, "validation": 0.15, "test": 0.15}
    names = list(ratios)
    fractions = np.array(list(ratios.values()), dtype=float)
    if (fractions <= 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError("Split ratios must be positive and sum to one")
    clusters = groups(records)
    if len(clusters) < len(names):
        raise ValueError("Too few independent groups for nonempty partitions")
    rng = np.random.default_rng(seed)
    y = labels_array(records)
    vectors = []
    for indices in clusters:
        counts = np.array([((y[indices, a]) == c).sum() for a in range(3) for c in range(3)])
        vectors.append(np.r_[counts, len(indices)])
    total = np.sum(vectors, axis=0)
    target = fractions[:, None] * total
    allocation = np.zeros_like(target)
    ties = rng.random(len(clusters))
    order = sorted(range(len(clusters)), key=lambda i: (-len(clusters[i]), ties[i]))
    manifest, occupied = {}, set()
    for step, i in enumerate(order):
        candidates = list(range(len(names)))
        empty = [j for j in candidates if j not in occupied]
        if len(order) - step == len(empty):
            candidates = empty
        cost = []
        for j in candidates:
            before = (allocation[j] - target[j]) ** 2
            after = (allocation[j] + vectors[i] - target[j]) ** 2
            cost.append(np.sum((after - before) / np.maximum(target[j], 1)))
        j = candidates[int(np.argmin(cost))]
        allocation[j] += vectors[i]
        occupied.add(j)
        for k in clusters[i]:
            manifest[records[k]["review_id"]] = names[j]
    validate_splits(records, manifest, allowed=names)
    return manifest


def validate_splits(records, manifest, allowed=None):
    validate_records(records)
    if set(manifest) != {r["review_id"] for r in records}:
        raise ValueError("Split manifest must cover exactly the input IDs")
    if allowed and set(manifest.values()) - set(allowed):
        raise ValueError("Unexpected partition in manifest")
    if any(not isinstance(s, str) or not s for s in manifest.values()):
        raise ValueError("Invalid partition name")
    for indices in groups(records):
        if len({manifest[records[i]["review_id"]] for i in indices}) != 1:
            raise ValueError("Duplicate/group leakage across partitions")


def select(records, manifest, name):
    return [r for r in records if manifest[r["review_id"]] == name]


def deduplicate(records):
    """Keep one record per normalized text; refuse contradictory observed labels."""
    records = validate_records(records)
    # Preserve connections introduced by a removed duplicate's explicit group_id.
    group_ids = {}
    for indices in groups(records):
        group_id = "dedup:" + min(records[i]["review_id"] for i in indices)
        for i in indices:
            group_ids[records[i]["review_id"]] = group_id
    output, seen = [], {}
    for row in records:
        text = normalize(row["text"])
        if text not in seen:
            seen[text] = len(output)
            output.append(
                {
                    **row,
                    "labels": dict(row.get("labels", {})),
                    "group_id": group_ids[row["review_id"]],
                }
            )
        else:
            prior = output[seen[text]]
            for a, label in row.get("labels", {}).items():
                old = prior["labels"].get(a)
                if old is not None and label is not None and old != label:
                    raise ValueError("Conflicting duplicate annotations require adjudication")
                if label is not None:
                    prior["labels"][a] = label
    return output
