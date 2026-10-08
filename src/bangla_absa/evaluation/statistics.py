import numpy as np
from scipy import stats


def summarize_seeds(seed_values):
    if len(seed_values) < 2:
        raise ValueError("Sample SD and across-seed CI require at least two runs")
    values = np.array(list(seed_values.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Seed metrics must be finite")
    mean, sd = values.mean(), values.std(ddof=1)
    half = stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))
    return {
        "per_seed": seed_values,
        "mean": float(mean),
        "sample_sd": float(sd),
        "seed_mean_ci95": [float(mean - half), float(mean + half)],
    }


def paired_seed_test(left, right):
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("At least two matched seeds required")
    keys = sorted(left)
    diff = np.array([left[k] - right[k] for k in keys], dtype=float)
    if not np.isfinite(diff).all():
        raise ValueError("Metrics must be finite")
    if np.std(diff, ddof=1) == 0:
        return {"t": None, "p": None, "reason": "Zero variance of paired differences"}
    result = stats.ttest_rel([left[k] for k in keys], [right[k] for k in keys])
    return {"t": float(result.statistic), "p": float(result.pvalue), "df": len(keys) - 1}


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be in [0,1]")
    order = np.argsort(p)
    corrected = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    result = np.empty_like(p)
    result[order] = corrected
    return result.tolist()


def mcnemar(y, left, right):
    mask = np.asarray(y) >= 0
    a = np.asarray(left)[mask] == np.asarray(y)[mask]
    b = np.asarray(right)[mask] == np.asarray(y)[mask]
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    total = n10 + n01
    return {
        "left_only_correct": n10,
        "right_only_correct": n01,
        "exact_binomial_p": float(stats.binomtest(n10, total, 0.5).pvalue) if total else 1.0,
        "warning": "Decision-level test assumes independence; also report review-cluster bootstrap",
    }


def cluster_bootstrap(y, left, right=None, seed=17, repetitions=1000):
    y, left = np.asarray(y), np.asarray(left)
    if y.shape != left.shape or y.ndim != 2 or not (y >= 0).any() or repetitions < 2:
        raise ValueError(
            "Expected aligned review/aspect observations with labels and >=2 replicates"
        )
    if right is not None and np.shape(right) != y.shape:
        raise ValueError("Paired predictions must align")
    rng = np.random.default_rng(seed)
    values = []
    # Units with no observations carry no metric information; retain labeled review clusters.
    available = np.flatnonzero((y >= 0).any(1))
    for _ in range(repetitions):
        indices = rng.choice(available, len(available), replace=True)
        labels, a = y[indices], left[indices]
        valid = labels >= 0
        score = (a[valid] == labels[valid]).mean()
        if right is not None:
            score -= (np.asarray(right)[indices][valid] == labels[valid]).mean()
        values.append(float(score))
    return {
        "review_cluster_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "repetitions": repetitions,
        "seed": seed,
        "paired_difference": right is not None,
    }
