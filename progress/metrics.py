"""Metrics without a scipy/sklearn dependency."""

import numpy as np


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def auroc(scores_pos, scores_neg) -> float:
    """P(score of a positive > score of a negative), ties count half."""
    pos, neg = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    ranks = _rank(np.concatenate([pos, neg])) + 1
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))
