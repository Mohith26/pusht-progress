"""Synthetic corruptions for the data-QA check, built from per-episode features.

Each corruption returns (f0, ft) arrays for one episode as the progress head
would see them. The QA score of an episode is 1 - mean predicted progress over
its last `tail` frames: a clean, finished demo should end near 1.
"""

import numpy as np


def clean(seq: np.ndarray, other: np.ndarray):
    return np.repeat(seq[:1], len(seq), axis=0), seq


def reversed_episode(seq: np.ndarray, other: np.ndarray):
    rev = seq[::-1]
    return np.repeat(rev[:1], len(rev), axis=0), rev


def truncated(seq: np.ndarray, other: np.ndarray, keep: float = 0.4):
    cut = seq[: max(2, int(len(seq) * keep))]
    return np.repeat(cut[:1], len(cut), axis=0), cut


def wrong_start(seq: np.ndarray, other: np.ndarray):
    """Pair this episode with the first frame of a different episode."""
    return np.repeat(other[:1], len(seq), axis=0), seq


CORRUPTIONS = {"reversed": reversed_episode, "truncated": truncated, "wrong_start": wrong_start}


def episode_inputs(f0: np.ndarray, ft: np.ndarray, mode: str) -> np.ndarray:
    if mode == "abs":
        return ft
    return np.concatenate([f0, ft, ft - f0], axis=1)


def qa_score(pred: np.ndarray, tail: int = 5) -> float:
    return float(1.0 - pred[-tail:].mean())
