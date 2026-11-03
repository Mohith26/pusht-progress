import numpy as np
import torch

from progress.data import first_frame_index
from progress.features import build_inputs
from progress.metrics import auroc, spearman
from progress.model import ProgressHead, bin_centers, predict, progress_to_bin, train_head
from progress.qa import clean, qa_score, reversed_episode, truncated, wrong_start


def test_progress_to_bin_edges():
    p = torch.tensor([0.0, 0.049, 0.05, 0.5, 0.999, 1.0])
    assert progress_to_bin(p, 20).tolist() == [0, 0, 1, 10, 19, 19]


def test_dist_prediction_is_expected_bin_center():
    head = ProgressHead(4, loss="dist", k=4)
    logits = torch.tensor([[0.0, 0.0, 0.0, 100.0]])
    assert torch.allclose(head.predict(logits), bin_centers(4)[3:])


def test_build_inputs_delta_layout():
    feats = np.arange(12, dtype=np.float32).reshape(4, 3)
    first = np.array([0, 0, 2, 2])
    x = build_inputs(feats, first, "delta")
    assert x.shape == (4, 9)
    np.testing.assert_array_equal(x[1], np.concatenate([feats[0], feats[1], feats[1] - feats[0]]))
    np.testing.assert_array_equal(build_inputs(feats, first, "abs"), feats)


def test_first_frame_index():
    ep = np.array([3, 3, 3, 7, 7])
    fr = np.array([0, 1, 2, 0, 1])
    assert first_frame_index(ep, fr).tolist() == [0, 0, 0, 3, 3]


def test_spearman_and_auroc():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.5], [0.5]) == 0.5


def test_corruptions_shapes():
    seq = np.arange(20, dtype=np.float32).reshape(10, 2)
    other = seq + 100
    f0, ft = clean(seq, other)
    assert (f0 == seq[0]).all() and (ft == seq).all()
    f0, ft = reversed_episode(seq, other)
    assert (ft[0] == seq[-1]).all() and (f0 == seq[-1]).all()
    f0, ft = truncated(seq, other)
    assert len(ft) == 4
    f0, ft = wrong_start(seq, other)
    assert (f0 == other[0]).all()
    assert qa_score(np.ones(10)) == 0.0


def test_heads_learn_a_linear_signal():
    rng = np.random.default_rng(0)
    y = rng.uniform(size=2000)
    x = np.stack([y, rng.normal(size=2000)], axis=1).astype(np.float32)
    for loss in ("mse", "dist"):
        head = train_head(x, y, loss=loss, epochs=20)
        assert np.abs(predict(head, x) - y).mean() < 0.08
