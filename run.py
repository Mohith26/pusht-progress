"""Progress estimation on LeRobot Push-T: abs vs delta inputs, MSE vs distributional loss.

Usage:
    python run.py --data pusht.parquet [--folds 5] [--epochs 15]
Writes results.json and prints a markdown table.
"""

import argparse
import json
import os
import time
import urllib.request

import numpy as np

from progress.data import PUSHT_URL, first_frame_index, load_pusht
from progress.features import build_inputs, resnet18_features
from progress.metrics import auroc, spearman
from progress.model import predict, train_head
from progress.qa import CORRUPTIONS, clean, episode_inputs, qa_score

CONFIGS = [("abs", "mse"), ("abs", "dist"), ("delta", "mse"), ("delta", "dist")]


def get_features(path: str, cache: str):
    if os.path.exists(cache):
        z = np.load(cache)
        return z["feats"], z["episode"], z["frame"], z["length"]
    eps = load_pusht(path)
    t = time.time()
    feats = resnet18_features(eps.images)
    print(f"extracted {feats.shape} features in {time.time() - t:.0f}s")
    np.savez(cache, feats=feats, episode=eps.episode, frame=eps.frame, length=eps.length)
    return feats, eps.episode, eps.frame, eps.length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pusht.parquet")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        urllib.request.urlretrieve(PUSHT_URL, args.data)
    feats, episode, frame, length = get_features(args.data, "features_resnet18.npz")
    y = frame / np.maximum(length - 1, 1)
    first = first_frame_index(episode, frame)

    rng = np.random.default_rng(args.seed)
    ep_ids = rng.permutation(np.unique(episode))
    folds = np.array_split(ep_ids, args.folds)

    rows = {f"{m}+{l}": {"mae": [], "spearman": [], "qa_auroc": {c: [] for c in CORRUPTIONS}} for m, l in CONFIGS}
    for k, test_eps in enumerate(folds):
        test = np.isin(episode, test_eps)
        for mode, loss in CONFIGS:
            key = f"{mode}+{loss}"
            x = build_inputs(feats, first, mode)
            head = train_head(x[~test], y[~test], loss=loss, k=args.bins, epochs=args.epochs, seed=args.seed + k)
            pred = predict(head, x[test])
            rows[key]["mae"].append(float(np.abs(pred - y[test]).mean()))
            ep_test = episode[test]
            rows[key]["spearman"].append(
                float(np.mean([spearman(pred[ep_test == e], y[test][ep_test == e]) for e in test_eps]))
            )
            # data-QA: clean held-out episodes vs synthetic corruptions of the same episodes
            seqs = [feats[(episode == e)][np.argsort(frame[episode == e])] for e in test_eps]
            others = seqs[1:] + seqs[:1]

            def score(fn):
                out = []
                for s, o in zip(seqs, others):
                    f0, ft = fn(s, o)
                    out.append(qa_score(predict(head, episode_inputs(f0, ft, mode))))
                return out

            clean_scores = score(clean)
            for name, fn in CORRUPTIONS.items():
                rows[key]["qa_auroc"][name].append(auroc(score(fn), clean_scores))
        print(f"fold {k + 1}/{args.folds} done")

    summary = {}
    for key, r in rows.items():
        summary[key] = {
            "mae": round(float(np.mean(r["mae"])), 4),
            "mae_std": round(float(np.std(r["mae"])), 4),
            "spearman": round(float(np.mean(r["spearman"])), 4),
            "qa_auroc": {c: round(float(np.mean(v)), 3) for c, v in r["qa_auroc"].items()},
        }
    result = {"args": vars(args), "episodes": int(len(ep_ids)), "frames": int(len(y)), "summary": summary, "folds": rows}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print("\n| input + loss | MAE | per-episode Spearman | QA AUROC reversed | truncated | wrong start |")
    print("|---|---|---|---|---|---|")
    for key, s in summary.items():
        q = s["qa_auroc"]
        print(
            f"| {key} | {s['mae']:.3f} (+/- {s['mae_std']:.3f}) | {s['spearman']:.3f} | "
            f"{q['reversed']:.3f} | {q['truncated']:.3f} | {q['wrong_start']:.3f} |"
        )


if __name__ == "__main__":
    main()
