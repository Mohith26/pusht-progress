# pusht-progress

A small reproduction of the progress estimator from Haichao Zhang's post
[Can Robots Tell If They're Making Progress?](https://haichao-zhang.github.io/)
on a much smaller setup: the LeRobot Push-T image dataset and a frozen ResNet-18
instead of a VLM backbone on Bridge.

The two ideas from the post that I wanted to test:

1. **Delta-aware input.** Feed the head `[f0, ft, ft - f0]` (start frame, current
   frame, change) instead of the current frame alone.
2. **Distributional loss.** Discretize progress into K bins and train with
   cross-entropy, instead of regressing a single number with MSE.

I also tried the data-QA use from the post: score each episode by how far its
last few predicted progress values are from 1, and see whether that separates
clean demos from synthetically broken ones.

## Setup

- Data: [`lerobot/pusht_image`](https://huggingface.co/datasets/lerobot/pusht_image),
  206 episodes, 25,650 frames (96x96). Downloaded automatically by `run.py`.
- Features: torchvision ResNet-18 (ImageNet weights), frozen, images resized to
  160x160, 512-d pooled features. Extracted once and cached (about 3.5 minutes on CPU).
- Target: time-normalized progress `frame / (episode_length - 1)`. The post does
  not say exactly how its labels are built; this is the simplest version.
- Head: one hidden layer MLP (256, GELU). MSE head outputs a sigmoid; the
  distributional head outputs K=20 logits and predicts the expected bin center.
- Evaluation: 5-fold cross-validation split by episode, so test episodes are
  never seen in training. One seed. Training all 4 configs x 5 folds takes about
  2.5 minutes on CPU.
- Data QA: for each held-out episode I build three corrupted versions:
  `reversed` (played backwards), `truncated` (only the first 40% of frames) and
  `wrong_start` (paired with another episode's first frame). The score is
  `1 - mean(predicted progress over the last 5 frames)`, and I report AUROC of
  corrupted vs clean.

## Results

From `results.json` (mean over 5 folds):

| input + loss | MAE | per-episode Spearman | QA AUROC reversed | truncated | wrong start |
|---|---|---|---|---|---|
| abs + mse | 0.102 (+/- 0.004) | 0.925 | 1.000 | 1.000 | 0.500 |
| abs + dist | 0.099 (+/- 0.005) | 0.926 | 1.000 | 1.000 | 0.500 |
| delta + mse | 0.096 (+/- 0.006) | 0.950 | 1.000 | 1.000 | 0.495 |
| delta + dist | 0.095 (+/- 0.003) | 0.926 | 1.000 | 0.998 | 0.492 |

What I take from it:

- The delta input helps, but modestly here. delta + mse has the best
  per-episode rank correlation in every one of the 5 folds (0.950 vs 0.925 mean),
  and delta + dist has lower MAE than abs + mse in all 5 folds.
- The distributional loss gives a small MAE gain over MSE with the same input,
  but with the delta input it lost the Spearman gain. With one seed I would not
  read much into the gap between the two delta rows.
- Reversed and truncated episodes are trivially caught (AUROC about 1.0).
- Wrong start frames are not caught at all (AUROC about 0.5). That makes sense
  for Push-T: the goal pose is fixed and every episode starts from a random T
  pose, so a different start frame does not change what "done" looks like at
  the end. For the abs model it is 0.5 by construction since it never sees f0.
  The kind of error the post shows on Bridge (wrong instruction, wrong initial
  scene) needs a task-conditioned model and a dataset where the start defines
  the task, which this setup does not have.

## Limits

- Push-T is a single 2D task, so this says nothing about cross-embodiment
  transfer, which is the main motivation in the post.
- ResNet-18 features are much weaker than a VLM, and the head never sees the
  language instruction.
- Time-normalized labels assume every demo progresses at a steady rate; they
  are a stand-in for real progress.
- Single seed, small heads, no tuning.

## Run it

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py            # downloads data, extracts features, writes results.json
.venv/bin/python -m pytest -q tests
```

Code: `progress/data.py` (loading), `progress/features.py` (ResNet-18 and the
`[f0, ft, ft - f0]` input), `progress/model.py` (MSE and K-bin heads),
`progress/qa.py` (corruptions and QA score), `progress/metrics.py` (Spearman and
AUROC without scipy).
