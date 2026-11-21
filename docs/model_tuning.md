# Model tuning notes

**Training model:** `HistGradientBoostingClassifier` from scikit-learn 1.7.2
**Where it lives:** `src/spaceship_titanic/model.py` (the `classifier` step of `model_pipeline`)
**Date:** 2026-11-21

Only the model settings were tuned. The feature steps and data prep did not change.

## Settings

"Original" is what the model used before. It set no values, so these are the
scikit-learn defaults. "Suggested with AI" is the setting picked from the search below.

| Setting | Original | Suggested with AI | What it does |
|---|---|---|---|
| `learning_rate` | 0.1 | 0.026 | How big a step each new tree takes |
| `max_iter` | 100 | 592 | How many trees are built |
| `max_depth` | None (no limit) | 5 | How deep each tree can grow |
| `max_leaf_nodes` | 31 | 48 | Most end points each tree can have |
| `min_samples_leaf` | 20 | 38 | Fewest passengers allowed in one end point |
| `l2_regularization` | 0 | 0.027 | Penalty that keeps the model from memorizing |
| `random_state` | 42 | 42 | Unchanged |

## Results

All scores are accuracy. Training rows and holdout rows use the same 80/20 split
(seed 42) as `scripts/train.py`.

| Check | Original | Suggested with AI |
|---|---|---|
| 5-fold CV, seed 42 (used for the search) | 80.43% | 80.97% |
| 25 fresh folds (5-fold x 5 repeats, seed 2024) | 80.70% | 80.83% |
| Holdout set (1,739 passengers) | 81.37% | 81.89% |

## How the setting was picked

1. Ran a random search of 60 setting mixes with `RandomizedSearchCV`, scored with
   the existing 5-fold CV.
2. Took the top 5 and scored them again on 25 fresh folds they had not seen.
   Picking the best of 60 on the same folds makes the winner look better than it is.
3. Chose the one with the best score on the fresh folds.

## Honest read

The gain is small. On the fresh folds it is about 0.13 points, while scores move
about 1.2 points from fold to fold. So the new setting is at least as good as the
original, but it is not clearly better. New features would likely help more than
more tuning.

The saved model in `artifacts/` was not retrained as part of this change.
Run `python scripts/train.py` to rebuild it with the new settings.
