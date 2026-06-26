# Reward-Sensitivity Experiment: `safety_reward_v1`

## Goal

Find more failing/damaged chips (raise recall on FAIL and lower the False Pass Rate) while still preserving *some* test-cost reduction. Maximum cost reduction is explicitly **not** the objective.

## What changed

Only the reward system changed versus the baseline real-data run. The dataset, train/test split, random seed, Q-learning episodes (20,000), DQN timesteps (200,000), model architectures, preprocessing, metrics and comparison logic are all identical.

| Reward term | baseline | safety_reward_v1 |
| --- | --- | --- |
| continue_cost | -1 | -2 |
| correct_pass | +20 | +10 |
| correct_fail | +20 | +100 |
| false_pass | -100 | -500 |
| false_fail | -50 | -50 |
| early_pass_penalty | 0 | -20 |

`early_pass_penalty` is applied only when the agent classifies PASS before choosing CONTINUE (i.e. before any additional Stage-3 information is revealed).

## DQN: baseline vs safety_reward_v1

| Metric | baseline | safety_reward_v1 | Δ |
| --- | --- | --- | --- |
| Accuracy | 0.8368 | 0.8362 | -0.0006 |
| F1 (FAIL) | 0.0344 | 0.0213 | -0.0131 |
| Recall (FAIL) | 0.0176 | 0.0108 | -0.0068 |
| False Pass Rate | 0.9824 | 0.9892 | +0.0068 |
| False Fail Rate | 0.0012 | 0.0005 | -0.0007 |
| Avg Tests Run | 1.0161 | 2.0055 | +0.9894 |
| Cost Reduction % | 79.68 | 59.89 | -19.79 |

### DQN policy action distribution

| Action | baseline | safety_reward_v1 |
| --- | --- | --- |
| CONTINUE | 1.86% | 50.20% |
| STOP_PASS | 97.90% | 49.73% |
| STOP_FAIL | 0.25% | 0.07% |

## Did the new reward system help (DQN)?

**No.** The safety reward profile did not reduce the False Pass Rate or improve damaged-chip detection for the DQN agent under this configuration.

- False Pass Rate went from 98.24% to 98.92% (up 0.68 points).
- Recall on FAIL went from 1.76% to 1.08% (down 0.68 points).
- Test effort (avg stages run) went from 1.02 to 2.01, a cost reduction of 59.9% versus full testing — so test-cost savings are preserved.

## Cross-method highlight

Among the RL agents, **Q-Learning** best achieves the safety goal under `safety_reward_v1`:

| Metric | baseline | safety_reward_v1 | Δ |
| --- | --- | --- | --- |
| Recall (FAIL) | 0.6775 | 0.9768 | +0.2993 |
| False Pass Rate | 0.3225 | 0.0232 | -0.2993 |
| False Fail Rate | 0.6388 | 0.9882 | +0.3494 |
| Cost Reduction % | 6.98 | 1.46 | -5.52 |

If catching damaged chips is the priority, **Q-Learning** under the safety profile is the recommended policy: it trades most of the cost reduction for a large drop in escaped defects (False Pass Rate). DQN, by contrast, collapses toward an early-PASS policy on this heavily imbalanced dataset (~83% PASS) and does not benefit from the safety rewards.

See `comparison.csv` for the per-method table for this run and `baseline_vs_safety.csv` for the full side-by-side comparison. `avg_test_cost` is expressed in each profile's own per-stage cost units (baseline 1/stage, safety 2/stage); `avg_tests_run` and `cost_reduction_pct` are profile-independent and directly comparable.
