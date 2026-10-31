# Adaptive CPU Chip Test Reduction with Reinforcement Learning

Testing every CPU chip through every manufacturing stage is slow and expensive.
This project trains a reinforcement-learning agent to **stop testing early** when
the outcome is already clear: at each stage it decides whether to run the next
(costly) test or to commit now to a PASS/FAIL verdict. The goal is to **cut
testing cost while still catching defective chips** — in particular, never
shipping a bad one.

**Approach:** the chip-testing flow is modelled as a Gymnasium environment with
three sequential states (metadata → Stage-2 measurements → Stage-3). Two RL
agents — tabular **Q-learning** and a **DQN** (Stable-Baselines3) — are compared
against simple heuristics (Always-Continue, Random, Rule-Based) and
full-information supervised baselines (Logistic Regression, XGBoost).

## Key results

On the held-out `full_stage_v1` test set (286,640 chips, ~54% fail rate), the DQN
agent reaches the best cost/safety trade-off: it catches **100% of Stage-2
failures** while running far fewer tests. Supervised models are more accurate but
run *every* test (no cost savings); Q-learning saves the most cost but over-fails.

| Method | Accuracy | F1 (FAIL) | False Pass | Cost Reduction | % Stage-2 Fails Caught |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.91 | 0.91 | 0.16 | 0% | 100% |
| XGBoost | 0.91 | 0.91 | 0.15 | 0% | 100% |
| Q-Learning | 0.54 | 0.70 | 0.004 | **84%** | 99.6% |
| **DQN** | **0.76** | **0.81** | 0.073 | **65%** | **100%** |

Full tables and the write-up live in
[`results/runs/full_stage_v1/summary.md`](results/runs/full_stage_v1/summary.md)
(cross-run comparison in
[`final_comparison.csv`](results/runs/full_stage_v1/final_comparison.csv)).
Training curves and figures are regenerated into each run directory by the
commands below.

## Tech stack

Python 3.11+ · Gymnasium · Stable-Baselines3 (PyTorch) · scikit-learn · XGBoost ·
pandas / NumPy · Matplotlib · pytest.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare data

Place the raw dataset at `data/raw/full_stage_df.csv`, then build the splits:

```bash
python -m src.data.prepare_full_stage_data
```

## Train

```bash
# Tabular Q-learning
python -m src.training.train_qlearning --run-name full_stage_v1

# Deep Q-Network (Stable-Baselines3)
python -m src.training.train_dqn --run-name full_stage_v1
```

Models and training curves are written to `results/runs/full_stage_v1/`.

## Evaluate & compare

```bash
python -m src.evaluation.comparison --run-name full_stage_v1
```

This evaluates every method (heuristics, supervised baselines, Q-learning, DQN)
on the test set and writes comparison tables and figures to the run directory.

Optional experiments:

```bash
# DQN reward-sensitivity sweep -> results/reward_sweep/
python experiments/run_reward_sweep.py --config configs/reward_sweep.yaml

# Supervised baselines vs best DQN
python -m src.baselines.full_stage_supervised
```

## Test

```bash
pytest
```

## Repository structure

```
.
├── configs/reward_sweep.yaml   # reward-sweep experiment config
├── experiments/                # reward-sweep driver + plots
├── data/
│   ├── raw/                    # full_stage_df.csv (not committed)
│   └── processed/              # generated train/test splits
├── src/
│   ├── config.py               # rewards, costs, paths, hyperparameters
│   ├── data/                   # dataset loading & preparation
│   ├── environment/            # MultiStageChipTestingEnv
│   ├── agents/                 # random, rule_based, q_learning, dqn
│   ├── baselines/              # logistic, xgboost, supervised comparison
│   ├── training/               # training entry points
│   └── evaluation/             # rollouts, metrics, comparison
├── results/runs/full_stage_v1/ # saved metrics & summaries
└── tests/                      # pytest suite
```

## Design notes

- **Config-first:** rewards, costs, paths, splits and hyperparameters all live in
  `src/config.py`.
- **Reproducible:** a single global seed (`Config.seed`) seeds Python, NumPy and
  PyTorch.
- **Conservative default:** an episode truncated without a decision is scored as
  FAIL — the safe industrial choice of not shipping a chip of unknown quality.
