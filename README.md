# Adaptive CPU Chip Test Reduction Using Reinforcement Learning

A graduate-level Reinforcement Learning project that learns an **adaptive
chip-testing policy** on a realistic **multi-stage manufacturing dataset**.
Instead of running every test on every CPU chip, an RL agent decides — stage by
stage — whether to keep testing or to stop early and commit to a PASS/FAIL
classification. The goal is to **minimise testing cost while preserving
classification quality**, especially avoiding false passes (shipping defective
chips).

Supervised baselines (Logistic Regression and XGBoost) provide full-information
comparisons. The main RL agents are tabular Q-learning and a Deep Q-Network
(DQN).

---

## 1. Project Overview

The project uses the expanded **`full_stage_v1`** dataset (`data/raw/full_stage_df.csv`),
which includes chips that **failed at Stage 2** and never reached Stage 3. The
environment models three sequential states:

| State | Information visible | Actions |
| ----- | ------------------- | ------- |
| 0 | Metadata only | RUN_STAGE2 / STOP_PASS / STOP_FAIL |
| 1 | Metadata + Stage-2 measurements | RUN_STAGE3 / STOP_PASS / STOP_FAIL |
| 2 | All features (no real Stage-3 measurements in this dataset) | STOP_PASS / STOP_FAIL |

Reward profile **`full_stage_v1`** is defined in `src/config.py` and can be swept
via `configs/reward_sweep.yaml` (see `experiments/run_reward_sweep.py`).

## 2. Installation

Requires **Python 3.11+**.

```bash
cd project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Place the raw dataset at `data/raw/full_stage_df.csv`, then prepare processed splits:

```bash
python -m src.data.prepare_full_stage_data
```

## 3. Training

```bash
# Tabular Q-learning
python -m src.training.train_qlearning \
  --dataset full_stage_v1 --run-name full_stage_v1

# Deep Q-Network (Stable-Baselines3)
python -m src.training.train_dqn \
  --dataset full_stage_v1 --run-name full_stage_v1
```

Models and training curves are saved under `results/runs/full_stage_v1/`.

## 4. Evaluation & Comparison

```bash
python -m src.evaluation.comparison \
  --environment multi_stage \
  --dataset full_stage_v1 \
  --run-name full_stage_v1
```

This evaluates Always Continue, Random, Rule-Based, Logistic Regression, XGBoost,
Q-Learning and DQN on the held-out test set and writes comparison tables and
figures to the run directory.

### Reward-sensitivity sweep (DQN)

```bash
python experiments/run_reward_sweep.py --config configs/reward_sweep.yaml
```

Outputs go to `results/reward_sweep/`.

### Supervised baselines vs best DQN

```bash
python -m src.baselines.full_stage_supervised
```

## 5. Testing

```bash
cd project
pytest
```

## 6. Repository Structure

```
project/
├── README.md
├── requirements.txt
├── configs/reward_sweep.yaml
├── experiments/             # reward sweep scripts
├── data/
│   ├── raw/full_stage_df.csv
│   └── processed/full_stage_v1/
├── src/
│   ├── config.py
│   ├── data/                # full_stage_loader, prepare_full_stage_data
│   ├── environment/         # MultiStageChipTestingEnv
│   ├── agents/              # random, rule_based, q_learning, dqn
│   ├── baselines/           # logistic, xgboost, full_stage_supervised
│   ├── training/
│   └── evaluation/
├── results/
│   ├── runs/full_stage_v1/
│   └── reward_sweep/
└── tests/
```

## 7. Design Notes

* **Configuration-first:** rewards, costs, paths, splits and hyperparameters
  live in `src/config.py`.
* **Reproducibility:** a single global seed (`Config.seed`) seeds Python, NumPy
  and PyTorch.
* **Conservative default:** truncated episodes without a decision are recorded
  as FAIL in evaluation.
