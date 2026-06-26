# Adaptive CPU Chip Test Reduction Using Reinforcement Learning

A graduate-level Reinforcement Learning project that learns an **adaptive
chip-testing policy**. Instead of running every manufacturing test on every CPU
chip, an RL agent decides — measurement by measurement — whether to keep
testing or to stop early and commit to a PASS/FAIL classification. The goal is
to **minimise testing cost while preserving classification quality**.

This project extends a prior Data Science effort that predicted CPU chip
failures from Stage-2 manufacturing test measurements using supervised learning
(XGBoost and Logistic Regression). Here those supervised models become
*baselines*, and the new contribution is a sequential, cost-aware decision
policy learned with reinforcement learning.

---

## 1. Project Overview

In high-volume semiconductor manufacturing, exhaustive testing of every die is
expensive. Much of that cost is wasted on chips whose pass/fail status is
already obvious after only a few measurements. **Adaptive test reduction**
treats testing as a sequential decision problem: continue testing only while
the extra information is worth its cost.

We frame this as a Markov Decision Process and train agents (tabular
Q-learning and a Deep Q-Network) to trade off test cost against the risk of
misclassification — in particular the very costly error of shipping a defective
chip (a *false pass*).

## 2. Problem Formulation

* **Goal:** classify each chip as PASS (good) or FAIL (defective).
* **Constraint / cost:** each additional test stage incurs a cost.
* **Trade-off:** stopping too early risks misclassification; testing too long
  wastes money. The optimal policy stops as soon as it is confident enough.

## 3. RL Formulation

### State
Each chip is one episode. The observation is a continuous vector containing:

* the (standardised) Stage-2 measurements, **engineered features**, and
  **wafer/location metadata**, with not-yet-revealed features masked;
* a binary **reveal mask** indicating which features are currently known;
* a scalar **testing-progress** signal.

Information is revealed **sequentially**: only the first test stage is visible
at episode start, and each `CONTINUE` reveals the next stage.

### Actions
A `Discrete(3)` action space:

| Action | Meaning |
| ------ | ------- |
| `0` `CONTINUE` | Pay `test_cost` and reveal the next test stage |
| `1` `STOP_PASS` | Stop and classify the chip as PASS |
| `2` `STOP_FAIL` | Stop and classify the chip as FAIL |

### Rewards (all configurable in `src/config.py`)

| Outcome | Reward |
| ------- | ------ |
| Continue testing | `-test_cost` |
| Correct PASS | `+20` |
| Correct FAIL | `+20` |
| False FAIL (good chip scrapped) | `-50` |
| False PASS (defect shipped) | `-100` |

### Episode
One chip = one episode. `CONTINUE` reveals more information and continues;
`STOP_PASS`/`STOP_FAIL` terminate the episode and the reward is computed from
the chip's true label.

## 4. Installation

Requires **Python 3.11+**.

```bash
cd project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> The project ships **no proprietary data**. On first run a realistic
> **synthetic** Stage-2 dataset is generated automatically into
> `data/raw/chip_tests.csv`. To use real data, place a CSV with the same
> schema (feature columns plus a binary `label` column) at that path.

### Generate a (small) synthetic dataset
Before integrating the real chip dataset, you can materialise a small synthetic
dataset and processed splits to exercise the whole pipeline end-to-end:

```bash
# Small dataset (1500 chips) + train/test splits for a fast end-to-end run.
python -m src.data.make_dataset --samples 1500 --features 20
```

To switch to the real dataset later, drop your CSV at
`data/raw/chip_tests.csv` (same schema) and skip this step — every downstream
script consumes the same processed splits regardless of data source.

### Prepare the real chip dataset
The original manufacturing data lives at `data/raw/base_data.csv`. Convert it
and rebuild processed splits with:

```bash
python -m src.data.prepare_real_data
```

This writes the canonical `data/raw/chip_tests.csv` (Stage-2/Stage-3
measurements, wafer metadata, binary label) and train/test CSVs under
`data/processed/`. Then train and evaluate as usual:

```bash
python -m src.training.train_qlearning --episodes 20000
python -m src.training.train_dqn --timesteps 200000
python -c "from src.evaluation.comparison import run_default_comparison; print(run_default_comparison()[0])"
```

## 5. Training Instructions

Generate/refresh the processed splits and train the agents:

```bash
# Tabular Q-learning
python -m src.training.train_qlearning --episodes 20000

# Deep Q-Network (Stable-Baselines3)
python -m src.training.train_dqn --timesteps 200000
```

Each script loads data, builds the environment, trains the agent, and saves:

* the trained model to `results/models/`,
* training statistics (`*_training.json`) to `results/metrics/`,
* a reward curve to `results/figures/`.

## 6. Evaluation Instructions

Run the full cross-method comparison (baselines + RL agents). This evaluates
every method on the held-out test set and writes tables and figures:

```python
from src.evaluation.comparison import run_default_comparison

table, metrics = run_default_comparison()
print(table)
```

Outputs:

* `results/metrics/comparison.csv` and `comparison.md` — the comparison table;
* `results/figures/` — confusion matrices, cost-savings bar chart, action
  distributions and a precision/recall comparison.

Reported metrics: **Accuracy, Precision, Recall, F1, False Pass Rate,
False Fail Rate, Average Reward, Average Test Cost, Cost Reduction %**, compared
across **Always Continue, Random, Rule-Based, Logistic Regression, XGBoost,
Q-Learning and DQN**.

### Notebooks
The end-to-end story is also told across five notebooks in `notebooks/`:

1. `01_data_exploration.ipynb` — dataset overview and distributions
2. `02_baseline_models.ipynb` — Logistic Regression and XGBoost baselines
3. `03_environment_validation.ipynb` — environment sanity checks
4. `04_rl_training.ipynb` — Q-learning and DQN training
5. `05_results_analysis.ipynb` — comparison tables and visualisations

## 7. Testing

```bash
cd project
pytest
```

## 8. Repository Structure

```
project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── conftest.py
├── data/
│   ├── raw/                 # raw (or synthetic) dataset
│   └── processed/           # train/test splits
├── notebooks/               # 01..05 analysis notebooks
├── src/
│   ├── config.py            # central configuration (rewards, costs, paths, seeds)
│   ├── data/                # loader, preprocessing, feature_engineering
│   ├── environment/         # ChipTestingEnv (Gymnasium)
│   ├── agents/              # random, rule_based, q_learning, dqn
│   ├── baselines/           # logistic, xgboost
│   ├── training/            # train_qlearning, train_dqn
│   ├── evaluation/          # metrics, evaluate, comparison
│   └── utils/               # plotting, helpers
├── results/
│   ├── figures/             # auto-saved plots
│   ├── metrics/             # tables and training stats
│   └── models/              # trained models
└── tests/                   # pytest unit tests
```

## 9. Design Notes

* **Configuration-first:** every reward, cost, path, split and hyperparameter
  lives in `src/config.py`. No absolute paths are hardcoded.
* **Reproducibility:** a single global seed (`Config.seed`) seeds Python,
  NumPy and PyTorch.
* **Code quality:** type hints and docstrings throughout, `logging` instead of
  `print`, a modular package layout, and unit tests.
* **Conservative default:** if an episode is truncated without a decision, the
  evaluator records FAIL — never silently shipping a chip of unknown quality.
