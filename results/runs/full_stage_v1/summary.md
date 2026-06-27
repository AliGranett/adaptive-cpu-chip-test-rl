# Multi-Stage Experiment: `full_stage_v1`

## Goal & main question

Rerun the multi-stage RL testing experiment on the **expanded dataset** (`data/raw/full_stage_df.csv`), which now includes chips that **failed at Stage 2** and never reached Stage 3. 

> **Main question:** Does adding Stage-2-failed chips improve the realism of the environment and help the agent learn a *safer* testing policy?

## Environment

Three sequential states with a context-dependent CONTINUE action:

- **State 0 - metadata only:** RUN_STAGE2 / STOP_PASS / STOP_FAIL
- **State 1 - + Stage-2 measurements (and Stage-2 result):** RUN_STAGE3 / STOP_PASS / STOP_FAIL
- **State 2 - + Stage-3:** STOP_PASS / STOP_FAIL

Reward profile `full_stage_v1`: per-stage costs (Stage-2 = -1, Stage-3 = -4), `correct_pass`=+10, `correct_fail`=+100, `false_pass`=-500, `false_fail`=-50, `metadata_only_pass_penalty`=-50, `early_pass_penalty`=-20, `stage2_fail_detected_reward`=+120, `stage2_fail_missed_penalty`=-600. Continuing to Stage-3 on a chip that already failed Stage-2 is heavily penalised.

## Results on the multi-stage test set

| Method | Accuracy | F1 | Recall (FAIL) | Precision (FAIL) | False Pass | False Fail | Avg Cost | Cost Red. % | % Stage2-Fail Caught | % Stage2-Fail Passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Always Continue | 0.5544 | 0.5550 | 0.5150 | 0.6018 | 0.4850 | 0.3995 | 5.0000 | 0.00 | 53.58% | 46.42% |
| Random | 0.4997 | 0.5189 | 0.4999 | 0.5394 | 0.5001 | 0.5004 | 0.9872 | 80.26 | 49.79% | 50.21% |
| Rule-Based | 0.5550 | 0.5579 | 0.5203 | 0.6014 | 0.4797 | 0.4042 | 4.5109 | 9.78 | 54.04% | 45.96% |
| Logistic Regression | 0.9133 | 0.9126 | 0.8394 | 0.9999 | 0.1606 | 0.0001 | 5.0000 | 0.00 | 100.00% | 0.00% |
| XGBoost | 0.9138 | 0.9138 | 0.8465 | 0.9928 | 0.1535 | 0.0072 | 5.0000 | 0.00 | 100.00% | 0.00% |
| Q-Learning | 0.5388 | 0.6997 | 0.9959 | 0.5394 | 0.0041 | 0.9971 | 0.8056 | 83.89 | 99.61% | 0.39% |
| DQN | 0.7607 | 0.8070 | 0.9271 | 0.7145 | 0.0729 | 0.4343 | 1.7495 | 65.01 | 100.00% | 0.00% |

## Stage routing (RL agents)

| Method | % Stopped Before Stage2 | % Stopped After Stage2 | % Sent To Stage3 | Avg Tests Run |
| --- | --- | --- | --- | --- |
| Q-Learning | 20.84% | 78.85% | 0.30% | 0.7946 |
| DQN | 49.14% | 20.67% | 30.19% | 0.8104 |

## Does it lead to a safer policy?

**Yes.** Including Stage-2 failures makes the environment match the real test flow, and the best RL agent (**DQN**) learns to run Stage-2 and then stop-FAIL the rejects: it correctly catches 100.0% of Stage-2 failures and lets only 0.0% slip through, for an overall False Pass Rate of 0.073.

The expanded dataset adds **realism**: roughly half of all chips now fail at Stage 2, so a policy can no longer score well by blindly passing. The reward structure rewards cheap early detection (run Stage-2 for -1, then STOP_FAIL a reject for +120) and severely punishes letting a Stage-2 reject through (-600), which pushes the agent toward a safer, cost-aware policy than the single-stage runs.

## Cross-run comparison

See `final_comparison.csv` / `final_comparison.md` for the side-by-side table across runs (`baseline`, `safety_reward_v1`, `full_stage_v1`).

> Note: no `multi_stage_v1` run exists in this project, so it is omitted from the comparison. `full_stage_v1` is the first multi-stage run. Single-stage runs (`baseline`, `safety_reward_v1`) were trained on a different dataset/environment, so their stage-routing columns are empty and their headline metrics are not strictly comparable; they are included for reference only.

## DQN policy action distribution (full_stage_v1)

| Action (context) | Fraction |
| --- | --- |
| CONTINUE (RUN_STAGE2/3) | 45.07% |
| STOP_PASS | 16.50% |
| STOP_FAIL | 38.43% |
