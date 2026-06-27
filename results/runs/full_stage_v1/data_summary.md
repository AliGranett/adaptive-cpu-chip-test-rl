# Data Summary: `full_stage_v1`

Expanded multi-stage dataset that **includes chips that failed during Stage-2 testing** (previously excluded). Source file: `data/raw/full_stage_df.csv`.

## Dataset composition

| Quantity | Value |
| --- | --- |
| Number of chips | 286,640 |
| PASS (label 0) | 131,954 |
| FAIL (label 1) | 154,686 |
| Fail rate | 53.97% |
| Stage-2 failures | 129,676 |
| Stage-3 / final failures | 25,010 |
| Chips with missing Stage-3 data | 135,182 |
| Stage-2-pass chips with no final result (ambiguous -> PASS) | 5,506 |

## Label logic

- `FinalRes_Stage2 == fail` -> **FAIL** (even with no Stage-3 data).
- `FinalRes_Stage2 == pass` and `final_res == fail` -> **FAIL**.
- `FinalRes_Stage2 == pass` and `final_res == pass` -> **PASS**.
- `FinalRes_Stage2 == pass` and `final_res` missing -> **PASS** (passed the only completed stage; no failure recorded).

## Feature columns used at each state

- **State 0 (metadata only):** meta_lot, meta_wafer, meta_x, meta_y, meta_radial
- **State 1 (after Stage-2):** s2_power, s2_speedh, s2_speedl, s2_duration, stage2_fail_flag
- **State 2 (after Stage-3):** (none - dataset has no Stage-3 measurements)

## Leakage handling

Excluded outcome-encoding columns (they perfectly encode the Stage-2 result and would leak the label at State 0): Category_code_, Test_code.

`Test_Duration` is treated as a **Stage-2 feature** (a by-product of running Stage-2), available only at State 1. The Stage-2 result itself is exposed at State 1 via the `stage2_fail_flag` feature.

> **Note:** this dataset contains **no Stage-3 measurement columns**. The multi-stage environment still supports the Stage-3 step, but it reveals no real measurements here; Stage-3 features are always masked.
