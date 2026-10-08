# 35-gene baseline (reproduced from scratch)

Verification passed: **True** (see `verification.json`).

| Metric | CV mean ± SD | Locked test |
|---|---|---|
| accuracy | 0.880 ± 0.021 | 0.871 |
| balanced_accuracy | 0.747 ± 0.059 | 0.646 |
| precision | - | 0.700 |
| recall | 0.552 ± 0.127 | 0.318 |
| f1 | 0.591 ± 0.093 | 0.438 |
| roc_auc | 0.877 ± 0.032 | 0.826 |
| pr_auc | 0.689 ± 0.063 | 0.552 |
| mcc | 0.530 ± 0.098 | 0.413 |
| specificity | 0.943 ± 0.025 | 0.974 |

Majority-class accuracy: 0.840. Confusion matrix (locked test): TN 114, FP 3, FN 15, TP 7.