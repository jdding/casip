# Tmall Learned CASP Gate

- Model: `utility_hgb`
- Selected threshold: `0.15040953371686996`
- Feasible validation thresholds: `101`
- Unit of learning: one user-source-semantic-budget action row.

## Selected Result

| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.817 | 224 | 332 | 349 | 349 | 56 | 0 | 56 | 0.000 |
| test | 0.812 | 21305 | 24804 | 25916 | 26124 | 1074 | 168 | 906 | 0.156 |

## Interpretation

Learned utility_hgb selected a validation-feasible action threshold over user-source-budget rows. Test net@100=906, ratio=0.1564245810055866, gate_pass=True. This baseline tests whether a learned constrained gate can replace the transparent CASP rule.
