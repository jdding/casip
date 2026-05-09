# Tmall Learned CASP Gate

- Model: `two_head_hgb`
- Selected threshold: `0.17163885110572516`
- Feasible validation thresholds: `101`
- Unit of learning: one user-source-semantic-budget action row.

## Selected Result

| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.882 | 225 | 315 | 350 | 350 | 57 | 0 | 57 | 0.000 |
| test | 0.862 | 21304 | 24263 | 25984 | 26030 | 995 | 21 | 974 | 0.021 |

## Interpretation

Learned two_head_hgb selected a validation-feasible action threshold over user-source-budget rows. Test net@100=974, ratio=0.021105527638190954, gate_pass=True. This baseline tests whether a learned constrained gate can replace the transparent CASP rule.
