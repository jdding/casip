# REES46 Learned CASP Gate

- Model: `utility_hgb`
- Selected threshold: `0.17620118078362795`
- Feasible validation thresholds: `70`
- Training target: validation promotion-action utility, with target-derived columns excluded from features.

## Selected Result

| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.400 | 3700 | 4189 | 4333 | 4614 | 62 | 0 | 62 | 0.000 |
| test | 0.449 | 3044 | 5338 | 6421 | 8835 | 69 | 40 | 29 | 0.580 |

## Interpretation

Learned utility_hgb selected a validation-feasible score threshold. Test net@100=29, ratio=0.5797101449275363, gate_pass=False. This is a learned constrained baseline, not the main transparent CASP solver.
