# REES46 Learned CASP Gate

- Model: `two_head_hgb`
- Selected threshold: `0.1684673908949428`
- Feasible validation thresholds: `68`
- Training target: validation promotion-action utility, with target-derived columns excluded from features.

## Selected Result

| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.420 | 3700 | 4189 | 4332 | 4613 | 61 | 0 | 61 | 0.000 |
| test | 0.416 | 3044 | 5337 | 6417 | 8837 | 64 | 39 | 25 | 0.609 |

## Interpretation

Learned two_head_hgb selected a validation-feasible score threshold. Test net@100=25, ratio=0.609375, gate_pass=False. This is a learned constrained baseline, not the main transparent CASP solver.
