# REES46 Learned CASP Gate

- Model: `logistic`
- Selected threshold: `0.7510103863067825`
- Feasible validation thresholds: `21`
- Training target: validation promotion-action utility, with target-derived columns excluded from features.

## Selected Result

| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.588 | 3700 | 4173 | 4320 | 4605 | 49 | 0 | 49 | 0.000 |
| test | 0.508 | 3044 | 5348 | 6438 | 8838 | 80 | 34 | 46 | 0.425 |

## Interpretation

L2-logistic selected a validation-feasible score threshold from feature set share_only. Test net@100=46, ratio=0.425, gate_pass=True. This is a regularized CASP solver for the fixed promotion action; transparent thresholds remain the interpretability anchor.

## Model Details

| Feature | Coef |
|---|---:|
| `top_semid_share` | 1.3015 |
