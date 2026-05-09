# REES46 Learned CASP Gate

- Model: `logistic`
- Selected threshold: `0.785851111804339`
- Feasible validation thresholds: `34`
- Training target: validation promotion-action utility, with target-derived columns excluded from features.

## Selected Result

| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.690 | 3627 | 4103 | 4223 | 4461 | 43 | 0 | 43 | 0.000 |
| test | 0.668 | 3044 | 5354 | 6424 | 8836 | 88 | 56 | 32 | 0.636 |

## Interpretation

L2-logistic selected a validation-feasible score threshold from feature set compact. Test net@100=32, ratio=0.6363636363636364, gate_pass=False. This is a regularized CASP solver for the fixed promotion action; transparent thresholds remain the interpretability anchor.

## Model Details

| Feature | Coef |
|---|---:|
| `log_n_semids` | -2.1736 |
| `top_bucket_specificity` | -0.9345 |
| `semantic_only_share` | -0.7489 |
| `top_semid_share` | -0.4607 |
| `semid_entropy` | 0.1668 |
| `top_semid_margin` | -0.0340 |
