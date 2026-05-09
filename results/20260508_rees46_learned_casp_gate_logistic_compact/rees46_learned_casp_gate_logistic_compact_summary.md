# REES46 Learned CASP Gate

- Model: `logistic`
- Selected threshold: `0.8626842054052231`
- Feasible validation thresholds: `30`
- Training target: validation promotion-action utility, with target-derived columns excluded from features.

## Selected Result

| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.700 | 3700 | 4169 | 4316 | 4605 | 45 | 0 | 45 | 0.000 |
| test | 0.640 | 3044 | 5347 | 6446 | 8851 | 97 | 43 | 54 | 0.443 |

## Interpretation

L2-logistic selected a validation-feasible score threshold from feature set compact. Test net@100=54, ratio=0.44329896907216493, gate_pass=True. This is a regularized CASP solver for the fixed promotion action; transparent thresholds remain the interpretability anchor.

## Model Details

| Feature | Coef |
|---|---:|
| `semantic_only_share` | 1.3445 |
| `top_bucket_specificity` | -1.2778 |
| `top_semid_share` | 0.9225 |
| `semid_entropy` | 0.7178 |
| `top_semid_margin` | 0.5527 |
| `log_n_semids` | -0.4634 |
