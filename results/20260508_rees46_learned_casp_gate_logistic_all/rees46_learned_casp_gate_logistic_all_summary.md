# REES46 Learned CASP Gate

- Model: `logistic`
- Selected threshold: `0.9433368429378198`
- Feasible validation thresholds: `30`
- Training target: validation promotion-action utility, with target-derived columns excluded from features.

## Selected Result

| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.680 | 3700 | 4162 | 4309 | 4606 | 38 | 0 | 38 | 0.000 |
| test | 0.628 | 3044 | 5341 | 6435 | 8845 | 83 | 40 | 43 | 0.482 |

## Interpretation

L2-logistic selected a validation-feasible score threshold from feature set all. Test net@100=43, ratio=0.4819277108433735, gate_pass=True. This is a regularized CASP solver for the fixed promotion action; transparent thresholds remain the interpretability anchor.

## Model Details

| Feature | Coef |
|---|---:|
| `top_bucket_specificity` | -1.1564 |
| `top_semid_share` | 0.7420 |
| `semantic_top50_n_semids` | 0.6647 |
| `log_n_semids` | -0.4717 |
| `semantic_top50_semid_entropy` | 0.4256 |
| `top_semid_margin` | 0.4069 |
| `log_top_bucket_size` | 0.3813 |
| `confidence_entropy` | -0.3678 |
| `semid_entropy` | 0.3393 |
| `top_semid_count` | 0.3170 |
| `confidence_specificity` | 0.2950 |
| `selected_existing_rank_median_fill` | 0.2388 |
| `inv_selected_existing_rank_min` | -0.2187 |
| `selected_semid_count` | -0.2144 |
| `semantic_only_share` | 0.1920 |
| `selected_semantic_only_count` | 0.1920 |
| `selected_tail_overlap_count` | -0.1920 |
| `tail_overlap_share` | -0.1920 |
| `selected_semantic_rank_mean` | 0.1840 |
| `selected_overlap_top500_count` | -0.1829 |
