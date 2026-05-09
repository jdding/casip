# REES46 Stage P List-Residual Audit

- Selected rule: `top_semid_share_ge_0.545455`
- Target: user-action delta Hit@100, not direct estimation of displaced-item purchase probability.

## Selected Result

| Split | Open rate | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.601 | 3700 | 4173 | 4320 | 4605 | 49 | 0 | 49 | 0.000 |
| test | 0.522 | 3044 | 5350 | 6438 | 8840 | 83 | 37 | 46 | 0.446 |

## Top Validation Rules

| Rule | Open rate | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|
| `selected_semantic_only_count_ge_0` | 1.000 | 62 | 2 | 60 | 0.032 |
| `selected_tail_overlap_count_ge_0` | 1.000 | 62 | 2 | 60 | 0.032 |
| `collision_top10_semid_le_1` | 1.000 | 62 | 2 | 60 | 0.032 |
| `collision_top100_semid_le_1` | 1.000 | 62 | 2 | 60 | 0.032 |
| `semantic_top50_n_semids_le_2` | 0.903 | 56 | 2 | 54 | 0.036 |
| `selected_existing_rank_mean_fill_ge_123` | 0.755 | 54 | 2 | 52 | 0.037 |
| `selected_rank_gap_mean_ge_113` | 0.755 | 54 | 2 | 52 | 0.037 |
| `top_semid_share_ge_0.545455` | 0.601 | 49 | 0 | 49 | 0.000 |
| `top_semid_share_ge_0.545455__selected_semantic_only_count_ge_0` | 0.601 | 49 | 0 | 49 | 0.000 |
| `top_semid_share_ge_0.545455__selected_tail_overlap_count_ge_0` | 0.601 | 49 | 0 | 49 | 0.000 |
| `top_semid_share_ge_0.545455__collision_top10_semid_le_1` | 0.601 | 49 | 0 | 49 | 0.000 |
| `top_semid_share_ge_0.545455__collision_top100_semid_le_1` | 0.601 | 49 | 0 | 49 | 0.000 |
| `top_semid_share_ge_0.545455__semantic_top50_n_semids_le_2` | 0.581 | 47 | 0 | 47 | 0.000 |
| `semantic_top50_semid_entropy_le_0.529361` | 0.751 | 48 | 2 | 46 | 0.042 |
| `top_semid_margin_ge_0.29044` | 0.600 | 45 | 0 | 45 | 0.000 |
| `top_semid_margin_ge_0.29044__selected_semantic_only_count_ge_0` | 0.600 | 45 | 0 | 45 | 0.000 |
| `top_semid_margin_ge_0.29044__selected_tail_overlap_count_ge_0` | 0.600 | 45 | 0 | 45 | 0.000 |
| `top_semid_margin_ge_0.29044__collision_top10_semid_le_1` | 0.600 | 45 | 0 | 45 | 0.000 |
| `top_semid_margin_ge_0.29044__collision_top100_semid_le_1` | 0.600 | 45 | 0 | 45 | 0.000 |
| `selected_semantic_rank_mean_le_21` | 0.757 | 46 | 2 | 44 | 0.043 |

## Interpretation

Selected `top_semid_share_ge_0.545455` over action/list residual features. Test net@100=46, ratio=0.4457831325301205, gate_pass=True. The supervised target is the realized promotion-action delta; displaced-item probability remains unobserved.
