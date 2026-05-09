# REES46 Stage P Interaction Rule Grid

- Selected rule: `top_semid_margin_ge_0.0714286__n_context_le_17`
- Selection constraints: validation ratio <= `0.0`, net@100 >= `20`, open rate <= `0.7`
- Rule class: semantic-confidence condition plus optional one user-state/candidate condition.

## Selected Result

| Split | Open rate | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.618 | 3700 | 4177 | 4323 | 4607 | 52 | 0 | 52 | 0.000 |
| test | 0.541 | 3044 | 5347 | 6430 | 8841 | 81 | 43 | 38 | 0.531 |

## Top Validation Rules

| Rule | Open rate | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|
| `top_semid_share_ge_0.35` | 0.801 | 57 | 1 | 56 | 0.018 |
| `top_semid_margin_ge_0.0714286` | 0.801 | 55 | 0 | 55 | 0.000 |
| `top_semid_share_ge_0.35__n_context_le_17` | 0.649 | 54 | 1 | 53 | 0.019 |
| `top_semid_margin_ge_0.0714286__n_context_le_17` | 0.618 | 52 | 0 | 52 | 0.000 |
| `confidence_entropy_ge_0.0179116` | 0.800 | 52 | 0 | 52 | 0.000 |
| `top_semid_share_ge_0.35__semantic_candidate_count_le_341` | 0.643 | 53 | 1 | 52 | 0.019 |
| `top_semid_margin_ge_0.0714286__semantic_candidate_count_le_341` | 0.618 | 51 | 0 | 51 | 0.000 |
| `top_semid_margin_ge_0.166667` | 0.708 | 51 | 0 | 51 | 0.000 |
| `top_semid_share_ge_0.473684` | 0.701 | 52 | 1 | 51 | 0.019 |
| `semid_entropy_le_0.919611` | 0.800 | 50 | 0 | 50 | 0.000 |
| `confidence_entropy_ge_0.0397033` | 0.700 | 50 | 0 | 50 | 0.000 |
| `top_semid_share_ge_0.545455` | 0.601 | 49 | 0 | 49 | 0.000 |
| `confidence_entropy_ge_0.0179116__n_context_le_17` | 0.591 | 49 | 0 | 49 | 0.000 |
| `confidence_entropy_ge_0.0179116__semantic_candidate_count_le_341` | 0.605 | 49 | 0 | 49 | 0.000 |
| `top_semid_share_ge_0.35__n_history_ge_3` | 0.646 | 48 | 0 | 48 | 0.000 |
| `top_semid_margin_ge_0.166667__n_context_le_17` | 0.574 | 48 | 0 | 48 | 0.000 |
| `top_semid_share_ge_0.473684__n_context_le_17` | 0.599 | 49 | 1 | 48 | 0.020 |
| `top_semid_share_ge_0.473684__semantic_candidate_count_le_341` | 0.590 | 49 | 1 | 48 | 0.020 |
| `top_semid_margin_ge_0.166667__semantic_candidate_count_le_341` | 0.570 | 47 | 0 | 47 | 0.000 |
| `semid_entropy_le_0.919611__n_context_le_17` | 0.596 | 47 | 0 | 47 | 0.000 |

## Interpretation

Selected `top_semid_margin_ge_0.0714286__n_context_le_17`. Test net@100=38, ratio=0.5308641975308642, gate_pass=False. This is the transparent interaction-rule P-B variant; compare against P-A and learned LR/tree.
