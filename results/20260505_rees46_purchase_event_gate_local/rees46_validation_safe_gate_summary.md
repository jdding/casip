# REES46 Validation-Safe Gate

- Artifact: `<repo-root>/results/20260505_rees46_protocol_parallel_all_protocol/rees46_protocol_artifact.pkl`
- Users: validation `5689`, test `16954`
- Source top-k per component: `500`
- Selected by validation Hit@10: `global_purchase_popularity_0.025__context_event_weighted_replay_0.975`
- Selected weights: `{'global_purchase_popularity': 0.025, 'context_event_weighted_replay': 0.975}`

## Selected Validation-To-Test Result

| Split | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| validation | 3700 | 4128 | 4271 | 4588 |
| test | 3044 | 5319 | 6392 | 8816 |

## Top Validation Configs

| Config | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| `global_purchase_popularity_0.025__context_event_weighted_replay_0.975` | 3700 | 4128 | 4271 | 4588 |
| `global_purchase_popularity_0.050__context_event_weighted_replay_0.950` | 3700 | 4128 | 4271 | 4588 |
| `global_purchase_popularity_0.075__context_event_weighted_replay_0.925` | 3700 | 4128 | 4271 | 4588 |
| `global_purchase_popularity_0.175__context_event_weighted_replay_0.825` | 3688 | 4130 | 4271 | 4588 |
| `global_purchase_popularity_0.100__context_event_weighted_replay_0.900` | 3688 | 4129 | 4271 | 4588 |
| `global_purchase_popularity_0.125__context_event_weighted_replay_0.875` | 3688 | 4129 | 4271 | 4588 |
| `global_purchase_popularity_0.150__context_event_weighted_replay_0.850` | 3688 | 4129 | 4271 | 4588 |
| `global_purchase_popularity_0.200__context_event_weighted_replay_0.800` | 3668 | 4128 | 4271 | 4588 |
| `global_purchase_popularity_0.250__context_event_weighted_replay_0.750` | 3668 | 4128 | 4271 | 4588 |
| `global_purchase_popularity_0.225__context_event_weighted_replay_0.775` | 3668 | 4127 | 4271 | 4588 |
| `global_popularity_0.025__context_feedback_replay_0.975` | 3660 | 4131 | 4249 | 4561 |
| `global_popularity_0.050__context_feedback_replay_0.950` | 3660 | 4131 | 4249 | 4561 |
| `global_popularity_0.075__context_feedback_replay_0.925` | 3660 | 4131 | 4249 | 4561 |
| `global_purchase_popularity_0.275__context_event_weighted_replay_0.725` | 3640 | 4128 | 4271 | 4588 |
| `global_popularity_0.3__context_event_weighted_replay_0.7` | 3637 | 4131 | 4249 | 4561 |

## Top Test Configs (Oracle Diagnostic Only)

These rows are sorted with test labels and are not deployable selection evidence.

| Config | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| `global_purchase_popularity_0.3__context_event_weighted_replay_0.7` | 3107 | 5333 | 6388 | 8816 |
| `global_purchase_popularity_0.300__context_event_weighted_replay_0.700` | 3107 | 5333 | 6388 | 8816 |
| `global_purchase_popularity_0.325__context_event_weighted_replay_0.675` | 3105 | 5334 | 6388 | 8816 |
| `global_purchase_popularity_0.275__context_event_weighted_replay_0.725` | 3104 | 5333 | 6388 | 8816 |
| `global_purchase_popularity_0.350__context_event_weighted_replay_0.650` | 3103 | 5337 | 6388 | 8816 |
| `global_purchase_popularity_0.575__context_event_weighted_replay_0.425` | 3102 | 5327 | 6385 | 8816 |
| `global_purchase_popularity_0.550__context_event_weighted_replay_0.450` | 3101 | 5328 | 6389 | 8816 |
| `global_purchase_popularity_0.600__context_event_weighted_replay_0.400` | 3100 | 5322 | 6386 | 8816 |
| `global_purchase_popularity_0.625__context_event_weighted_replay_0.375` | 3100 | 5317 | 6384 | 8816 |
| `global_purchase_popularity_0.425__context_event_weighted_replay_0.575` | 3095 | 5336 | 6386 | 8816 |
| `global_purchase_popularity_0.400__context_event_weighted_replay_0.600` | 3095 | 5335 | 6387 | 8816 |
| `global_purchase_popularity_0.650__context_event_weighted_replay_0.350` | 3095 | 5315 | 6381 | 8817 |
| `global_purchase_popularity_0.375__context_event_weighted_replay_0.625` | 3093 | 5337 | 6388 | 8816 |
| `global_purchase_popularity_0.525__context_event_weighted_replay_0.475` | 3092 | 5329 | 6389 | 8816 |
| `global_purchase_popularity_0.450__context_event_weighted_replay_0.550` | 3088 | 5334 | 6385 | 8816 |
