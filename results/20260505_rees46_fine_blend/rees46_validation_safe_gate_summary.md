# REES46 Validation-Safe Gate

- Artifact: `<external-workspace>`
- Users: validation `14776`, test `16954`
- Source top-k per component: `500`
- Selected by validation Hit@10: `global_popularity_0.025__context_feedback_replay_0.975`
- Selected weights: `{'global_popularity': 0.025, 'context_feedback_replay': 0.975}`

## Selected Validation-To-Test Result

| Split | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| validation | 9206 | 10249 | 10595 | 11587 |
| test | 2933 | 5203 | 6204 | 8517 |

## Top Validation Configs

| Config | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| `global_popularity_0.025__context_feedback_replay_0.975` | 9206 | 10249 | 10595 | 11587 |
| `global_popularity_0.050__context_feedback_replay_0.950` | 9206 | 10249 | 10595 | 11587 |
| `global_popularity_0.075__context_feedback_replay_0.925` | 9206 | 10249 | 10595 | 11587 |
| `global_popularity_0.100__context_feedback_replay_0.900` | 9171 | 10249 | 10595 | 11587 |
| `global_popularity_0.125__context_feedback_replay_0.875` | 9171 | 10248 | 10595 | 11587 |
| `global_popularity_0.150__context_feedback_replay_0.850` | 9171 | 10248 | 10595 | 11587 |
| `global_popularity_0.175__context_feedback_replay_0.825` | 9168 | 10248 | 10595 | 11587 |
| `global_popularity_0.200__context_feedback_replay_0.800` | 9114 | 10248 | 10595 | 11587 |
| `global_popularity_0.225__context_feedback_replay_0.775` | 9114 | 10248 | 10595 | 11587 |
| `global_popularity_0.250__context_feedback_replay_0.750` | 9109 | 10246 | 10595 | 11587 |
| `global_popularity_0.275__context_feedback_replay_0.725` | 9054 | 10246 | 10595 | 11587 |
| `global_popularity_0.3__context_feedback_replay_0.7` | 9053 | 10245 | 10594 | 11587 |
| `global_popularity_0.300__context_feedback_replay_0.700` | 9053 | 10245 | 10594 | 11587 |
| `global_popularity_0.325__context_feedback_replay_0.675` | 9049 | 10244 | 10594 | 11587 |
| `global_popularity_0.350__context_feedback_replay_0.650` | 9042 | 10244 | 10594 | 11587 |

## Top Test Configs (Oracle Diagnostic Only)

These rows are sorted with test labels and are not deployable selection evidence.

| Config | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| `global_popularity_0.475__context_feedback_replay_0.525` | 2935 | 5207 | 6198 | 8517 |
| `global_popularity_0.5__context_feedback_replay_0.5` | 2934 | 5208 | 6198 | 8517 |
| `global_popularity_0.500__context_feedback_replay_0.500` | 2934 | 5208 | 6198 | 8517 |
| `global_popularity_0.075__context_feedback_replay_0.925` | 2933 | 5209 | 6204 | 8517 |
| `global_popularity_0.025__context_feedback_replay_0.975` | 2933 | 5203 | 6204 | 8517 |
| `global_popularity_0.050__context_feedback_replay_0.950` | 2933 | 5203 | 6204 | 8517 |
| `global_popularity_0.550__context_feedback_replay_0.450` | 2932 | 5200 | 6196 | 8517 |
| `global_popularity_0.150__context_feedback_replay_0.850` | 2929 | 5211 | 6203 | 8517 |
| `global_popularity_0.175__context_feedback_replay_0.825` | 2929 | 5211 | 6203 | 8517 |
| `global_popularity_0.100__context_feedback_replay_0.900` | 2929 | 5210 | 6204 | 8517 |
| `global_popularity_0.125__context_feedback_replay_0.875` | 2929 | 5210 | 6204 | 8517 |
| `global_popularity_0.525__context_feedback_replay_0.475` | 2928 | 5204 | 6197 | 8517 |
| `global_popularity_0.600__context_feedback_replay_0.400` | 2922 | 5185 | 6194 | 8516 |
| `global_popularity_0.250__context_feedback_replay_0.750` | 2921 | 5214 | 6200 | 8517 |
| `global_popularity_0.575__context_feedback_replay_0.425` | 2921 | 5194 | 6195 | 8517 |
