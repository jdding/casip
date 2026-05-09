# REES46 Validation-Safe Gate

- Artifact: `<external-workspace>`
- Users: validation `14776`, test `16954`
- Source top-k per component: `500`
- Selected by validation Hit@10: `global_popularity_0.3__context_feedback_replay_0.7`
- Selected weights: `{'global_popularity': 0.3, 'context_feedback_replay': 0.7}`

## Selected Validation-To-Test Result

| Split | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| validation | 9053 | 10245 | 10594 | 11587 |
| test | 2910 | 5211 | 6199 | 8517 |

## Top Validation Configs

| Config | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| `global_popularity_0.3__context_feedback_replay_0.7` | 9053 | 10245 | 10594 | 11587 |
| `global_popularity_0.5__context_feedback_replay_0.5` | 8740 | 10229 | 10594 | 11587 |
| `context_feedback_replay` | 8686 | 8946 | 8948 | 8948 |
| `global_popularity__context_feedback_replay__category_or_brand_replacement_unseen__equal` | 8110 | 10286 | 10683 | 11744 |
| `seen_product_replay` | 8038 | 9368 | 9413 | 9417 |
| `global_popularity_0.7__context_feedback_replay_0.3` | 8028 | 10194 | 10582 | 11587 |
| `global_popularity_0.3__seen_product_replay_0.7` | 7754 | 10332 | 10785 | 11741 |
| `seen_product_replay_0.7__brand_replacement_unseen_0.3` | 7696 | 10354 | 10754 | 11379 |
| `seen_product_replay_0.7__category_or_brand_replacement_unseen_0.3` | 7658 | 10318 | 10750 | 11491 |
| `global_popularity_0.5__seen_product_replay_0.5` | 7023 | 10223 | 10767 | 11741 |
| `seen_product_replay_0.5__brand_replacement_unseen_0.5` | 6994 | 10242 | 10730 | 11379 |
| `seen_product_replay_0.5__category_or_brand_replacement_unseen_0.5` | 6942 | 10204 | 10726 | 11491 |
| `seen_product_replay__brand_overlap_seen_allowed__category_or_brand_replacement_unseen__equal` | 6102 | 10021 | 10713 | 11522 |
| `global_popularity__seen_product_replay__brand_replacement_unseen__equal` | 6022 | 10131 | 10843 | 11908 |
| `global_popularity_0.7__seen_product_replay_0.3` | 5856 | 9838 | 10674 | 11740 |

## Top Test Configs (Oracle Diagnostic Only)

These rows are sorted with test labels and are not deployable selection evidence.

| Config | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| `global_popularity_0.5__context_feedback_replay_0.5` | 2934 | 5208 | 6198 | 8517 |
| `global_popularity_0.3__context_feedback_replay_0.7` | 2910 | 5211 | 6199 | 8517 |
| `global_popularity_0.7__context_feedback_replay_0.3` | 2849 | 5158 | 6184 | 8517 |
| `global_popularity_0.5__seen_product_replay_0.5` | 2730 | 5185 | 6252 | 8596 |
| `global_popularity_0.7__seen_product_replay_0.3` | 2717 | 5137 | 6223 | 8595 |
| `global_popularity__context_feedback_replay__category_or_brand_replacement_unseen__equal` | 2642 | 5073 | 6075 | 8560 |
| `seen_product_replay_0.5__category_or_brand_replacement_unseen_0.5` | 2579 | 4523 | 5273 | 6745 |
| `global_popularity_0.3__seen_product_replay_0.7` | 2575 | 5194 | 6253 | 8596 |
| `seen_product_replay_0.7__brand_replacement_unseen_0.3` | 2571 | 4299 | 4903 | 5796 |
| `seen_product_replay_0.5__brand_replacement_unseen_0.5` | 2557 | 4268 | 4894 | 5796 |
| `seen_product_replay_0.7__category_or_brand_replacement_unseen_0.3` | 2552 | 4552 | 5275 | 6745 |
| `global_popularity__seen_product_replay__brand_replacement_unseen__equal` | 2532 | 5080 | 6088 | 8484 |
| `seen_product_replay__brand_overlap_seen_allowed__category_or_brand_replacement_unseen__equal` | 2500 | 4497 | 5259 | 6764 |
| `seen_product_replay_0.3__category_or_brand_replacement_unseen_0.7` | 2425 | 4453 | 5228 | 6743 |
| `global_popularity` | 2394 | 4320 | 5366 | 7936 |
