# REES46 Semantic-ID Bridge Audit

- Artifact: `results/20260505_rees46_protocol_parallel_all_protocol/rees46_protocol_artifact.pkl`
- Existing per-user file: `results/20260505_rees46_purchase_event_gate_local/rees46_validation_safe_gate_test_per_user.csv`
- Semantic schemes: `category, brand, code_family, category_brand, code_brand`
- Source scopes: `history, context, history_context`
- Source topK per semantic bucket: `500`
- Intra-ID ranking: `val_purchase_popularity -> val_implicit_popularity -> val_all_popularity -> pre_all_popularity within semantic bucket`

## Support Gate Summary

| Group | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Delta@100 vs existing | Delta@500 vs existing |
|---|---:|---:|---:|---:|---:|---:|
| `existing_union` | 3896 | 5810 | 6708 | 9080 | 0 | 0 |
| `semantic_all_oracle` | 2584 | 4285 | 5102 | 6624 | -1606 | -2456 |
| `existing_plus_semantic_all` | 5043 | 7342 | 8351 | 10672 | 1643 | 1592 |

## Best Semantic Bridge Sources

| Source | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Existing miss -> hit@100 | Existing miss -> hit@500 | Mean candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| `semid_category_context` | 860 | 1602 | 1973 | 2677 | 484 | 490 | 1308.6 |
| `semid_brand_history_context` | 997 | 1546 | 1817 | 2464 | 474 | 396 | 1913.8 |
| `semid_brand_context` | 909 | 1408 | 1635 | 2081 | 448 | 359 | 1008.2 |
| `semid_category_history_context` | 872 | 1552 | 1892 | 2709 | 438 | 479 | 2304.8 |
| `semid_category_brand_history_context` | 670 | 1138 | 1354 | 2200 | 405 | 400 | 582.6 |
| `semid_category_brand_context` | 677 | 1101 | 1255 | 1707 | 392 | 320 | 283.6 |
| `semid_code_brand_history_context` | 834 | 1340 | 1546 | 2268 | 390 | 281 | 920.2 |
| `semid_code_family_history_context` | 1132 | 2174 | 2792 | 4126 | 373 | 521 | 1270.3 |
| `semid_code_brand_context` | 735 | 1197 | 1371 | 1816 | 362 | 249 | 502.4 |
| `semid_brand_history` | 786 | 1215 | 1391 | 1793 | 352 | 250 | 1137.3 |
| `semid_code_family_context` | 1042 | 1982 | 2554 | 3717 | 350 | 479 | 886.5 |
| `semid_category_history` | 707 | 1257 | 1502 | 1984 | 314 | 300 | 1241.4 |
| `semid_code_family_history` | 941 | 1796 | 2300 | 3317 | 293 | 422 | 695.9 |
| `semid_code_brand_history` | 611 | 966 | 1094 | 1454 | 254 | 150 | 530.1 |
| `semid_category_brand_history` | 469 | 759 | 870 | 1197 | 227 | 175 | 346.9 |

## Decision

PASS support gate on `semid_category_context`: existing-miss gains @100=484, @500=490.

## Interpretation

This is a no-training support audit only. A pass means the Semantic-ID bridge changes candidate reachability enough to justify a calibrated bridge model. A fail means the tested semantic IDs mostly reshuffle support already covered by popularity/replay/purchase-event sources.
